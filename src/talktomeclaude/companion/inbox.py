"""Durable reply-effect recovery and bounded background ownership.

The reply receiver makes an upstream ACK eligible as soon as the canonical
event is durable.  The companion has a later durability boundary: it must not
forget a reply until the runtime has accepted its local effect.  This module
joins those boundaries without putting answer content in lifecycle status.
"""

from __future__ import annotations

import math
import threading
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Protocol

from talktomeclaude.reply import (
    AckDisposition,
    ReplyAck,
    ReplyEvent,
    ReplyReceiver,
    ReplySpool,
)
from talktomeclaude.reply.contracts import AckResult, ReceiveResult
from talktomeclaude.reply.spool import SpoolRecord
from talktomeclaude.reply.ssh import TransportResult


class InboxStatusCode(str, Enum):
    """Content-free observations from inbox and transport ownership."""

    STARTED = "started"
    RECOVERY_FOUND = "recovery_found"
    REPLY_RECEIVED = "reply_received"
    REPLY_ACCEPTED = "reply_accepted"
    REPLY_DEFERRED = "reply_deferred"
    CONSUMED_COMMITTED = "consumed_committed"
    ACK_COMMITTED = "ack_committed"
    DURABLE_FAULT = "durable_fault"
    TRANSPORT_STARTED = "transport_started"
    TRANSPORT_FAULT = "transport_fault"
    SHUTDOWN_TIMEOUT = "shutdown_timeout"
    STOPPED = "stopped"


@dataclass(frozen=True, slots=True)
class InboxStatus:
    """One answer-free lifecycle observation."""

    code: InboxStatusCode
    count: int | None = None


@dataclass(frozen=True, slots=True)
class InboxDrainResult:
    recovered: int = 0
    received: int = 0
    accepted: int = 0
    consumed: int = 0
    acknowledged: int = 0
    deferred: int = 0
    faults: int = 0


@dataclass(frozen=True, slots=True)
class OwnerStopResult:
    stopped: bool
    boundary_replacement_required: bool


class SpoolBoundary(Protocol):
    def pending(self) -> tuple[SpoolRecord, ...]: ...

    def commit_ack(self, ack: ReplyAck) -> AckResult: ...


class ReceiverBoundary(Protocol):
    def receive(self, wire_bytes: bytes) -> ReceiveResult: ...

    def pending_committed(self) -> tuple[ReplyEvent, ...]: ...

    def commit_consumed(self, ack: ReplyAck) -> bool: ...


class TransportRunner(Protocol):
    def run(self, stop: threading.Event) -> TransportResult: ...


ThreadFactory = Callable[..., threading.Thread]
StatusObserver = Callable[[InboxStatus], object]
Sleep = Callable[[float], object]
ReplyEffect = Callable[[ReplyEvent], bool]


class DurableReplyInbox:
    """Apply local reply effects before consuming and acknowledging events.

    ``drain_once`` first recovers receiver commits that survived a prior owner,
    then drains the local spool.  If a prior process committed ``consumed`` but
    crashed before the spool ACK, replay advances only the ACK and never calls
    ``on_reply`` again.
    """

    def __init__(
        self,
        spool: SpoolBoundary | ReplySpool,
        receiver: ReceiverBoundary | ReplyReceiver,
        on_reply: ReplyEffect,
        *,
        poll_interval_seconds: float = 0.1,
        shutdown_timeout_seconds: float = 2.0,
        on_status: StatusObserver | None = None,
        thread_factory: ThreadFactory = threading.Thread,
        sleep: Sleep | None = None,
    ) -> None:
        if (
            not math.isfinite(poll_interval_seconds)
            or poll_interval_seconds <= 0
            or poll_interval_seconds > 10
        ):
            raise ValueError("poll interval must be in (0, 10] seconds")
        if (
            not math.isfinite(shutdown_timeout_seconds)
            or shutdown_timeout_seconds < 0
            or shutdown_timeout_seconds > 30
        ):
            raise ValueError("shutdown timeout must be in [0, 30] seconds")
        self._spool = spool
        self._receiver = receiver
        self._on_reply = on_reply
        self._poll_interval = poll_interval_seconds
        self._shutdown_timeout = shutdown_timeout_seconds
        self._on_status = on_status
        self._thread_factory = thread_factory
        self._sleep = sleep
        self._guard = threading.RLock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _emit(self, code: InboxStatusCode, *, count: int | None = None) -> None:
        if self._on_status is None:
            return
        try:
            self._on_status(InboxStatus(code, count))
        except Exception:
            pass

    def _accept(self, event: ReplyEvent) -> bool:
        try:
            accepted = self._on_reply(event) is True
        except Exception:
            accepted = False
        if not accepted:
            self._emit(InboxStatusCode.REPLY_DEFERRED)
            return False
        self._emit(InboxStatusCode.REPLY_ACCEPTED)
        return True

    def _consume(self, event: ReplyEvent) -> bool:
        if not self._accept(event):
            return False
        try:
            committed = self._receiver.commit_consumed(ReplyAck.for_event(event))
        except Exception:
            committed = False
        if not committed:
            self._emit(InboxStatusCode.DURABLE_FAULT)
            return False
        self._emit(InboxStatusCode.CONSUMED_COMMITTED)
        return True

    def drain_once(self) -> InboxDrainResult:
        """Process one deterministic recovery and spool snapshot."""

        recovered = received = accepted = consumed = acknowledged = deferred = (
            faults
        ) = 0
        deferred_identities: set[tuple[str, str]] = set()

        try:
            pending_recovery = self._receiver.pending_committed()
        except Exception:
            self._emit(InboxStatusCode.DURABLE_FAULT)
            return InboxDrainResult(faults=1)

        if pending_recovery:
            self._emit(InboxStatusCode.RECOVERY_FOUND, count=len(pending_recovery))
        for event in pending_recovery:
            recovered += 1
            if self._consume(event):
                accepted += 1
                consumed += 1
            else:
                deferred += 1
                deferred_identities.add((event.event_id, event.digest))

        try:
            records = self._spool.pending()
        except Exception:
            self._emit(InboxStatusCode.DURABLE_FAULT)
            return InboxDrainResult(
                recovered,
                received,
                accepted,
                consumed,
                acknowledged,
                deferred,
                faults + 1,
            )

        for record in records:
            identity = (record.event.event_id, record.event.digest)
            try:
                result = self._receiver.receive(record.wire_bytes)
            except Exception:
                faults += 1
                self._emit(InboxStatusCode.DURABLE_FAULT)
                continue
            ack = result.ack
            if (
                ack is None
                or ack.event_id != record.event.event_id
                or ack.digest != record.event.digest
            ):
                faults += 1
                self._emit(InboxStatusCode.DURABLE_FAULT)
                continue
            received += 1
            self._emit(InboxStatusCode.REPLY_RECEIVED)

            needs_effect = result.apply
            if not needs_effect and identity not in deferred_identities:
                try:
                    needs_effect = any(
                        item.event_id == ack.event_id and item.digest == ack.digest
                        for item in self._receiver.pending_committed()
                    )
                except Exception:
                    faults += 1
                    self._emit(InboxStatusCode.DURABLE_FAULT)
                    continue

            if needs_effect:
                if identity in deferred_identities:
                    continue
                if not self._consume(record.event):
                    deferred += 1
                    deferred_identities.add(identity)
                    continue
                accepted += 1
                consumed += 1

            try:
                ack_result = self._spool.commit_ack(ack)
            except Exception:
                faults += 1
                self._emit(InboxStatusCode.DURABLE_FAULT)
                continue
            if ack_result.disposition not in {
                AckDisposition.COMMITTED,
                AckDisposition.ALREADY_COMMITTED,
            }:
                faults += 1
                self._emit(InboxStatusCode.DURABLE_FAULT)
                continue
            acknowledged += 1
            self._emit(InboxStatusCode.ACK_COMMITTED)

        return InboxDrainResult(
            recovered,
            received,
            accepted,
            consumed,
            acknowledged,
            deferred,
            faults,
        )

    def _wait(self) -> bool:
        if self._sleep is None:
            return self._stop.wait(self._poll_interval)
        self._sleep(self._poll_interval)
        return self._stop.is_set()

    def _run(self) -> None:
        self._emit(InboxStatusCode.STARTED)
        try:
            while not self._stop.is_set():
                try:
                    self.drain_once()
                except Exception:
                    self._emit(InboxStatusCode.DURABLE_FAULT)
                if self._wait():
                    break
        finally:
            self._emit(InboxStatusCode.STOPPED)

    def start(self) -> bool:
        """Start one daemon owner; return False when it is already running."""

        with self._guard:
            if self._thread is not None and self._thread.is_alive():
                return False
            self._stop = threading.Event()
            thread = self._thread_factory(
                target=self._run,
                name="talktomeclaude-reply-inbox",
                daemon=True,
            )
            self._thread = thread
            thread.start()
            return True

    def stop(self, timeout_seconds: float | None = None) -> OwnerStopResult:
        """Request shutdown and wait no longer than the configured bound."""

        timeout = self._shutdown_timeout if timeout_seconds is None else timeout_seconds
        if not math.isfinite(timeout) or timeout < 0 or timeout > 30:
            raise ValueError("shutdown timeout must be in [0, 30] seconds")
        with self._guard:
            thread = self._thread
            self._stop.set()
        if thread is None:
            return OwnerStopResult(True, False)
        if thread is threading.current_thread():
            self._emit(InboxStatusCode.SHUTDOWN_TIMEOUT)
            return OwnerStopResult(False, True)
        thread.join(timeout)
        stopped = not thread.is_alive()
        if stopped:
            with self._guard:
                if self._thread is thread:
                    self._thread = None
        else:
            self._emit(InboxStatusCode.SHUTDOWN_TIMEOUT)
        return OwnerStopResult(stopped, not stopped)


class SSHTransportOwner:
    """Bounded thread owner for ``PersistentSSHReplyTransport``."""

    def __init__(
        self,
        transport: TransportRunner,
        *,
        shutdown_timeout_seconds: float = 3.0,
        on_status: StatusObserver | None = None,
        thread_factory: ThreadFactory = threading.Thread,
    ) -> None:
        if (
            not math.isfinite(shutdown_timeout_seconds)
            or shutdown_timeout_seconds < 0
            or shutdown_timeout_seconds > 30
        ):
            raise ValueError("shutdown timeout must be in [0, 30] seconds")
        self._transport = transport
        self._shutdown_timeout = shutdown_timeout_seconds
        self._on_status = on_status
        self._thread_factory = thread_factory
        self._guard = threading.RLock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._result: TransportResult | None = None
        self._failed = False

    @property
    def result(self) -> TransportResult | None:
        with self._guard:
            return self._result

    def _emit(self, code: InboxStatusCode) -> None:
        if self._on_status is None:
            return
        try:
            self._on_status(InboxStatus(code))
        except Exception:
            pass

    def _run(self) -> None:
        self._emit(InboxStatusCode.TRANSPORT_STARTED)
        try:
            result = self._transport.run(self._stop)
            with self._guard:
                self._result = result
                self._failed = result.boundary_replacement_required
            if result.boundary_replacement_required:
                self._emit(InboxStatusCode.TRANSPORT_FAULT)
        except Exception:
            with self._guard:
                self._failed = True
            self._emit(InboxStatusCode.TRANSPORT_FAULT)
        finally:
            self._emit(InboxStatusCode.STOPPED)

    def start(self) -> bool:
        with self._guard:
            if self._thread is not None and self._thread.is_alive():
                return False
            self._stop = threading.Event()
            self._result = None
            self._failed = False
            thread = self._thread_factory(
                target=self._run,
                name="talktomeclaude-reply-ssh",
                daemon=True,
            )
            self._thread = thread
            thread.start()
            return True

    def stop(self, timeout_seconds: float | None = None) -> OwnerStopResult:
        timeout = self._shutdown_timeout if timeout_seconds is None else timeout_seconds
        if not math.isfinite(timeout) or timeout < 0 or timeout > 30:
            raise ValueError("shutdown timeout must be in [0, 30] seconds")
        with self._guard:
            thread = self._thread
            self._stop.set()
        if thread is None:
            return OwnerStopResult(True, self._failed)
        if thread is threading.current_thread():
            self._emit(InboxStatusCode.SHUTDOWN_TIMEOUT)
            return OwnerStopResult(False, True)
        thread.join(timeout)
        stopped = not thread.is_alive()
        with self._guard:
            failed = self._failed
            if stopped and self._thread is thread:
                self._thread = None
        if not stopped:
            self._emit(InboxStatusCode.SHUTDOWN_TIMEOUT)
        return OwnerStopResult(stopped, failed or not stopped)


__all__ = [
    "DurableReplyInbox",
    "InboxDrainResult",
    "InboxStatus",
    "InboxStatusCode",
    "OwnerStopResult",
    "SSHTransportOwner",
]
