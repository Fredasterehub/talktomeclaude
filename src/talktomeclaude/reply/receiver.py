"""Durable local commit and dedupe boundary for reply events."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TypeVar

from talktomeclaude.storage.atomic import AtomicJsonTransaction

from .contracts import (
    DiagnosticCode,
    ReceiveDisposition,
    ReceiveResult,
    ReplyAck,
    ReplyDiagnostic,
    ReplyEvent,
    ReplyProtocolError,
)
from .spool import (
    DurabilityCapabilities,
    SpoolDurabilityError,
    _fsync_directory,
    probe_durability,
)

_Result = TypeVar("_Result")


@dataclass(frozen=True, slots=True)
class ReceiverRetentionResult:
    """Result of receiver-owned, content-safe housekeeping."""

    quarantine_removed: int
    stale_temps_removed: int
    canonical_preserved: int
    pending_preserved: int


class ReplyReceiver:
    """Validate, durably commit, and only then make an ACK eligible.

    The canonical event and its small dedupe record are separate immutable
    files.  A crash between them is recovered by replay: the exact canonical
    event is found, the missing dedupe record is committed, and ``apply`` is
    true exactly once.  Once the dedupe record exists, replay returns an ACK
    with ``apply=False``.
    """

    def __init__(
        self,
        root: str | os.PathLike[str],
        *,
        phase_hook: Callable[[str], None] | None = None,
        on_diagnostic: Callable[[ReplyDiagnostic], None] | None = None,
        id_factory: Callable[[], str] | None = None,
        capabilities: DurabilityCapabilities | None = None,
    ) -> None:
        self.root = Path(root)
        self.canonical = self.root / "canonical"
        self.dedupe = self.root / "dedupe"
        self.consumed = self.root / "consumed"
        self.quarantine = self.root / "quarantine"
        for directory in (
            self.root,
            self.canonical,
            self.dedupe,
            self.consumed,
            self.quarantine,
        ):
            directory.mkdir(parents=True, exist_ok=True)
        self._phase_hook = phase_hook
        self._on_diagnostic = on_diagnostic
        self._id_factory = id_factory or (lambda: uuid.uuid4().hex)
        self._guard = AtomicJsonTransaction(
            self.root / ".receiver-state.json",
            purpose="reply-receiver",
        )
        self.capabilities = capabilities or probe_durability(self.root)
        if not self.capabilities.file_fsync or not self.capabilities.atomic_rename:
            raise SpoolDurabilityError(
                "configured receiver lacks file fsync or atomic rename"
            )

    def _locked(self, operation: Callable[[], _Result]) -> _Result:
        result: list[_Result] = []

        def run(current: dict[str, object]) -> dict[str, object]:
            result.append(operation())
            return current

        self._guard.update(run)
        return result[0]

    def _phase(self, name: str) -> None:
        if self._phase_hook is not None:
            self._phase_hook(name)

    def _emit(
        self,
        code: DiagnosticCode,
        *,
        event_id: str | None = None,
        digest: str | None = None,
    ) -> None:
        if self._on_diagnostic is not None:
            try:
                self._on_diagnostic(ReplyDiagnostic(code, event_id, digest))
            except Exception:
                # Observability must not change commit or ACK eligibility.
                pass

    def _sync_directory_if_supported(self, path: Path) -> None:
        if self.capabilities.directory_fsync and not _fsync_directory(path):
            raise SpoolDurabilityError("configured directory fsync failed")

    def _write_atomic(self, destination: Path, wire: bytes, *, phase: str) -> None:
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
        )
        temporary = Path(temporary_name)
        keep_temporary = False
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(wire)
                handle.flush()
                os.fsync(handle.fileno())
            keep_temporary = True
            self._phase(f"after_{phase}_file_fsync_before_rename")
            os.replace(temporary, destination)
            keep_temporary = False
            self._phase(f"after_{phase}_rename_before_directory_fsync")
            self._sync_directory_if_supported(destination.parent)
            self._phase(f"after_{phase}_commit")
        finally:
            if not keep_temporary:
                try:
                    temporary.unlink()
                except FileNotFoundError:
                    pass

    @staticmethod
    def _dedupe_bytes(event: ReplyEvent) -> bytes:
        return json.dumps(
            {
                "digest": event.digest,
                "event_id": event.event_id,
                "session": event.session,
                "version": event.version,
            },
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")

    def _quarantine(self, wire: bytes, *, event_id: str | None) -> None:
        identity = event_id or "invalid"
        wire_digest = hashlib.sha256(wire).hexdigest()
        destination = self.quarantine / f"{identity}-{wire_digest}.bad"
        if destination.exists():
            return
        if len(wire) > 8 * 1024 * 1024:
            wire = json.dumps(
                {"digest": wire_digest, "size": len(wire)},
                separators=(",", ":"),
                sort_keys=True,
            ).encode("ascii")
        self._write_atomic(destination, wire, phase="quarantine")

    def _result_corrupt(self, wire: bytes) -> ReceiveResult:
        self._quarantine(wire, event_id=None)
        self._emit(DiagnosticCode.EVENT_CORRUPT)
        return ReceiveResult(
            ReceiveDisposition.CORRUPT,
            None,
            False,
            DiagnosticCode.EVENT_CORRUPT,
        )

    def receive(self, wire_bytes: bytes) -> ReceiveResult:
        """Commit one complete canonical frame.

        Partial framing is a transport concern and must never call this method.
        Every result carrying ``ack`` is therefore proof that both local files
        reached their configured durability boundary.
        """
        return self._locked(lambda: self._receive_unlocked(wire_bytes))

    def _receive_unlocked(self, wire_bytes: bytes) -> ReceiveResult:
        try:
            event = ReplyEvent.from_bytes(wire_bytes)
        except ReplyProtocolError:
            return self._result_corrupt(wire_bytes)

        canonical_path = self.canonical / f"{event.event_id}.json"
        dedupe_path = self.dedupe / f"{event.event_id}.json"
        existing_wire: bytes | None
        try:
            existing_wire = canonical_path.read_bytes()
        except FileNotFoundError:
            existing_wire = None

        if existing_wire is not None:
            try:
                existing = ReplyEvent.from_bytes(existing_wire)
            except ReplyProtocolError:
                self._quarantine(wire_bytes, event_id=event.event_id)
                self._emit(
                    DiagnosticCode.EVENT_DIGEST_CONFLICT,
                    event_id=event.event_id,
                    digest=event.digest,
                )
                return ReceiveResult(
                    ReceiveDisposition.QUARANTINED,
                    None,
                    False,
                    DiagnosticCode.EVENT_DIGEST_CONFLICT,
                    event.event_id,
                    event.digest,
                )
            # Exact canonical equality additionally detects any wire-level
            # identity tampering if the protocol evolves beyond the digest.
            if existing.digest != event.digest or existing_wire != wire_bytes:
                self._quarantine(wire_bytes, event_id=event.event_id)
                self._emit(
                    DiagnosticCode.EVENT_DIGEST_CONFLICT,
                    event_id=event.event_id,
                    digest=event.digest,
                )
                return ReceiveResult(
                    ReceiveDisposition.QUARANTINED,
                    None,
                    False,
                    DiagnosticCode.EVENT_DIGEST_CONFLICT,
                    event.event_id,
                    event.digest,
                )
        else:
            self._write_atomic(canonical_path, wire_bytes, phase="local_canonical")

        dedupe_wire = self._dedupe_bytes(event)
        try:
            existing_dedupe = dedupe_path.read_bytes()
        except FileNotFoundError:
            existing_dedupe = None
        if existing_dedupe is not None and existing_dedupe != dedupe_wire:
            self._quarantine(wire_bytes, event_id=event.event_id)
            self._emit(
                DiagnosticCode.EVENT_DIGEST_CONFLICT,
                event_id=event.event_id,
                digest=event.digest,
            )
            return ReceiveResult(
                ReceiveDisposition.QUARANTINED,
                None,
                False,
                DiagnosticCode.EVENT_DIGEST_CONFLICT,
                event.event_id,
                event.digest,
            )

        apply = existing_dedupe is None
        if apply:
            self._write_atomic(dedupe_path, dedupe_wire, phase="local_dedupe")
        self._phase("after_local_commit_before_ack_eligibility")
        ack = ReplyAck.for_event(event)
        if apply:
            self._emit(
                DiagnosticCode.EVENT_COMMITTED,
                event_id=event.event_id,
                digest=event.digest,
            )
            return ReceiveResult(
                ReceiveDisposition.COMMITTED,
                ack,
                True,
                DiagnosticCode.EVENT_COMMITTED,
                event.event_id,
                event.digest,
            )
        self._emit(
            DiagnosticCode.EVENT_DUPLICATE,
            event_id=event.event_id,
            digest=event.digest,
        )
        return ReceiveResult(
            ReceiveDisposition.DUPLICATE,
            ack,
            False,
            DiagnosticCode.EVENT_DUPLICATE,
            event.event_id,
            event.digest,
        )

    def read_committed(self, event_id: str) -> ReplyEvent | None:
        """Read a locally committed event only when its dedupe commit exists."""
        return self._locked(lambda: self._read_committed_unlocked(event_id))

    def _read_committed_unlocked(self, event_id: str) -> ReplyEvent | None:
        # Validate the identity without exposing a separate public validator.
        probe = ReplyAck(1, event_id, "0" * 64)
        canonical_path = self.canonical / f"{probe.event_id}.json"
        dedupe_path = self.dedupe / f"{probe.event_id}.json"
        try:
            wire = canonical_path.read_bytes()
            dedupe_wire = dedupe_path.read_bytes()
            event = ReplyEvent.from_bytes(wire)
        except (FileNotFoundError, ReplyProtocolError):
            return None
        if dedupe_wire != self._dedupe_bytes(event):
            return None
        return event

    @staticmethod
    def _consumed_bytes(event: ReplyEvent) -> bytes:
        return json.dumps(
            {
                "digest": event.digest,
                "event_id": event.event_id,
                "version": event.version,
            },
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")

    def pending_committed(self) -> tuple[ReplyEvent, ...]:
        """Enumerate durable commits not yet marked consumed, in identity order.

        This is the local inbox recovery cursor.  It is intentionally independent
        of ``ReceiveResult.apply`` so a crash after commit but before the caller's
        side effect cannot lose an event merely because transport replay is a
        duplicate.
        """
        return self._locked(self._pending_committed_unlocked)

    def _pending_committed_unlocked(self) -> tuple[ReplyEvent, ...]:
        pending: list[ReplyEvent] = []
        for path in sorted(self.canonical.glob("*.json"), key=lambda item: item.name):
            event = self._read_committed_unlocked(path.stem)
            if event is None or path.name != f"{event.event_id}.json":
                continue
            consumed_path = self.consumed / f"{event.event_id}.json"
            try:
                consumed_wire = consumed_path.read_bytes()
            except FileNotFoundError:
                pending.append(event)
                continue
            if consumed_wire != self._consumed_bytes(event):
                # A malformed/mismatched marker never suppresses a valid inbox
                # item.  Exact consumption can still repair it only by an
                # operator-governed recovery; silently replacing it is unsafe.
                pending.append(event)
        return tuple(pending)

    def commit_consumed(self, ack: ReplyAck) -> bool:
        """Durably mark the exact committed identity/digest consumed.

        ACK eligibility remains at canonical+dedupe commit.  This later marker
        only advances the local caller-effect cursor and is idempotent.
        """
        return self._locked(lambda: self._commit_consumed_unlocked(ack))

    def _commit_consumed_unlocked(self, ack: ReplyAck) -> bool:
        event = self._read_committed_unlocked(ack.event_id)
        if event is None or event.digest != ack.digest:
            return False
        destination = self.consumed / f"{event.event_id}.json"
        expected = self._consumed_bytes(event)
        try:
            existing = destination.read_bytes()
        except FileNotFoundError:
            existing = None
        if existing is not None:
            return existing == expected
        self._write_atomic(destination, expected, phase="local_consumed")
        return True

    @staticmethod
    def _eligible_for_removal(
        paths: list[Path], *, now: float, max_age_seconds: float, max_count: int
    ) -> list[Path]:
        existing: list[tuple[float, Path]] = []
        for path in paths:
            try:
                existing.append((path.stat().st_mtime, path))
            except FileNotFoundError:
                continue
        existing.sort(key=lambda item: (item[0], item[1].name))
        old = {path for mtime, path in existing if now - mtime >= max_age_seconds}
        survivors = [path for _mtime, path in existing if path not in old]
        old.update(survivors[: max(0, len(survivors) - max_count)])
        return sorted(old, key=lambda path: path.name)

    def apply_retention(
        self,
        *,
        max_age_seconds: float,
        max_count: int,
        stale_temp_age_seconds: float | None = None,
        now: float | None = None,
    ) -> ReceiverRetentionResult:
        """Bound quarantine and stale crash temps without touching valid events."""
        return self._locked(
            lambda: self._apply_retention_unlocked(
                max_age_seconds=max_age_seconds,
                max_count=max_count,
                stale_temp_age_seconds=stale_temp_age_seconds,
                now=now,
            )
        )

    def _apply_retention_unlocked(
        self,
        *,
        max_age_seconds: float,
        max_count: int,
        stale_temp_age_seconds: float | None,
        now: float | None,
    ) -> ReceiverRetentionResult:
        if max_age_seconds < 0 or max_count < 0:
            raise ValueError("retention bounds must be non-negative")
        temp_age = max_age_seconds if stale_temp_age_seconds is None else stale_temp_age_seconds
        if temp_age < 0:
            raise ValueError("stale temp age must be non-negative")
        timestamp = time.time() if now is None else now
        quarantine = self._eligible_for_removal(
            list(self.quarantine.glob("*.bad")),
            now=timestamp,
            max_age_seconds=max_age_seconds,
            max_count=max_count,
        )
        stale_temps: list[Path] = []
        for directory in (
            self.root,
            self.canonical,
            self.dedupe,
            self.consumed,
            self.quarantine,
        ):
            for path in directory.glob(".*.tmp"):
                try:
                    if timestamp - path.stat().st_mtime >= temp_age:
                        stale_temps.append(path)
                except FileNotFoundError:
                    continue
        touched: set[Path] = set()
        for path in (*quarantine, *stale_temps):
            try:
                path.unlink()
                touched.add(path.parent)
            except FileNotFoundError:
                pass
        for directory in touched:
            self._sync_directory_if_supported(directory)
        return ReceiverRetentionResult(
            quarantine_removed=len(quarantine),
            stale_temps_removed=len(stale_temps),
            canonical_preserved=len(tuple(self.canonical.glob("*.json"))),
            pending_preserved=len(self._pending_committed_unlocked()),
        )
