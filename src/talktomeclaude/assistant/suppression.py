"""Durable speech-director recursion suppression and launch ordering."""

from __future__ import annotations

import os
import re
import subprocess
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Generic, Literal, TypeVar

from talktomeclaude.storage import AtomicJsonTransaction

DIRECTOR_ROLE = "speech-director"
ROLE_ENV = "TALKTO_ME_CLAUDE_ROLE"
CORRELATION_ENV = "TALKTO_ME_CLAUDE_CORRELATION_ID"
REGISTRY_VERSION = 1
_CORRELATION_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,127}$")

TProcess = TypeVar("TProcess")


class SuppressionError(RuntimeError):
    """Suppression could not be durably established or validated."""


def _empty_registry() -> dict[str, Any]:
    return {"version": REGISTRY_VERSION, "directors": {}}


def _validate_registry(value: dict[str, Any]) -> dict[str, Any]:
    if not value:
        return _empty_registry()
    if value.get("version") != REGISTRY_VERSION or not isinstance(
        value.get("directors"), dict
    ):
        raise SuppressionError("director suppression registry is invalid")
    return value


class SuppressionRegistry:
    """Cross-process durable role/correlation/session suppression state."""

    def __init__(
        self,
        path: str | os.PathLike[str],
        *,
        clock: Callable[[], float] = time.time,
        lock_timeout: float = 5.0,
    ) -> None:
        self.path = Path(path)
        self._clock = clock
        self._transaction = AtomicJsonTransaction(
            self.path, timeout=lock_timeout, purpose="director-suppression"
        )

    def preregister(
        self, correlation_id: str, *, role: str = DIRECTOR_ROLE
    ) -> "DirectorLease":
        if role != DIRECTOR_ROLE:
            raise ValueError("director role is invalid")
        if (
            not isinstance(correlation_id, str)
            or _CORRELATION_ID.fullmatch(correlation_id) is None
        ):
            raise ValueError("director correlation_id is invalid")
        now = self._clock()

        def update(current: dict[str, Any]) -> dict[str, Any]:
            state = _validate_registry(current)
            directors = dict(state["directors"])
            existing = directors.get(correlation_id)
            if existing is not None and existing.get("expires_at") is None:
                raise SuppressionError("director correlation is already active")
            directors[correlation_id] = {
                "role": role,
                "session_ids": [],
                "registered_at": now,
                "exited_at": None,
                "expires_at": None,
            }
            return {"version": REGISTRY_VERSION, "directors": directors}

        try:
            self._transaction.update(update, force=True)
        except SuppressionError:
            raise
        except Exception as exc:
            raise SuppressionError(
                "director suppression preregistration failed"
            ) from exc
        return DirectorLease(self, correlation_id)

    def register_session(self, correlation_id: str, session_id: str) -> None:
        if (
            not isinstance(session_id, str)
            or not (1 <= len(session_id) <= 256)
            or any(
                ord(character) < 0x20 or ord(character) == 0x7F
                for character in session_id
            )
        ):
            raise ValueError("director session_id is invalid")

        def update(current: dict[str, Any]) -> dict[str, Any]:
            state = _validate_registry(current)
            directors = dict(state["directors"])
            record = directors.get(correlation_id)
            if not isinstance(record, dict) or record.get("expires_at") is not None:
                raise SuppressionError("director correlation is not active")
            sessions = list(record.get("session_ids", []))
            if session_id not in sessions:
                sessions.append(session_id)
            replacement = dict(record)
            replacement["session_ids"] = sessions
            directors[correlation_id] = replacement
            return {"version": REGISTRY_VERSION, "directors": directors}

        try:
            self._transaction.update(update, force=True)
        except SuppressionError:
            raise
        except Exception as exc:
            raise SuppressionError("director session registration failed") from exc

    def mark_exited(self, correlation_id: str, *, drain_seconds: float) -> None:
        if drain_seconds < 0:
            raise ValueError("drain_seconds cannot be negative")
        now = self._clock()

        def update(current: dict[str, Any]) -> dict[str, Any]:
            state = _validate_registry(current)
            directors = dict(state["directors"])
            record = directors.get(correlation_id)
            if not isinstance(record, dict):
                raise SuppressionError("director correlation is unknown")
            replacement = dict(record)
            replacement["exited_at"] = now
            replacement["expires_at"] = now + drain_seconds
            directors[correlation_id] = replacement
            return {"version": REGISTRY_VERSION, "directors": directors}

        self._transaction.update(update, force=True)

    def _active_records(self) -> dict[str, dict[str, Any]]:
        try:
            state = _validate_registry(self._transaction.read())
        except SuppressionError:
            raise
        except Exception as exc:
            raise SuppressionError("director suppression read failed") from exc
        now = self._clock()
        return {
            correlation: record
            for correlation, record in state["directors"].items()
            if isinstance(record, dict)
            and (record.get("expires_at") is None or record["expires_at"] >= now)
        }

    def reason_for(
        self, event: object, *, environment: Mapping[str, str] | None = None
    ) -> str | None:
        environment = environment or {}
        if environment.get(ROLE_ENV) == DIRECTOR_ROLE:
            return "suppressed_role"
        role = getattr(event, "role", None)
        if role == DIRECTOR_ROLE:
            return "suppressed_role"
        records = self._active_records()
        session_id = getattr(event, "session_id", None)
        if any(
            session_id in record.get("session_ids", []) for record in records.values()
        ):
            return "suppressed_session"
        correlation = environment.get(CORRELATION_ENV) or getattr(
            event, "correlation_id", None
        )
        if correlation in records:
            return "suppressed_correlation"
        return None

    def session_registered(self, correlation_id: str, session_id: str) -> bool:
        record = self._active_records().get(correlation_id)
        return record is not None and session_id in record.get("session_ids", [])

    def prune_expired(self) -> int:
        now = self._clock()
        removed = 0

        def update(current: dict[str, Any]) -> dict[str, Any]:
            nonlocal removed
            state = _validate_registry(current)
            retained: dict[str, Any] = {}
            for correlation, record in state["directors"].items():
                expires = record.get("expires_at") if isinstance(record, dict) else 0
                if expires is not None and expires < now:
                    removed += 1
                else:
                    retained[correlation] = record
            return {"version": REGISTRY_VERSION, "directors": retained}

        self._transaction.update(update, force=removed > 0)
        return removed


@dataclass(frozen=True, slots=True)
class DirectorLease:
    registry: SuppressionRegistry
    correlation_id: str

    def register_initialization(
        self, session_id: str, accept: Callable[[str], None] | None = None
    ) -> None:
        """Commit session suppression before exposing initialization upstream."""
        self.registry.register_session(self.correlation_id, session_id)
        if accept is not None:
            accept(session_id)

    def permit_result(self, session_id: str, accept: Callable[[], None]) -> bool:
        """Accept a result only after this exact session is durably registered."""
        if not self.registry.session_registered(self.correlation_id, session_id):
            return False
        accept()
        return True

    def mark_exited(self, *, drain_seconds: float) -> None:
        self.registry.mark_exited(self.correlation_id, drain_seconds=drain_seconds)


@dataclass(slots=True)
class ManagedDirectorProcess(Generic[TProcess]):
    """Own a director process and close suppression only after confirmed exit."""

    process: TProcess
    lease: DirectorLease
    drain_seconds: float
    terminate_timeout_seconds: float = 1.0
    kill_timeout_seconds: float = 1.0
    _closed: bool = field(default=False, init=False)
    _close_lock: threading.Lock = field(default_factory=threading.Lock, init=False)

    def __post_init__(self) -> None:
        if self.drain_seconds < 0:
            raise ValueError("drain_seconds cannot be negative")
        if self.terminate_timeout_seconds < 0 or self.kill_timeout_seconds < 0:
            raise ValueError("process shutdown timeouts cannot be negative")

    def _close_confirmed(self) -> None:
        with self._close_lock:
            if self._closed:
                return
            self.lease.mark_exited(drain_seconds=self.drain_seconds)
            self._closed = True

    def close(self) -> bool:
        """Close only if a non-blocking poll confirms that the child exited."""

        if self._closed:
            return True
        result = getattr(self.process, "poll")()
        if result is None:
            return False
        self._close_confirmed()
        return True

    @staticmethod
    def _note_cleanup(primary: BaseException, cleanup_error: BaseException) -> None:
        primary.add_note(
            "director process cleanup also failed: "
            f"{type(cleanup_error).__name__}"
        )

    def wait(self, *args: Any, **kwargs: Any) -> Any:
        try:
            result = getattr(self.process, "wait")(*args, **kwargs)
        except BaseException as primary:
            try:
                # wait(timeout=...) may fail while the child is still alive.
                # A poll-confirmed exit is the only safe close boundary.
                self.close()
            except Exception as cleanup_error:
                self._note_cleanup(primary, cleanup_error)
            raise
        # A successful process wait is itself confirmation that the child was
        # reaped, including for process doubles whose poll value is stale.
        self._close_confirmed()
        return result

    def poll(self) -> Any:
        result = getattr(self.process, "poll")()
        if result is not None:
            self._close_confirmed()
        return result

    def terminate_and_reap(self) -> Any:
        """Boundedly terminate, then kill and reap the owned child if needed."""

        polled = getattr(self.process, "poll")()
        if polled is not None:
            self._close_confirmed()
            return polled

        getattr(self.process, "terminate")()
        try:
            return self.wait(timeout=self.terminate_timeout_seconds)
        except subprocess.TimeoutExpired as terminate_timeout:
            try:
                # The child may have exited at the timeout boundary.
                if self.close():
                    return getattr(self.process, "poll")()
                getattr(self.process, "kill")()
                return self.wait(timeout=self.kill_timeout_seconds)
            except BaseException as kill_error:
                self._note_cleanup(terminate_timeout, kill_error)
                raise terminate_timeout

    def __enter__(self) -> "ManagedDirectorProcess[TProcess]":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: object,
    ) -> Literal[False]:
        try:
            self.terminate_and_reap()
        except BaseException as cleanup_error:
            if exc is None:
                raise
            self._note_cleanup(exc, cleanup_error)
        return False


class DirectorLaunchGuard:
    """Enforce durable preregistration before any process-spawn callback."""

    def __init__(self, registry: SuppressionRegistry) -> None:
        self._registry = registry

    def launch(
        self,
        command: Sequence[str],
        correlation_id: str,
        spawn: Callable[[tuple[str, ...], Mapping[str, str]], TProcess],
        *,
        environment: Mapping[str, str] | None = None,
        drain_seconds: float = 5.0,
    ) -> ManagedDirectorProcess[TProcess]:
        if drain_seconds < 0:
            raise ValueError("drain_seconds cannot be negative")
        lease = self._registry.preregister(correlation_id, role=DIRECTOR_ROLE)
        child_environment = dict(os.environ if environment is None else environment)
        child_environment[ROLE_ENV] = DIRECTOR_ROLE
        child_environment[CORRELATION_ENV] = correlation_id
        try:
            process = spawn(tuple(command), child_environment)
        except BaseException as primary:
            # No child exists, but keep the durable identity through this
            # immediate exit boundary so an already queued hook still drains.
            try:
                lease.mark_exited(drain_seconds=0.0)
            except Exception as cleanup_error:
                primary.add_note(
                    "director suppression cleanup also failed: "
                    f"{type(cleanup_error).__name__}"
                )
            raise
        return ManagedDirectorProcess(process, lease, drain_seconds)


class DirectorEventGate:
    """Small lifecycle gate separating initialization from plan/result acceptance."""

    def __init__(self, lease: DirectorLease) -> None:
        self._lease = lease

    def initialization(self, session_id: str, accept: Callable[[str], None]) -> None:
        self._lease.register_initialization(session_id, accept)

    def result(self, session_id: str, accept: Callable[[], None]) -> bool:
        return self._lease.permit_result(session_id, accept)
