"""Atomic file-per-event remote reply spool.

``ready`` is the only replay source.  An ACK atomically moves its exact event
to ``acked``; retention never examines or removes an unacknowledged ready file.
"""

from __future__ import annotations

import os
import tempfile
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TypeVar

from talktomeclaude.storage.atomic import AtomicJsonTransaction

from .contracts import (
    AckDisposition,
    AckResult,
    DiagnosticCode,
    ReplyAck,
    ReplyDiagnostic,
    ReplyEvent,
    ReplyProtocolError,
)

_Result = TypeVar("_Result")


class ReplySpoolError(RuntimeError):
    """Base class for content-safe spool failures."""


class SpoolFullError(ReplySpoolError):
    code = DiagnosticCode.SPOOL_FULL


class SpoolConflictError(ReplySpoolError):
    code = DiagnosticCode.EVENT_DIGEST_CONFLICT


class SpoolDurabilityError(ReplySpoolError):
    """The configured filesystem lacks a required durability primitive."""


@dataclass(frozen=True, slots=True)
class DurabilityCapabilities:
    file_fsync: bool
    atomic_rename: bool
    directory_fsync: bool

    @property
    def guarantee(self) -> str:
        if self.file_fsync and self.atomic_rename and self.directory_fsync:
            return "file_and_directory_durable"
        if self.file_fsync and self.atomic_rename:
            return "file_durable_atomic_rename_directory_sync_unavailable"
        return "required_durability_unavailable"


@dataclass(frozen=True, slots=True)
class SpoolRecord:
    event: ReplyEvent
    wire_bytes: bytes = field(repr=False)
    path: Path = field(repr=False)


@dataclass(frozen=True, slots=True)
class CursorSnapshot:
    ready_event_ids: tuple[str, ...]
    diagnostic: DiagnosticCode = DiagnosticCode.CURSOR_RECOVERED


@dataclass(frozen=True, slots=True)
class RetentionResult:
    acked_removed: int
    quarantine_removed: int
    stale_temps_removed: int
    ready_preserved: int
    diagnostic: DiagnosticCode = DiagnosticCode.RETENTION_APPLIED


def _fsync_directory(path: Path) -> bool:
    """Sync *path* and return whether this platform/filesystem supports it."""
    flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    try:
        descriptor = os.open(path, flags)
    except OSError:
        return False
    try:
        os.fsync(descriptor)
    except OSError:
        return False
    finally:
        os.close(descriptor)
    return True


def probe_durability(root: Path) -> DurabilityCapabilities:
    """Exercise, rather than infer, the configured filesystem primitives."""
    root.mkdir(parents=True, exist_ok=True)
    probe = root / f".durability-{uuid.uuid4().hex}"
    renamed = root / f".durability-{uuid.uuid4().hex}.ready"
    file_fsync = False
    atomic_rename = False
    try:
        with probe.open("xb") as handle:
            handle.write(b"durability-probe")
            handle.flush()
            os.fsync(handle.fileno())
            file_fsync = True
        os.replace(probe, renamed)
        atomic_rename = renamed.read_bytes() == b"durability-probe"
        directory_fsync = _fsync_directory(root)
        return DurabilityCapabilities(file_fsync, atomic_rename, directory_fsync)
    except OSError:
        return DurabilityCapabilities(file_fsync, atomic_rename, False)
    finally:
        for path in (probe, renamed):
            try:
                path.unlink()
            except FileNotFoundError:
                pass


class ReplySpool:
    """A replay-safe, bounded file spool for authoritative replies."""

    def __init__(
        self,
        root: str | os.PathLike[str],
        *,
        max_ready_events: int = 10_000,
        max_ready_bytes: int = 256 * 1024 * 1024,
        phase_hook: Callable[[str], None] | None = None,
        on_diagnostic: Callable[[ReplyDiagnostic], None] | None = None,
        id_factory: Callable[[], str] | None = None,
        capabilities: DurabilityCapabilities | None = None,
    ) -> None:
        if max_ready_events < 1 or max_ready_bytes < 1:
            raise ValueError("spool limits must be positive")
        self.root = Path(root)
        self.ready = self.root / "ready"
        self.acked = self.root / "acked"
        self.quarantine = self.root / "quarantine"
        for directory in (self.ready, self.acked, self.quarantine):
            directory.mkdir(parents=True, exist_ok=True)
        self.max_ready_events = max_ready_events
        self.max_ready_bytes = max_ready_bytes
        self._phase_hook = phase_hook
        self._on_diagnostic = on_diagnostic
        self._id_factory = id_factory or (lambda: uuid.uuid4().hex)
        self._guard = AtomicJsonTransaction(
            self.root / ".spool-state.json",
            purpose="reply-spool",
        )
        self.capabilities = capabilities or probe_durability(self.root)
        if not self.capabilities.file_fsync or not self.capabilities.atomic_rename:
            raise SpoolDurabilityError(
                "configured spool lacks file fsync or atomic rename"
            )

    def _phase(self, name: str) -> None:
        if self._phase_hook is not None:
            self._phase_hook(name)

    def _locked(self, operation: Callable[[], _Result]) -> _Result:
        result: list[_Result] = []

        def run(current: dict[str, object]) -> dict[str, object]:
            result.append(operation())
            return current

        self._guard.update(run)
        return result[0]

    def _emit(
        self,
        code: DiagnosticCode,
        *,
        event_id: str | None = None,
        digest: str | None = None,
        count: int | None = None,
    ) -> None:
        if self._on_diagnostic is not None:
            try:
                self._on_diagnostic(ReplyDiagnostic(code, event_id, digest, count))
            except Exception:
                # Observability must never control a durable spool transition.
                pass

    @staticmethod
    def _event_path(directory: Path, event_id: str) -> Path:
        # ReplyEvent/ReplyAck validate this as a single safe path component.
        return directory / f"{event_id}.json"

    def _sync_directory_if_supported(self, path: Path) -> None:
        if self.capabilities.directory_fsync and not _fsync_directory(path):
            raise SpoolDurabilityError("configured directory fsync failed")

    def _ready_usage(self) -> tuple[int, int]:
        paths = tuple(self.ready.glob("*.json"))
        size = 0
        for path in paths:
            try:
                size += path.stat().st_size
            except FileNotFoundError:
                continue
        return len(paths), size

    def enqueue(self, event: ReplyEvent) -> SpoolRecord:
        return self._locked(lambda: self._enqueue_unlocked(event))

    def _enqueue_unlocked(self, event: ReplyEvent) -> SpoolRecord:
        wire = event.to_bytes()
        destination = self._event_path(self.ready, event.event_id)
        acked = self._event_path(self.acked, event.event_id)
        for existing in (destination, acked):
            try:
                existing_wire = existing.read_bytes()
            except FileNotFoundError:
                continue
            try:
                prior = ReplyEvent.from_bytes(existing_wire)
            except ReplyProtocolError as exc:
                raise SpoolConflictError("existing spool identity is corrupt") from exc
            if prior.digest != event.digest or existing_wire != wire:
                raise SpoolConflictError("spool identity already has another digest")
            return SpoolRecord(prior, existing_wire, existing)

        count, byte_count = self._ready_usage()
        if count >= self.max_ready_events or byte_count + len(wire) > self.max_ready_bytes:
            self._emit(DiagnosticCode.SPOOL_FULL, count=count)
            raise SpoolFullError("reply spool capacity reached")

        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{event.event_id}.", suffix=".tmp", dir=self.ready
        )
        temporary = Path(temporary_name)
        keep_temporary = False
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(wire)
                self._phase("before_temp_flush")
                handle.flush()
                os.fsync(handle.fileno())
                self._phase("after_file_fsync_before_ready_rename")
            keep_temporary = True
            os.replace(temporary, destination)
            keep_temporary = False
            self._phase("after_ready_rename_before_directory_fsync")
            self._sync_directory_if_supported(self.ready)
            self._phase("after_ready_directory_fsync")
            return SpoolRecord(event, wire, destination)
        finally:
            # Ordinary failures before the durable rename leave the temp file as
            # crash evidence. It is never part of pending enumeration.
            if not keep_temporary:
                try:
                    temporary.unlink()
                except FileNotFoundError:
                    pass

    def _quarantine_path(self, source: Path) -> Path:
        del source
        return self.quarantine / f"quarantine-{self._id_factory()}.bad"

    def _quarantine_file(self, source: Path) -> None:
        destination = self._quarantine_path(source)
        os.replace(source, destination)
        self._sync_directory_if_supported(source.parent)
        self._sync_directory_if_supported(self.quarantine)

    def pending(self) -> tuple[SpoolRecord, ...]:
        """Return a deterministic replay snapshot; temp files are invisible."""
        return self._locked(self._pending_unlocked)

    def _pending_unlocked(self) -> tuple[SpoolRecord, ...]:
        records: list[SpoolRecord] = []
        for path in sorted(self.ready.glob("*.json"), key=lambda item: item.name):
            try:
                wire = path.read_bytes()
                event = ReplyEvent.from_bytes(wire)
                if path.name != f"{event.event_id}.json":
                    raise ReplyProtocolError("ready filename does not match event identity")
            except (OSError, ReplyProtocolError):
                try:
                    self._quarantine_file(path)
                except FileNotFoundError:
                    pass
                self._emit(DiagnosticCode.EVENT_CORRUPT)
                continue
            records.append(SpoolRecord(event, wire, path))
        if records:
            self._emit(DiagnosticCode.READY_REPLAY, count=len(records))
        return tuple(records)

    def recover_cursor(self) -> CursorSnapshot:
        snapshot = CursorSnapshot(tuple(record.event.event_id for record in self.pending()))
        self._emit(DiagnosticCode.CURSOR_RECOVERED, count=len(snapshot.ready_event_ids))
        return snapshot

    def _rejected_ack(self, ack: ReplyAck) -> AckResult:
        self._emit(
            DiagnosticCode.ACK_REJECTED,
            event_id=ack.event_id,
            digest=ack.digest,
        )
        return AckResult(
            AckDisposition.REJECTED,
            DiagnosticCode.ACK_REJECTED,
            ack.event_id,
            ack.digest,
        )

    def commit_ack(self, ack: ReplyAck) -> AckResult:
        return self._locked(lambda: self._commit_ack_unlocked(ack))

    def _commit_ack_unlocked(self, ack: ReplyAck) -> AckResult:
        ready = self._event_path(self.ready, ack.event_id)
        committed = self._event_path(self.acked, ack.event_id)

        try:
            committed_event = ReplyEvent.from_bytes(committed.read_bytes())
        except FileNotFoundError:
            committed_event = None
        except ReplyProtocolError:
            return self._rejected_ack(ack)
        if committed_event is not None:
            if committed_event.digest == ack.digest:
                self._emit(
                    DiagnosticCode.ACK_DUPLICATE,
                    event_id=ack.event_id,
                    digest=ack.digest,
                )
                return AckResult(
                    AckDisposition.ALREADY_COMMITTED,
                    DiagnosticCode.ACK_DUPLICATE,
                    ack.event_id,
                    ack.digest,
                )
            return self._rejected_ack(ack)

        try:
            event = ReplyEvent.from_bytes(ready.read_bytes())
        except (FileNotFoundError, ReplyProtocolError):
            return self._rejected_ack(ack)
        if event.event_id != ack.event_id or event.digest != ack.digest:
            return self._rejected_ack(ack)

        self._phase("before_remote_ack_commit")
        os.replace(ready, committed)
        self._phase("after_remote_ack_rename_before_directory_fsync")
        self._sync_directory_if_supported(self.ready)
        self._sync_directory_if_supported(self.acked)
        self._phase("after_remote_ack_commit_before_cleanup")
        self._emit(
            DiagnosticCode.ACK_COMMITTED,
            event_id=ack.event_id,
            digest=ack.digest,
        )
        return AckResult(
            AckDisposition.COMMITTED,
            DiagnosticCode.ACK_COMMITTED,
            ack.event_id,
            ack.digest,
        )

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
        excess = max(0, len(survivors) - max_count)
        old.update(survivors[:excess])
        return sorted(old, key=lambda path: path.name)

    def apply_retention(
        self,
        *,
        max_age_seconds: float,
        max_count: int,
        stale_temp_age_seconds: float | None = None,
        now: float | None = None,
    ) -> RetentionResult:
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
    ) -> RetentionResult:
        if max_age_seconds < 0 or max_count < 0:
            raise ValueError("retention bounds must be non-negative")
        temp_age = (
            max_age_seconds
            if stale_temp_age_seconds is None
            else stale_temp_age_seconds
        )
        if temp_age < 0:
            raise ValueError("stale temp age must be non-negative")
        timestamp = time.time() if now is None else now
        acked = self._eligible_for_removal(
            list(self.acked.glob("*.json")),
            now=timestamp,
            max_age_seconds=max_age_seconds,
            max_count=max_count,
        )
        quarantined = self._eligible_for_removal(
            list(self.quarantine.iterdir()),
            now=timestamp,
            max_age_seconds=max_age_seconds,
            max_count=max_count,
        )
        stale_temps = self._eligible_for_removal(
            list(self.ready.glob(".*.tmp")),
            now=timestamp,
            max_age_seconds=temp_age,
            max_count=max_count,
        )
        for path in (*acked, *quarantined, *stale_temps):
            try:
                path.unlink()
            except FileNotFoundError:
                pass
        if acked:
            self._sync_directory_if_supported(self.acked)
        if quarantined:
            self._sync_directory_if_supported(self.quarantine)
        if stale_temps:
            self._sync_directory_if_supported(self.ready)
        result = RetentionResult(
            acked_removed=len(acked),
            quarantine_removed=len(quarantined),
            stale_temps_removed=len(stale_temps),
            ready_preserved=len(tuple(self.ready.glob("*.json"))),
        )
        self._emit(
            DiagnosticCode.RETENTION_APPLIED,
            count=(
                result.acked_removed
                + result.quarantine_removed
                + result.stale_temps_removed
            ),
        )
        return result
