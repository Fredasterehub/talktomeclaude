"""Persistent selected-voice synthesis worker boundary.

The boundary deliberately owns no canonical answer or cursor state.  A worker
may be replaced after a bounded shutdown without changing the selected voice
or the opaque parent state handed back to its caller.
"""

from __future__ import annotations

import threading
import multiprocessing
import importlib
import os
import queue
import tempfile
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any
from typing import Generic, Protocol, TypeVar

from talktomeclaude.core.deadlines import DEFAULT_DEADLINES, DeadlineName


class SpeechFaultCode(str, Enum):
    SYNTHESIS_FAILED = "synthesis_failed"
    SYNTHESIS_TIMEOUT = "synthesis_timeout"
    STALE_ARTIFACT = "stale_artifact"
    PLAYBACK_FAILED = "playback_failed"
    PLAYBACK_STOP_TIMEOUT = "playback_stop_timeout"
    DEVICE_LOST = "device_lost"
    STOPPED = "stopped"
    WORKER_SHUTDOWN_TIMEOUT = "worker_shutdown_timeout"
    WORKER_RESTART_FAILED = "worker_restart_failed"


class SpeechDiagnosticCode(str, Enum):
    SYNTHESIS_STARTED = "synthesis_started"
    SYNTHESIS_READY = "synthesis_ready"
    SYNTHESIS_REJECTED = "synthesis_rejected"
    PLAYBACK_STARTED = "playback_started"
    PLAYBACK_FINISHED = "playback_finished"
    PLAYBACK_ABORTED = "playback_aborted"
    RETRY_AVAILABLE = "retry_available"
    GENERATION_STOPPED = "generation_stopped"
    WORKER_RESTARTED = "worker_restarted"


@dataclass(frozen=True, slots=True)
class SpeechDiagnostic:
    """Content-free speech observability."""

    code: SpeechDiagnosticCode
    generation: int
    queue_depth: int
    fault: SpeechFaultCode | None = None


@dataclass(frozen=True, slots=True)
class SynthesisRequest:
    generation: int
    unit_id: str
    text: str = field(repr=False)

    def __post_init__(self) -> None:
        if self.generation < 0:
            raise ValueError("speech generation must be non-negative")
        if not self.unit_id or not self.text:
            raise ValueError("speech synthesis request is incomplete")


class SpeechArtifact:
    """Generation-tagged opaque audio with idempotent discard."""

    __slots__ = (
        "generation",
        "unit_id",
        "_payload",
        "_discard",
        "_discarded",
        "_lock",
    )

    def __init__(
        self,
        *,
        generation: int,
        unit_id: str,
        payload: object,
        discard: Callable[[object], None] | None = None,
    ) -> None:
        if generation < 0 or not unit_id:
            raise ValueError("speech artifact identity is invalid")
        self.generation = generation
        self.unit_id = unit_id
        self._payload = payload
        self._discard = discard
        self._discarded = False
        self._lock = threading.Lock()

    @property
    def payload(self) -> object:
        return self._payload

    @property
    def discarded(self) -> bool:
        with self._lock:
            return self._discarded

    def discard(self) -> bool:
        with self._lock:
            if self._discarded:
                return False
            self._discarded = True
            callback = self._discard
            payload = self._payload
        if callback is not None:
            try:
                callback(payload)
            except Exception:
                pass
        return True

    def __repr__(self) -> str:
        return (
            "SpeechArtifact("
            f"generation={self.generation}, unit_id={self.unit_id!r}, "
            f"discarded={self.discarded})"
        )


@dataclass(frozen=True, slots=True)
class SynthesisResult:
    request: SynthesisRequest
    artifact: SpeechArtifact | None = field(default=None, repr=False)
    fault: SpeechFaultCode | None = None

    def __post_init__(self) -> None:
        if (self.artifact is None) == (self.fault is None):
            raise ValueError("synthesis result requires exactly one outcome")

    @classmethod
    def ready(
        cls, request: SynthesisRequest, artifact: SpeechArtifact
    ) -> "SynthesisResult":
        return cls(request, artifact=artifact)

    @classmethod
    def failed(
        cls, request: SynthesisRequest, fault: SpeechFaultCode
    ) -> "SynthesisResult":
        return cls(request, fault=fault)


SynthesisCallback = Callable[[SynthesisResult], None]


class SynthesisWorker(Protocol):
    """One persistent worker initialized with exactly one selected voice."""

    def submit(self, request: SynthesisRequest, callback: SynthesisCallback) -> None: ...

    def terminate(self) -> None: ...

    def kill(self) -> None: ...

    def wait(self, timeout: float) -> object: ...


class SpeechRuntimeError(RuntimeError):
    """A content-free synthesis boundary failure."""


ParentState = TypeVar("ParentState")


@dataclass(frozen=True, slots=True)
class WorkerRestartResult(Generic[ParentState]):
    selected_voice: str = field(repr=False)
    parent_state: ParentState = field(repr=False)
    old_worker_reaped: bool
    terminate_sent: bool
    kill_sent: bool
    boundary_replacement_required: bool
    fault: SpeechFaultCode | None


def _bounded_action(action: Callable[[], object], deadline_seconds: float) -> bool:
    finished = threading.Event()
    succeeded = [False]

    def run() -> None:
        try:
            action()
        except BaseException:
            pass
        else:
            succeeded[0] = True
        finally:
            finished.set()

    thread = threading.Thread(
        target=run,
        name="ttc-speech-worker-lifecycle",
        daemon=True,
    )
    thread.start()
    finished.wait(deadline_seconds)
    return finished.is_set() and succeeded[0]


def _bounded_wait(worker: SynthesisWorker, deadline_seconds: float) -> bool:
    return _bounded_action(
        lambda: worker.wait(deadline_seconds), deadline_seconds
    )


def _bounded_create(
    factory: Callable[[str], SynthesisWorker],
    selected_voice: str,
    deadline_seconds: float,
) -> SynthesisWorker | None:
    finished = threading.Event()
    abandoned = threading.Event()
    created: list[SynthesisWorker] = []
    guard = threading.Lock()

    def create() -> None:
        try:
            worker = factory(selected_voice)
        except BaseException:
            finished.set()
            return
        with guard:
            late = abandoned.is_set()
            if not late:
                created.append(worker)
                finished.set()
        if late:
            # A factory that completes after its ownership deadline must not
            # leak a second warm worker into the process.
            _bounded_action(worker.terminate, deadline_seconds)
            _bounded_action(worker.kill, deadline_seconds)
        return

    thread = threading.Thread(
        target=create,
        name="ttc-speech-worker-create",
        daemon=True,
    )
    thread.start()
    if not finished.wait(deadline_seconds):
        with guard:
            if created:
                return created[0]
            abandoned.set()
            return None
    return created[0] if created else None


class PersistentSpeechRuntime:
    """Own a warm synthesis worker without ever selecting a fallback voice."""

    def __init__(
        self,
        selected_voice: str,
        worker_factory: Callable[[str], SynthesisWorker],
        *,
        shutdown_deadline_seconds: float | None = None,
    ) -> None:
        if not isinstance(selected_voice, str) or not selected_voice.strip():
            raise ValueError("selected voice must not be empty")
        deadline = (
            DEFAULT_DEADLINES[DeadlineName.WORKER_SHUTDOWN].seconds
            if shutdown_deadline_seconds is None
            else shutdown_deadline_seconds
        )
        if deadline < 0:
            raise ValueError("worker shutdown deadline cannot be negative")
        self._selected_voice = selected_voice
        self._factory = worker_factory
        self._shutdown_deadline_seconds = deadline
        self._lock = threading.RLock()
        self._worker: SynthesisWorker | None = _bounded_create(
            worker_factory, selected_voice, deadline
        )
        self._available = self._worker is not None
        if self._worker is None:
            raise SpeechRuntimeError("selected voice worker initialization failed")

    @property
    def selected_voice(self) -> str:
        return self._selected_voice

    def submit(self, request: SynthesisRequest, callback: SynthesisCallback) -> None:
        with self._lock:
            worker = self._worker
            available = self._available
        if worker is None or not available:
            raise SpeechRuntimeError("selected voice worker is unavailable")
        if not _bounded_action(
            lambda: worker.submit(request, callback),
            self._shutdown_deadline_seconds,
        ):
            with self._lock:
                self._available = False
            raise SpeechRuntimeError("speech worker submission failed")

    def restart(self, parent_state: ParentState) -> WorkerRestartResult[ParentState]:
        """Boundedly replace the worker and return the identical parent state."""

        with self._lock:
            deadline = time.monotonic() + self._shutdown_deadline_seconds

            def remaining() -> float:
                return max(0.0, deadline - time.monotonic())

            old_worker = self._worker
            self._available = False
            if old_worker is None:
                terminate_sent = False
                kill_sent = False
                reaped = True
            else:
                terminate_sent = _bounded_action(
                    old_worker.terminate, remaining()
                )
                reaped = _bounded_wait(old_worker, remaining())
                if not reaped:
                    kill_sent = _bounded_action(
                        old_worker.kill, remaining()
                    )
                    reaped = _bounded_wait(old_worker, remaining())
                else:
                    kill_sent = False
            if not reaped:
                raise SpeechRuntimeError("speech worker could not be reaped")
            self._worker = None
            replacement = _bounded_create(
                self._factory,
                self._selected_voice,
                remaining(),
            )
            if replacement is None:
                raise SpeechRuntimeError("selected voice worker restart failed")
            self._worker = replacement
            self._available = True
            return WorkerRestartResult(
                selected_voice=self._selected_voice,
                parent_state=parent_state,
                old_worker_reaped=reaped,
                terminate_sent=terminate_sent,
                kill_sent=kill_sent,
                boundary_replacement_required=not reaped,
                fault=(
                    None if reaped else SpeechFaultCode.WORKER_SHUTDOWN_TIMEOUT
                ),
            )

    def reset_synthesis_boundary(self) -> bool:
        try:
            self.restart(None)
        except SpeechRuntimeError:
            return False
        return True

    def shutdown(self) -> bool:
        """Boundedly reap the current worker without creating a replacement."""

        with self._lock:
            worker = self._worker
            self._available = False
            if worker is None:
                return True
            deadline = time.monotonic() + self._shutdown_deadline_seconds

            def remaining() -> float:
                return max(0.0, deadline - time.monotonic())

            _bounded_action(worker.terminate, remaining())
            reaped = _bounded_wait(worker, remaining())
            if not reaped:
                _bounded_action(worker.kill, remaining())
                reaped = _bounded_wait(worker, remaining())
            if reaped:
                self._worker = None
            return reaped


@dataclass(frozen=True, slots=True)
class _ProcessJob:
    job_id: str
    request: SynthesisRequest


@dataclass(frozen=True, slots=True)
class _ProcessReply:
    job_id: str
    succeeded: bool
    artifact_path: str | None = field(default=None, repr=False)


def _production_synthesize(text: str, path: Path, selected_voice: str) -> None:
    voices = importlib.import_module("talktomeclaude.speech.voices")
    voices.synthesize(text, path, selected_voice)


def _synthesis_process_main(
    selected_voice: str,
    artifact_root: str,
    requests: Any,
    replies: Any,
    stopping: Any,
    synthesize_fn: Callable[[str, Path, str], None],
) -> None:
    root = Path(artifact_root)
    root.mkdir(parents=True, exist_ok=True)
    while not stopping.is_set():
        try:
            job = requests.get(timeout=0.1)
        except queue.Empty:
            continue
        if job is None or not isinstance(job, _ProcessJob):
            return
        descriptor, raw_temporary = tempfile.mkstemp(
            prefix=f".{job.job_id}.", suffix=".tmp.wav", dir=root
        )
        os.close(descriptor)
        temporary = Path(raw_temporary)
        final = root / f"{job.job_id}.wav"
        reply = _ProcessReply(job.job_id, False)
        try:
            synthesize_fn(job.request.text, temporary, selected_voice)
            if not temporary.is_file() or temporary.stat().st_size <= 0:
                raise OSError("synthesis produced no artifact")
            os.replace(temporary, final)
            reply = _ProcessReply(job.job_id, True, str(final))
        except BaseException:
            final.unlink(missing_ok=True)
        finally:
            temporary.unlink(missing_ok=True)
        try:
            replies.put(reply, timeout=0.1)
        except queue.Full:
            final.unlink(missing_ok=True)


class SpawnSynthesisWorker:
    """Windows-spawn-safe fixed-voice synthesis process and callback pump."""

    def __init__(
        self,
        selected_voice: str,
        *,
        queue_capacity: int = 2,
        artifact_root: str | os.PathLike[str] | None = None,
        synthesize_fn: Callable[
            [str, Path, str], None
        ] = _production_synthesize,
        start_method: str = "spawn",
    ) -> None:
        if queue_capacity < 1:
            raise ValueError("synthesis queue capacity must be positive")
        self.selected_voice = selected_voice
        context: Any = multiprocessing.get_context(start_method)
        self._context = context
        self._owns_root = artifact_root is None
        self._root = Path(
            tempfile.mkdtemp(prefix="talktomeclaude-speech-")
            if artifact_root is None
            else artifact_root
        )
        self._root.mkdir(parents=True, exist_ok=True)
        self._requests = self._context.Queue(maxsize=queue_capacity)
        self._replies = self._context.Queue(maxsize=queue_capacity)
        self._stopping = self._context.Event()
        self._callbacks: dict[str, tuple[SynthesisRequest, SynthesisCallback]] = {}
        self._callbacks_lock = threading.Lock()
        self._artifacts: set[Path] = set()
        self._process_exited = False
        self._pump_stop = threading.Event()
        self._process = self._context.Process(
            target=_synthesis_process_main,
            args=(
                selected_voice,
                str(self._root),
                self._requests,
                self._replies,
                self._stopping,
                synthesize_fn,
            ),
            name="talktomeclaude-speech-synthesizer",
            daemon=True,
        )
        self._process.start()
        self._pump = threading.Thread(
            target=self._pump_replies,
            name="ttc-speech-result-pump",
            daemon=True,
        )
        self._pump.start()

    def submit(self, request: SynthesisRequest, callback: SynthesisCallback) -> None:
        if self._stopping.is_set() or not self._process.is_alive():
            raise SpeechRuntimeError("speech synthesis process is unavailable")
        job_id = uuid.uuid4().hex
        with self._callbacks_lock:
            self._callbacks[job_id] = (request, callback)
        try:
            self._requests.put_nowait(_ProcessJob(job_id, request))
        except queue.Full as exc:
            with self._callbacks_lock:
                self._callbacks.pop(job_id, None)
            raise SpeechRuntimeError("speech synthesis queue is full") from exc

    def _discard_path(self, payload: object) -> None:
        if isinstance(payload, Path):
            payload.unlink(missing_ok=True)
            with self._callbacks_lock:
                self._artifacts.discard(payload)
                can_remove_root = self._process_exited and not self._artifacts
            if self._owns_root and can_remove_root:
                try:
                    self._root.rmdir()
                except OSError:
                    pass

    def _pump_replies(self) -> None:
        while not self._pump_stop.is_set():
            try:
                reply = self._replies.get(timeout=0.05)
            except queue.Empty:
                continue
            if not isinstance(reply, _ProcessReply):
                continue
            with self._callbacks_lock:
                owned = self._callbacks.pop(reply.job_id, None)
            if owned is None:
                if reply.artifact_path is not None:
                    Path(reply.artifact_path).unlink(missing_ok=True)
                continue
            request, callback = owned
            artifact: SpeechArtifact | None = None
            if reply.succeeded and reply.artifact_path is not None:
                artifact_path = Path(reply.artifact_path)
                with self._callbacks_lock:
                    self._artifacts.add(artifact_path)
                artifact = SpeechArtifact(
                    generation=request.generation,
                    unit_id=request.unit_id,
                    payload=artifact_path,
                    discard=self._discard_path,
                )
                result = SynthesisResult.ready(request, artifact)
            else:
                result = SynthesisResult.failed(
                    request, SpeechFaultCode.SYNTHESIS_FAILED
                )
            try:
                callback(result)
            except Exception:
                if artifact is not None:
                    artifact.discard()

    def terminate(self) -> None:
        self._stopping.set()
        try:
            self._requests.put_nowait(None)
        except queue.Full:
            pass

    def kill(self) -> None:
        if hasattr(self._process, "kill"):
            self._process.kill()
        else:  # pragma: no cover - supported Python exposes kill on Windows.
            self._process.terminate()

    def wait(self, timeout: float) -> int | None:
        self._process.join(timeout)
        if self._process.is_alive():
            raise TimeoutError("speech synthesis process did not exit")
        self._pump_stop.set()
        self._pump.join(min(max(timeout, 0.0), 0.1))
        with self._callbacks_lock:
            self._process_exited = True
        self._cleanup_unclaimed()
        return self._process.exitcode

    def _cleanup_unclaimed(self) -> None:
        with self._callbacks_lock:
            self._callbacks.clear()
        for path in self._root.glob(".*.tmp.wav"):
            path.unlink(missing_ok=True)
        # Replies not observed before process exit may own final artifacts.
        while True:
            try:
                reply = self._replies.get_nowait()
            except queue.Empty:
                break
            if isinstance(reply, _ProcessReply) and reply.artifact_path is not None:
                Path(reply.artifact_path).unlink(missing_ok=True)
        if self._owns_root:
            try:
                self._root.rmdir()
            except OSError:
                pass
