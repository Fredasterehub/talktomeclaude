"""Dedicated microphone input ownership for the Windows companion."""

from __future__ import annotations

import importlib
import math
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any, Protocol


class AudioCaptureSink(Protocol):
    def add_audio(self, chunk: Any) -> None: ...


class AudioInputFaultCode(str, Enum):
    DEVICE_OPEN_FAILED = "device_open_failed"
    INPUT_STATUS = "input_status"
    CAPTURE_REJECTED = "capture_rejected"
    DEVICE_STOP_TIMEOUT = "device_stop_timeout"


@dataclass(frozen=True, slots=True)
class AudioInputFault:
    """Content-free device/capture fault notification."""

    code: AudioInputFaultCode


@dataclass(frozen=True, slots=True)
class AudioStopResult:
    silence_confirmed: bool
    boundary_replacement_required: bool
    fault: AudioInputFaultCode | None = None


class AudioInputError(RuntimeError):
    """The dedicated input boundary could not be established."""


def _bounded_action(action: Callable[[], None], deadline_seconds: float) -> bool:
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

    threading.Thread(
        target=run,
        name="ttc-dedicated-input-lifecycle",
        daemon=True,
    ).start()
    return finished.wait(max(0.0, deadline_seconds)) and succeeded[0]


class DedicatedAudioInput:
    """Own exactly one callback-driven ``sounddevice.InputStream``."""

    def __init__(
        self,
        sink: AudioCaptureSink,
        *,
        sample_rate: int = 16_000,
        blocksize: int = 1_600,
        sounddevice_module: Any | None = None,
        numpy_module: Any | None = None,
        on_level: Callable[[float], None] | None = None,
        on_fault: Callable[[AudioInputFault], None] | None = None,
        lifecycle_timeout_seconds: float = 0.25,
    ) -> None:
        if sample_rate <= 0 or blocksize <= 0:
            raise ValueError("audio input dimensions must be positive")
        if not math.isfinite(lifecycle_timeout_seconds) or lifecycle_timeout_seconds <= 0:
            raise ValueError("audio lifecycle timeout must be positive")
        self._sink = sink
        self._sample_rate = sample_rate
        self._blocksize = blocksize
        self._sounddevice = sounddevice_module
        self._numpy = numpy_module
        self._on_level = on_level
        self._on_fault = on_fault
        self._timeout = lifecycle_timeout_seconds
        self._lock = threading.RLock()
        self._lifecycle_lock = threading.RLock()
        self._next_token = 0
        self._active_token: int | None = None
        self._stream: Any | None = None
        self._closed = False
        self._tainted = False

    @property
    def active(self) -> bool:
        with self._lock:
            return self._active_token is not None and self._stream is not None

    def _modules(self) -> tuple[Any, Any]:
        with self._lock:
            if self._sounddevice is None:
                self._sounddevice = importlib.import_module("sounddevice")
            if self._numpy is None:
                self._numpy = importlib.import_module("numpy")
            return self._sounddevice, self._numpy

    def _fault(self, code: AudioInputFaultCode) -> None:
        if self._on_fault is not None:
            try:
                self._on_fault(AudioInputFault(code))
            except Exception:
                pass

    def start(self) -> bool:
        with self._lifecycle_lock:
            return self._start()

    def _start(self) -> bool:
        with self._lock:
            if self._closed:
                raise AudioInputError("audio input owner is closed")
            if self._tainted:
                raise AudioInputError("audio input owner requires replacement")
            if self._active_token is not None:
                return False
            self._next_token += 1
            token = self._next_token
            self._active_token = token

        def create_and_start() -> None:
            stream: Any | None = None
            try:
                sounddevice, _numpy = self._modules()
                stream = sounddevice.InputStream(
                    samplerate=self._sample_rate,
                    channels=1,
                    dtype="float32",
                    blocksize=self._blocksize,
                    callback=lambda data, frames, timing, status: self._callback(
                        token, data, frames, timing, status
                    ),
                )
                stream.start()
                with self._lock:
                    if self._active_token == token and not self._closed:
                        self._stream = stream
                        stream = None
            finally:
                if stream is not None:
                    self._cleanup_stream(stream, self._timeout)

        established = _bounded_action(create_and_start, self._timeout)
        orphaned_stream: Any | None = None
        with self._lock:
            adopted = self._stream is not None and self._active_token == token
            if not established or not adopted:
                if self._active_token == token:
                    self._active_token = None
                if self._stream is not None:
                    orphaned_stream = self._stream
                    self._stream = None
        if orphaned_stream is not None:
            self._cleanup_stream(orphaned_stream, self._timeout)
        if not established or not adopted:
            self._fault(AudioInputFaultCode.DEVICE_OPEN_FAILED)
            raise AudioInputError("dedicated audio input failed to start")
        return True

    def _callback(
        self,
        token: int,
        data: Any,
        frames: int,
        timing: Any,
        status: Any,
    ) -> None:
        del frames, timing
        with self._lock:
            if self._active_token != token or self._closed:
                return
        if status:
            self._fault(AudioInputFaultCode.INPUT_STATUS)
        try:
            _sounddevice, numpy = self._modules()
            block = numpy.asarray(data, dtype=numpy.float32).reshape(-1).copy()
            self._sink.add_audio(block)
        except Exception:
            self._fault(AudioInputFaultCode.CAPTURE_REJECTED)
            return
        if self._on_level is not None:
            try:
                level = (
                    float(numpy.sqrt(numpy.mean(numpy.square(block))))
                    if block.size
                    else 0.0
                )
                self._on_level(level)
            except Exception:
                pass

    @staticmethod
    def _cleanup_stream(stream: Any, deadline_seconds: float) -> bool:
        deadline = time.monotonic() + max(0.0, deadline_seconds)
        stopped = _bounded_action(
            stream.stop, max(0.0, deadline - time.monotonic())
        )
        if not stopped and hasattr(stream, "abort"):
            stopped = _bounded_action(
                stream.abort, max(0.0, deadline - time.monotonic())
            )
        closed = _bounded_action(
            stream.close, max(0.0, deadline - time.monotonic())
        )
        return stopped and closed

    def stop(self) -> AudioStopResult:
        with self._lifecycle_lock:
            return self._stop()

    def seal(self) -> None:
        """Reject late callbacks immediately without awaiting device teardown."""

        with self._lock:
            self._active_token = None

    def _stop(self) -> AudioStopResult:
        with self._lock:
            self._active_token = None
            stream = self._stream
            self._stream = None
            tainted = self._tainted
        if stream is None:
            if tainted:
                return AudioStopResult(
                    False,
                    True,
                    AudioInputFaultCode.DEVICE_STOP_TIMEOUT,
                )
            return AudioStopResult(True, False)
        confirmed = self._cleanup_stream(stream, self._timeout)
        if confirmed:
            return AudioStopResult(True, False)
        self._fault(AudioInputFaultCode.DEVICE_STOP_TIMEOUT)
        with self._lock:
            self._tainted = True
        return AudioStopResult(
            False,
            True,
            AudioInputFaultCode.DEVICE_STOP_TIMEOUT,
        )

    def close(self) -> AudioStopResult:
        with self._lifecycle_lock:
            with self._lock:
                if self._closed:
                    if self._tainted:
                        return AudioStopResult(
                            False,
                            True,
                            AudioInputFaultCode.DEVICE_STOP_TIMEOUT,
                        )
                    return AudioStopResult(True, False)
                self._closed = True
            return self._stop()


class Float32AudioAssembler:
    """Losslessly concatenate float32 callback blocks into Faster Whisper input."""

    def __init__(self, numpy_module: Any | None = None) -> None:
        self._numpy = numpy_module

    def __call__(self, chunks: Sequence[Any]) -> Any:
        numpy = self._numpy or importlib.import_module("numpy")
        blocks: list[Any] = []
        for chunk in chunks:
            block = numpy.asarray(chunk)
            if block.dtype != numpy.float32:
                raise ValueError("audio blocks must contain float32 samples")
            blocks.append(block.reshape(-1))
        if not blocks:
            return numpy.empty((0,), dtype=numpy.float32)
        if len(blocks) == 1:
            return numpy.ascontiguousarray(blocks[0]).copy()
        return numpy.ascontiguousarray(numpy.concatenate(blocks, axis=0))
