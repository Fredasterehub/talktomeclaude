from __future__ import annotations

import threading
import time
import unittest
from typing import Any

import numpy as np

from talktomeclaude.capture.service import CaptureService
from talktomeclaude.companion.audio import (
    AudioInputError,
    AudioInputFault,
    AudioInputFaultCode,
    DedicatedAudioInput,
    Float32AudioAssembler,
)


class _FakeStream:
    def __init__(
        self,
        callback: Any,
        *,
        start_error: BaseException | None = None,
        start_gate: threading.Event | None = None,
        stop_gate: threading.Event | None = None,
        abort_hangs: bool = False,
    ) -> None:
        self.callback = callback
        self.start_error = start_error
        self.start_gate = start_gate
        self.stop_gate = stop_gate
        self.abort_hangs = abort_hangs
        self.started = 0
        self.stopped = 0
        self.aborted = 0
        self.closed = 0

    def start(self) -> None:
        self.started += 1
        if self.start_gate is not None:
            self.start_gate.wait(2.0)
        if self.start_error is not None:
            raise self.start_error

    def stop(self) -> None:
        self.stopped += 1
        if self.stop_gate is not None:
            self.stop_gate.wait(2.0)

    def abort(self) -> None:
        self.aborted += 1
        if self.stop_gate is not None:
            if self.abort_hangs:
                self.stop_gate.wait(2.0)
            else:
                self.stop_gate.set()

    def close(self) -> None:
        self.closed += 1

    def emit(self, data: Any, status: Any = None) -> None:
        self.callback(data, len(data), object(), status)


class _FakeSoundDevice:
    def __init__(
        self,
        *,
        start_error: BaseException | None = None,
        start_gate: threading.Event | None = None,
        stop_gate: threading.Event | None = None,
        abort_hangs: bool = False,
    ) -> None:
        self.start_error = start_error
        self.start_gate = start_gate
        self.stop_gate = stop_gate
        self.abort_hangs = abort_hangs
        self.calls: list[dict[str, Any]] = []
        self.streams: list[_FakeStream] = []
        self.global_stop_calls = 0

    def InputStream(self, **kwargs: Any) -> _FakeStream:  # noqa: N802
        self.calls.append(kwargs)
        stream = _FakeStream(
            kwargs["callback"],
            start_error=self.start_error,
            start_gate=self.start_gate,
            stop_gate=self.stop_gate,
            abort_hangs=self.abort_hangs,
        )
        self.streams.append(stream)
        return stream

    def stop(self) -> None:
        self.global_stop_calls += 1
        raise AssertionError("global sounddevice.stop() must not be used")


class _RejectingSink:
    def add_audio(self, chunk: Any) -> None:
        del chunk
        raise RuntimeError("private transcript material must not escape")


def _wait_until(predicate: Any, timeout: float = 0.5) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.005)
    return bool(predicate())


class DedicatedAudioInputTests(unittest.TestCase):
    def test_lazily_owns_one_configured_input_stream(self) -> None:
        sounddevice = _FakeSoundDevice()
        capture = CaptureService()
        owner = DedicatedAudioInput(
            capture,
            sounddevice_module=sounddevice,
            numpy_module=np,
        )

        self.assertEqual(sounddevice.calls, [])
        self.assertTrue(owner.start())
        self.assertFalse(owner.start())

        self.assertEqual(len(sounddevice.calls), 1)
        call = sounddevice.calls[0]
        self.assertEqual(call["samplerate"], 16_000)
        self.assertEqual(call["blocksize"], 1_600)
        self.assertEqual(call["channels"], 1)
        self.assertEqual(call["dtype"], "float32")
        self.assertTrue(callable(call["callback"]))

    def test_callback_feeds_capture_service_as_copied_flat_float32(self) -> None:
        sounddevice = _FakeSoundDevice()
        capture = CaptureService(audio_assembler=Float32AudioAssembler(np))
        levels: list[float] = []
        owner = DedicatedAudioInput(
            capture,
            sounddevice_module=sounddevice,
            numpy_module=np,
            on_level=levels.append,
        )
        capture.toggle()
        owner.start()
        source = np.array([[0.25], [-0.5]], dtype=np.float32)

        sounddevice.streams[0].emit(source)
        source[:] = 1.0
        completion = capture.toggle()

        self.assertFalse(isinstance(completion, int))
        chunks = completion.audio.chunks
        self.assertEqual(len(chunks), 1)
        np.testing.assert_array_equal(
            chunks[0], np.array([0.25, -0.5], dtype=np.float32)
        )
        self.assertEqual(chunks[0].ndim, 1)
        self.assertAlmostEqual(levels[0], np.sqrt((0.25**2 + 0.5**2) / 2))

    def test_stop_and_close_are_idempotent_and_ignore_late_callbacks(self) -> None:
        sounddevice = _FakeSoundDevice()
        capture = CaptureService()
        owner = DedicatedAudioInput(
            capture,
            sounddevice_module=sounddevice,
            numpy_module=np,
        )
        capture.toggle()
        owner.start()
        stream = sounddevice.streams[0]

        first = owner.stop()
        second = owner.stop()
        stream.emit(np.array([0.5], dtype=np.float32))
        closed = owner.close()
        closed_again = owner.close()
        completion = capture.toggle()

        self.assertTrue(first.silence_confirmed)
        self.assertEqual(first, second)
        self.assertEqual(first, closed)
        self.assertEqual(closed, closed_again)
        self.assertEqual(completion.audio.chunks, ())
        self.assertEqual(stream.stopped, 1)
        self.assertEqual(stream.closed, 1)
        self.assertEqual(sounddevice.global_stop_calls, 0)

    def test_callback_propagates_status_and_capture_rejection_as_codes(self) -> None:
        sounddevice = _FakeSoundDevice()
        faults: list[AudioInputFault] = []
        owner = DedicatedAudioInput(
            _RejectingSink(),
            sounddevice_module=sounddevice,
            numpy_module=np,
            on_fault=faults.append,
        )
        owner.start()

        sounddevice.streams[0].emit(
            np.array([0.25], dtype=np.float32), status="overflow"
        )

        self.assertEqual(
            [fault.code for fault in faults],
            [
                AudioInputFaultCode.INPUT_STATUS,
                AudioInputFaultCode.CAPTURE_REJECTED,
            ],
        )

    def test_device_start_failure_is_cleaned_and_reported_without_details(self) -> None:
        sounddevice = _FakeSoundDevice(start_error=RuntimeError("device secret"))
        faults: list[AudioInputFault] = []
        owner = DedicatedAudioInput(
            CaptureService(),
            sounddevice_module=sounddevice,
            numpy_module=np,
            on_fault=faults.append,
        )

        with self.assertRaisesRegex(AudioInputError, "failed to start") as raised:
            owner.start()

        self.assertNotIn("device secret", str(raised.exception))
        self.assertEqual(faults[-1].code, AudioInputFaultCode.DEVICE_OPEN_FAILED)
        self.assertEqual(sounddevice.streams[0].closed, 1)
        self.assertFalse(owner.active)

    def test_start_timeout_cannot_late_adopt_a_stream(self) -> None:
        gate = threading.Event()
        sounddevice = _FakeSoundDevice(start_gate=gate)
        owner = DedicatedAudioInput(
            CaptureService(),
            sounddevice_module=sounddevice,
            numpy_module=np,
            lifecycle_timeout_seconds=0.02,
        )

        started_at = time.monotonic()
        with self.assertRaises(AudioInputError):
            owner.start()
        self.assertLess(time.monotonic() - started_at, 0.2)
        gate.set()

        self.assertTrue(_wait_until(lambda: sounddevice.streams[0].closed == 1))
        self.assertFalse(owner.active)

    def test_stuck_stop_is_bounded_taints_owner_and_never_calls_global_stop(self) -> None:
        stop_gate = threading.Event()
        sounddevice = _FakeSoundDevice(stop_gate=stop_gate, abort_hangs=True)
        faults: list[AudioInputFault] = []
        owner = DedicatedAudioInput(
            CaptureService(),
            sounddevice_module=sounddevice,
            numpy_module=np,
            on_fault=faults.append,
            lifecycle_timeout_seconds=0.02,
        )
        owner.start()

        started_at = time.monotonic()
        result = owner.stop()

        self.assertLess(time.monotonic() - started_at, 0.2)
        self.assertFalse(result.silence_confirmed)
        self.assertTrue(result.boundary_replacement_required)
        self.assertEqual(result.fault, AudioInputFaultCode.DEVICE_STOP_TIMEOUT)
        self.assertEqual(owner.stop(), result)
        with self.assertRaisesRegex(AudioInputError, "requires replacement"):
            owner.start()
        self.assertEqual(faults[-1].code, AudioInputFaultCode.DEVICE_STOP_TIMEOUT)
        self.assertEqual(sounddevice.global_stop_calls, 0)
        self.assertTrue(_wait_until(lambda: sounddevice.streams[0].aborted == 1))
        stop_gate.set()

    def test_concurrent_start_calls_create_one_owner(self) -> None:
        sounddevice = _FakeSoundDevice()
        owner = DedicatedAudioInput(
            CaptureService(),
            sounddevice_module=sounddevice,
            numpy_module=np,
        )
        barrier = threading.Barrier(8)
        results: list[bool] = []

        def start() -> None:
            barrier.wait()
            results.append(owner.start())

        threads = [threading.Thread(target=start) for _ in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(1.0)

        self.assertEqual(results.count(True), 1)
        self.assertEqual(results.count(False), 7)
        self.assertEqual(len(sounddevice.streams), 1)


class Float32AudioAssemblerTests(unittest.TestCase):
    def test_losslessly_flattens_and_concatenates_float32_blocks(self) -> None:
        first = np.array([[0.25], [-0.5]], dtype=np.float32)
        second = np.array([0.75], dtype=np.float32)

        assembled = Float32AudioAssembler(np)((first, second))
        first[:] = 1.0

        np.testing.assert_array_equal(
            assembled, np.array([0.25, -0.5, 0.75], dtype=np.float32)
        )
        self.assertEqual(assembled.ndim, 1)
        self.assertEqual(assembled.dtype, np.float32)
        self.assertTrue(assembled.flags.c_contiguous)

    def test_empty_input_is_one_dimensional_float32(self) -> None:
        assembled = Float32AudioAssembler(np)(())

        self.assertEqual(assembled.shape, (0,))
        self.assertEqual(assembled.dtype, np.float32)

    def test_rejects_non_float32_blocks_instead_of_lossy_conversion(self) -> None:
        assembler = Float32AudioAssembler(np)

        for dtype in (np.float64, np.int16):
            with self.subTest(dtype=dtype):
                with self.assertRaisesRegex(ValueError, "float32"):
                    assembler((np.array([1], dtype=dtype),))


if __name__ == "__main__":
    unittest.main()
