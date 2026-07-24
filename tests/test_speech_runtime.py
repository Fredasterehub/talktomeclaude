from __future__ import annotations

import threading
import time
import tempfile
import unittest
from pathlib import Path

from talktomeclaude.speech.runtime import (
    PersistentSpeechRuntime,
    SpawnSynthesisWorker,
    SpeechArtifact,
    SpeechFaultCode,
    SpeechRuntimeError,
    SynthesisRequest,
    SynthesisResult,
)


def _fake_spawn_synthesize(text: str, path: Path, selected_voice: str) -> None:
    path.write_bytes(f"{selected_voice}|{text}".encode("utf-8"))


def _fake_spawn_failure(_text: str, path: Path, _selected_voice: str) -> None:
    path.write_bytes(b"partial")
    raise RuntimeError("model failure")


class _Worker:
    def __init__(
        self,
        *,
        wait_forever: bool = False,
        lifecycle_forever: bool = False,
        submit_forever: bool = False,
    ) -> None:
        self.wait_forever = wait_forever
        self.lifecycle_forever = lifecycle_forever
        self.submit_forever = submit_forever
        self.requests: list[tuple[SynthesisRequest, object]] = []
        self.terminated = 0
        self.killed = 0

    def submit(self, request: SynthesisRequest, callback: object) -> None:
        self.requests.append((request, callback))
        if self.submit_forever:
            threading.Event().wait(5)

    def terminate(self) -> None:
        self.terminated += 1
        if self.lifecycle_forever:
            threading.Event().wait(5)

    def kill(self) -> None:
        self.killed += 1
        if self.lifecycle_forever:
            threading.Event().wait(5)

    def wait(self, timeout: float) -> int:
        if self.wait_forever:
            threading.Event().wait(5)
        return 0


class PersistentSpeechRuntimeTests(unittest.TestCase):
    def test_persistent_boundary_submits_without_exposing_text_or_switching_voice(
        self,
    ) -> None:
        voices: list[str] = []
        workers: list[_Worker] = []

        def factory(voice: str) -> _Worker:
            voices.append(voice)
            worker = _Worker()
            workers.append(worker)
            return worker

        runtime = PersistentSpeechRuntime("rick", factory)
        request = SynthesisRequest(7, "unit-1", "SECRET full canonical text")
        accepted: list[SynthesisResult] = []
        runtime.submit(request, accepted.append)

        self.assertEqual(voices, ["rick"])
        self.assertEqual(workers[0].requests[0][0], request)
        self.assertNotIn("SECRET", repr(request))
        self.assertEqual(runtime.selected_voice, "rick")
        self.assertEqual(accepted, [])

    def test_graceful_restart_preserves_exact_voice_and_parent_state_identity(self) -> None:
        workers: list[_Worker] = []
        voices: list[str] = []

        def factory(voice: str) -> _Worker:
            voices.append(voice)
            worker = _Worker()
            workers.append(worker)
            return worker

        runtime = PersistentSpeechRuntime(
            "gimli", factory, shutdown_deadline_seconds=0.05
        )
        parent_state = {"cursor": 12, "canonical": object()}

        result = runtime.restart(parent_state)

        self.assertIs(result.parent_state, parent_state)
        self.assertEqual(result.selected_voice, "gimli")
        self.assertEqual(voices, ["gimli", "gimli"])
        self.assertTrue(result.old_worker_reaped)
        self.assertTrue(result.terminate_sent)
        self.assertFalse(result.kill_sent)
        self.assertFalse(result.boundary_replacement_required)
        self.assertEqual(workers[0].terminated, 1)
        self.assertEqual(workers[0].killed, 0)

    def test_noncooperative_worker_is_bounded_unavailable_and_never_replaced(self) -> None:
        workers: list[_Worker] = []
        voices: list[str] = []

        def factory(voice: str) -> _Worker:
            voices.append(voice)
            worker = _Worker(
                wait_forever=not workers,
                lifecycle_forever=not workers,
            )
            workers.append(worker)
            return worker

        runtime = PersistentSpeechRuntime(
            "rick", factory, shutdown_deadline_seconds=0.01
        )
        parent_state = object()
        started = time.monotonic()

        with self.assertRaises(SpeechRuntimeError):
            runtime.restart(parent_state)

        self.assertLess(time.monotonic() - started, 0.2)
        self.assertEqual(voices, ["rick"])
        self.assertEqual(workers[0].terminated, 1)
        with self.assertRaises(SpeechRuntimeError):
            runtime.submit(
                SynthesisRequest(0, "unit", "parent remains external"),
                lambda _result: None,
            )
        self.assertIsNotNone(parent_state)

    def test_restart_failure_is_content_free_and_never_tries_fallback_voice(self) -> None:
        voices: list[str] = []
        calls = 0

        def factory(voice: str) -> _Worker:
            nonlocal calls
            calls += 1
            voices.append(voice)
            if calls == 2:
                raise RuntimeError("SECRET voice model path")
            return _Worker()

        runtime = PersistentSpeechRuntime("rick", factory)

        with self.assertRaises(SpeechRuntimeError) as raised:
            runtime.restart({"text": "SECRET answer"})

        self.assertEqual(runtime.selected_voice, "rick")
        self.assertEqual(voices, ["rick", "rick"])
        self.assertNotIn("SECRET", str(raised.exception))

    def test_noncooperative_restart_factory_is_bounded_and_old_worker_not_reused(
        self,
    ) -> None:
        voices: list[str] = []
        release = threading.Event()

        def factory(voice: str) -> _Worker:
            voices.append(voice)
            if len(voices) == 2:
                release.wait(2)
            return _Worker()

        runtime = PersistentSpeechRuntime(
            "rick", factory, shutdown_deadline_seconds=0.01
        )
        started = time.monotonic()

        with self.assertRaises(SpeechRuntimeError):
            runtime.restart(object())

        self.assertLess(time.monotonic() - started, 0.2)
        self.assertEqual(runtime.selected_voice, "rick")
        with self.assertRaises(SpeechRuntimeError):
            runtime.submit(SynthesisRequest(0, "unit", "text"), lambda _result: None)
        self.assertEqual(voices, ["rick", "rick"])
        release.set()

    def test_artifact_discard_is_idempotent_and_cleanup_failure_is_contained(self) -> None:
        calls: list[object] = []

        def discard(payload: object) -> None:
            calls.append(payload)
            raise RuntimeError("cleanup")

        payload = object()
        artifact = SpeechArtifact(
            generation=3, unit_id="unit", payload=payload, discard=discard
        )

        self.assertTrue(artifact.discard())
        self.assertFalse(artifact.discard())
        self.assertEqual(calls, [payload])
        self.assertNotIn(str(payload), repr(artifact))

    def test_initial_factory_and_submission_each_fail_closed_within_budget(self) -> None:
        release = threading.Event()

        def stuck_factory(_voice: str) -> _Worker:
            release.wait(2)
            return _Worker()

        started = time.monotonic()
        with self.assertRaises(SpeechRuntimeError):
            PersistentSpeechRuntime(
                "rick", stuck_factory, shutdown_deadline_seconds=0.01
            )
        self.assertLess(time.monotonic() - started, 0.2)
        release.set()

        worker = _Worker(submit_forever=True)
        runtime = PersistentSpeechRuntime(
            "rick", lambda _voice: worker, shutdown_deadline_seconds=0.01
        )
        started = time.monotonic()
        with self.assertRaises(SpeechRuntimeError):
            runtime.submit(
                SynthesisRequest(0, "unit", "full text"), lambda _result: None
            )
        self.assertLess(time.monotonic() - started, 0.2)
        with self.assertRaises(SpeechRuntimeError):
            runtime.submit(
                SynthesisRequest(0, "later", "later"), lambda _result: None
            )

    def test_spawn_worker_synthesizes_with_fixed_voice_and_cleans_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            worker = SpawnSynthesisWorker(
                "rick",
                artifact_root=directory,
                synthesize_fn=_fake_spawn_synthesize,
            )
            request = SynthesisRequest(4, "spawn-unit", "model-free Unicode café")
            results: list[SynthesisResult] = []
            ready = threading.Event()

            def accept(result: SynthesisResult) -> None:
                results.append(result)
                ready.set()

            worker.submit(request, accept)
            self.assertTrue(ready.wait(10))
            self.assertEqual(len(results), 1)
            artifact = results[0].artifact
            assert artifact is not None
            path = artifact.payload
            self.assertIsInstance(path, Path)
            assert isinstance(path, Path)
            self.assertEqual(path.read_bytes(), "rick|model-free Unicode café".encode())
            self.assertFalse(tuple(Path(directory).glob("*.tmp.wav")))
            artifact.discard()
            self.assertFalse(path.exists())
            worker.terminate()
            worker.wait(5)

    def test_spawn_worker_failure_removes_partial_temp_and_returns_content_free_fault(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            worker = SpawnSynthesisWorker(
                "gimli",
                artifact_root=directory,
                synthesize_fn=_fake_spawn_failure,
            )
            request = SynthesisRequest(1, "failed-unit", "SECRET answer")
            results: list[SynthesisResult] = []
            ready = threading.Event()

            def accept(result: SynthesisResult) -> None:
                results.append(result)
                ready.set()

            worker.submit(request, accept)
            self.assertTrue(ready.wait(10))
            self.assertEqual(results[0].fault, SpeechFaultCode.SYNTHESIS_FAILED)
            self.assertNotIn("SECRET", repr(results[0]))
            self.assertEqual(tuple(Path(directory).iterdir()), ())
            worker.terminate()
            worker.wait(5)


if __name__ == "__main__":
    unittest.main()
