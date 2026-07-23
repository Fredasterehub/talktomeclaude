"""Deterministic capture contract tests; no microphone or Win32 required."""

from __future__ import annotations

import unittest
import threading
import time
from dataclasses import replace
from types import SimpleNamespace
from unittest import mock

from talktomeclaude.capture import (
    CaptureCancelled,
    CaptureContractError,
    CaptureEnd,
    CaptureMode,
    CapturePhase,
    CaptureService,
    CaptureSettings,
    SafetyNoticeCode,
    SnapshotCallableAdapter,
    TranscriptDisposition,
    Transcription,
)
from talktomeclaude import listen
from talktomeclaude.stt import CPU_TIER
from talktomeclaude.core.deadlines import DEFAULT_DEADLINES, DeadlineName


class _Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


class _Resolver:
    def __init__(self, events: list[object]) -> None:
        self.events = events

    def resolve_foreground(self, request):
        self.events.append(("snapshot", request))
        return {"opaque": request.turn_id}


class _Transcriber:
    def __init__(self, result, events: list[object] | None = None) -> None:
        self.result = result
        self.events = events

    def transcribe(self, audio):
        if self.events is not None:
            self.events.append(("stt", audio))
        return self.result


class CaptureStateTests(unittest.TestCase):
    def test_push_toggle_is_default_and_silence_never_finishes_the_turn(self) -> None:
        clock = _Clock()
        service = CaptureService(clock=clock)

        turn_id = service.toggle()
        service.add_audio(b"before pause")
        clock.now = 10.0 * 60.0
        progress = service.poll()
        service.add_audio(b"after pause")

        self.assertEqual(turn_id, 1)
        self.assertEqual(service.settings.mode, CaptureMode.PUSH_TOGGLE)
        self.assertEqual(progress.phase, CapturePhase.RECORDING)
        self.assertIsNone(progress.completion)
        completion = service.toggle()
        self.assertEqual(completion.audio.chunks, (b"before pause", b"after pause"))
        self.assertEqual(completion.audio.ended_by, CaptureEnd.FINISH_TOGGLE)

    def test_capture_and_stt_defaults_match_the_core_deadline_contract(self) -> None:
        service = CaptureService()

        self.assertEqual(
            service.settings.safety_ceiling_seconds,
            DEFAULT_DEADLINES[DeadlineName.CAPTURE_SAFETY_CEILING].seconds,
        )
        self.assertEqual(
            service._stt_construction_timeout_seconds,
            DEFAULT_DEADLINES[DeadlineName.STT_CONSTRUCTION].seconds,
        )
        self.assertEqual(
            service._stt_iteration_timeout_seconds,
            DEFAULT_DEADLINES[DeadlineName.STT_ITERATION].seconds,
        )

    def test_finish_toggle_resolves_exactly_once_and_start_evidence_is_diagnostic(self) -> None:
        events: list[object] = []
        diagnostic = {"window": "old"}
        service = CaptureService(
            snapshot_resolver=_Resolver(events),
            record_start_probe=lambda: diagnostic.copy(),
        )
        service.toggle()
        service.add_audio(b"audio")
        diagnostic["window"] = "new"

        completion = service.toggle()

        self.assertEqual(len(events), 1)
        self.assertEqual(events[0][0], "snapshot")
        self.assertTrue(completion.snapshot_request.ephemeral)
        self.assertEqual(completion.finish_snapshot, {"opaque": 1})
        self.assertEqual(
            completion.audio.record_start_diagnostic, {"window": "old"}
        )
        self.assertNotEqual(
            completion.audio.record_start_diagnostic, completion.finish_snapshot
        )

    def test_platform_snapshot_method_has_an_explicit_capture_adapter(self) -> None:
        snapshot = mock.Mock(return_value={"opaque": "target"})
        service = CaptureService(
            snapshot_resolver=SnapshotCallableAdapter(snapshot)
        )

        service.toggle()
        completion = service.toggle()

        snapshot.assert_called_once_with()
        self.assertEqual(completion.finish_snapshot, {"opaque": "target"})

    def test_hold_to_talk_is_independent_and_release_bounded(self) -> None:
        events: list[object] = []
        service = CaptureService(
            settings=CaptureSettings(mode=CaptureMode.HOLD_TO_TALK),
            snapshot_resolver=_Resolver(events),
        )

        service.start()
        service.add_audio(b"held")
        completion = service.release()

        self.assertEqual(completion.audio.mode, CaptureMode.HOLD_TO_TALK)
        self.assertEqual(completion.audio.ended_by, CaptureEnd.KEY_RELEASE)
        self.assertIsNotNone(completion.snapshot_request)
        self.assertTrue(completion.snapshot_request.ephemeral)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0][0], "snapshot")

    def test_safety_ceiling_warns_then_preserves_audio_for_recovery(self) -> None:
        clock = _Clock()
        service = CaptureService(
            settings=CaptureSettings(
                safety_ceiling_seconds=20.0,
                warning_before_seconds=5.0,
            ),
            clock=clock,
        )
        service.toggle()
        service.add_audio(b"important audio")

        clock.now = 15.0
        warning = service.poll()
        clock.now = 20.0
        stopped = service.poll()
        recovered = service.consume_preserved(stopped.completion)

        self.assertEqual(
            [notice.code for notice in warning.notices],
            [SafetyNoticeCode.CEILING_APPROACHING],
        )
        self.assertEqual(stopped.phase, CapturePhase.PRESERVED)
        self.assertEqual(stopped.completion.audio.chunks, (b"important audio",))
        self.assertEqual(stopped.completion.audio.ended_by, CaptureEnd.SAFETY_CEILING)
        self.assertEqual(
            [notice.code for notice in recovered.audio.notices],
            [SafetyNoticeCode.CEILING_APPROACHING, SafetyNoticeCode.CEILING_REACHED],
        )
        self.assertEqual(service.phase, CapturePhase.IDLE)
        self.assertIs(recovered, stopped.completion)

    def test_preserved_discard_requires_exact_identity_and_retains_on_failure(self) -> None:
        clock = _Clock()
        service = CaptureService(
            settings=CaptureSettings(
                safety_ceiling_seconds=2.0,
                warning_before_seconds=1.0,
            ),
            clock=clock,
        )
        service.toggle()
        service.add_audio(b"discard-secret")
        clock.now = 2.0
        completion = service.poll().completion
        equal_but_not_identical = replace(completion)

        with self.assertRaisesRegex(CaptureContractError, "identity"):
            service.discard_preserved(equal_but_not_identical)
        self.assertEqual(service.phase, CapturePhase.PRESERVED)

        discarded_turn = service.discard_preserved(completion)

        self.assertEqual(discarded_turn, completion.audio.turn_id)
        self.assertEqual(service.phase, CapturePhase.IDLE)
        with self.assertRaisesRegex(CaptureContractError, "no preserved"):
            service.discard_preserved(completion)

    def test_finish_toggle_and_ceiling_race_finalizes_the_take_once(self) -> None:
        clock = _Clock()
        events: list[object] = []
        service = CaptureService(
            settings=CaptureSettings(
                safety_ceiling_seconds=20.0,
                warning_before_seconds=5.0,
            ),
            clock=clock,
            snapshot_resolver=_Resolver(events),
        )
        service.toggle()
        service.add_audio(b"once")
        clock.now = 20.0
        barrier = threading.Barrier(3)
        completions: list[object] = []
        errors: list[BaseException] = []

        def finish_toggle() -> None:
            barrier.wait()
            try:
                result = service.toggle()
            except CaptureContractError:
                return
            except BaseException as exc:
                errors.append(exc)
                return
            if not isinstance(result, int):
                completions.append(result)

        def hit_ceiling() -> None:
            barrier.wait()
            result = service.poll()
            if result.completion is not None:
                completions.append(result.completion)

        threads = [
            threading.Thread(target=finish_toggle),
            threading.Thread(target=hit_ceiling),
        ]
        for thread in threads:
            thread.start()
        barrier.wait()
        for thread in threads:
            thread.join(1.0)

        self.assertEqual(len(completions), 1)
        self.assertEqual(errors, [])
        self.assertEqual(completions[0].audio.chunks, (b"once",))
        self.assertLessEqual(len(events), 1)
        # Polling preserved state does not emit the completion a second time.
        self.assertIsNone(service.poll().completion)


class TranscriptGateTests(unittest.TestCase):
    def _completion(self, service: CaptureService):
        service.toggle()
        service.add_audio(b"audio")
        return service.toggle()

    def test_empty_and_low_confidence_results_are_preserved_but_not_deliverable(self) -> None:
        service = CaptureService(settings=CaptureSettings(minimum_confidence=0.70))
        empty_completion = self._completion(service)
        empty = service.transcribe(
            empty_completion,
            lambda _cancelled: _Transcriber(Transcription("  ", 0.99)),
        )
        low_completion = self._completion(service)
        low = service.transcribe(
            low_completion,
            lambda _cancelled: _Transcriber(Transcription("keep me editable", 0.69)),
        )

        self.assertEqual(empty.transcript.disposition, TranscriptDisposition.EMPTY)
        self.assertFalse(empty.transcript.may_deliver)
        self.assertEqual(empty.transcript.text, "  ")
        self.assertEqual(
            low.transcript.disposition, TranscriptDisposition.LOW_CONFIDENCE
        )
        self.assertFalse(low.transcript.may_deliver)
        self.assertEqual(low.transcript.text, "keep me editable")

    def test_finish_snapshot_precedes_stt_and_acceptable_text_needs_no_confirmation(self) -> None:
        events: list[object] = []
        service = CaptureService(snapshot_resolver=_Resolver(events))
        completion = self._completion(service)
        result = service.transcribe(
            completion,
            lambda _cancelled: _Transcriber(
                Transcription("Café 世界 \N{WAVING HAND SIGN}", 0.95), events
            ),
        )

        self.assertEqual([event[0] for event in events], ["snapshot", "stt"])
        self.assertEqual(result.transcript.disposition, TranscriptDisposition.ACCEPTED)
        self.assertTrue(result.transcript.may_deliver)
        self.assertEqual(
            result.transcript.text, "Café 世界 \N{WAVING HAND SIGN}"
        )

    def test_cancellation_before_and_after_model_construction_skips_stt(self) -> None:
        service = CaptureService()
        completion = self._completion(service)
        pre_factory = mock.Mock()

        pre = service.transcribe(
            completion,
            pre_factory,
            cancelled=lambda: True,
        )

        pre_factory.assert_not_called()
        self.assertEqual(pre.transcript.disposition, TranscriptDisposition.CANCELLED)
        self.assertFalse(pre.boundary_replacement_required)

        cancelled = {"value": False}
        engine = mock.Mock()

        def construct(_probe):
            cancelled["value"] = True
            return engine

        post = service.transcribe(
            completion,
            construct,
            cancelled=lambda: cancelled["value"],
        )

        engine.transcribe.assert_not_called()
        self.assertEqual(post.transcript.disposition, TranscriptDisposition.CANCELLED)
        self.assertTrue(post.boundary_replacement_required)

    def test_stop_during_noncooperative_construction_discards_late_engine(self) -> None:
        service = CaptureService()
        completion = self._completion(service)
        entered = threading.Event()
        release = threading.Event()
        cancelled = threading.Event()
        engine = mock.Mock()
        results: list[object] = []

        def construct(_probe):
            entered.set()
            release.wait(1.0)  # Deliberately ignores the cancellation probe.
            return engine

        thread = threading.Thread(
            target=lambda: results.append(
                service.transcribe(
                    completion,
                    construct,
                    cancelled=cancelled.is_set,
                )
            )
        )
        thread.start()
        self.assertTrue(entered.wait(1.0))
        cancelled.set()
        release.set()
        thread.join(1.0)

        self.assertFalse(thread.is_alive())
        self.assertEqual(len(results), 1)
        engine.transcribe.assert_not_called()
        self.assertEqual(
            results[0].transcript.disposition, TranscriptDisposition.CANCELLED
        )
        self.assertTrue(results[0].boundary_replacement_required)

    def test_hung_construction_returns_by_deadline_and_stays_tainted(self) -> None:
        release = threading.Event()
        constructed = threading.Event()
        service = CaptureService(stt_construction_timeout_seconds=0.03)
        completion = self._completion(service)
        late_engine = mock.Mock()

        def hung_factory(_probe):
            release.wait(1.0)
            constructed.set()
            return late_engine

        started = time.monotonic()
        result = service.transcribe(completion, hung_factory)
        elapsed = time.monotonic() - started

        self.assertLess(elapsed, 0.30)
        self.assertEqual(
            result.transcript.disposition,
            TranscriptDisposition.CONSTRUCTION_TIMEOUT,
        )
        self.assertEqual(result.error_code, "stt_construction_timeout")
        self.assertTrue(result.boundary_replacement_required)
        self.assertTrue(service.boundary_replacement_required)

        release.set()
        self.assertTrue(constructed.wait(1.0))
        retry_factory = mock.Mock()
        retry = service.transcribe(completion, retry_factory)
        retry_factory.assert_not_called()
        self.assertEqual(
            retry.transcript.disposition, TranscriptDisposition.BOUNDARY_TAINTED
        )
        self.assertEqual(retry.error_code, "stt_boundary_tainted")
        self.assertFalse(retry.transcript.may_deliver)

    def test_hung_lazy_next_returns_by_iteration_deadline_with_no_late_acceptance(self) -> None:
        release = threading.Event()
        late_finished = threading.Event()
        service = CaptureService(stt_iteration_timeout_seconds=0.03)
        completion = self._completion(service)

        def segments():
            release.wait(1.0)
            yield SimpleNamespace(text="late secret transcript")
            late_finished.set()

        model = mock.Mock()
        model.transcribe.return_value = (segments(), None)

        def factory(probe):
            with mock.patch.object(
                listen, "detect_tier", return_value=CPU_TIER
            ), mock.patch.object(
                listen.UtteranceTranscriber, "_load", return_value=model
            ):
                return listen.UtteranceTranscriber("cpu", cancelled=probe)

        started = time.monotonic()
        result = service.transcribe(completion, factory)
        elapsed = time.monotonic() - started

        self.assertLess(elapsed, 0.30)
        self.assertEqual(
            result.transcript.disposition, TranscriptDisposition.ITERATION_TIMEOUT
        )
        self.assertEqual(result.error_code, "stt_iteration_timeout")
        self.assertTrue(result.boundary_replacement_required)
        self.assertFalse(result.transcript.may_deliver)

        release.set()
        self.assertTrue(late_finished.wait(1.0))
        self.assertTrue(service.boundary_replacement_required)
        self.assertEqual(
            service.transcribe(completion, mock.Mock()).transcript.disposition,
            TranscriptDisposition.BOUNDARY_TAINTED,
        )


class PrivacyRepresentationTests(unittest.TestCase):
    def _completion(self, service: CaptureService):
        service.toggle()
        service.add_audio(b"audio")
        return service.toggle()

    def test_capture_results_redact_audio_targets_diagnostics_and_text(self) -> None:
        chunk_secret = "SYNTHETIC-CHUNK-SECRET"
        diagnostic_secret = "SYNTHETIC-DIAGNOSTIC-SECRET"
        target_secret = "SYNTHETIC-TARGET-SECRET"
        transcript_secret = "SYNTHETIC-TRANSCRIPT-SECRET"
        service = CaptureService(
            snapshot_resolver=SnapshotCallableAdapter(
                lambda: {"target": target_secret}
            ),
            record_start_probe=lambda: {"diagnostic": diagnostic_secret},
        )
        service.toggle()
        service.add_audio(chunk_secret.encode())
        completion = service.toggle()
        turn = service.transcribe(
            completion,
            lambda _cancelled: _Transcriber(transcript_secret),
        )

        representations = (
            repr(Transcription(transcript_secret)),
            repr(completion.audio),
            repr(completion),
            repr(turn.transcript),
            repr(turn),
        )
        for rendered in representations:
            self.assertNotIn(chunk_secret, rendered)
            self.assertNotIn(diagnostic_secret, rendered)
            self.assertNotIn(target_secret, rendered)
            self.assertNotIn(transcript_secret, rendered)

    def test_injected_capture_classifier_owns_below_equal_above_threshold(self) -> None:
        cases = (
            (0.69, TranscriptDisposition.LOW_CONFIDENCE),
            (0.70, TranscriptDisposition.ACCEPTED),
            (0.71, TranscriptDisposition.ACCEPTED),
        )
        for confidence, expected in cases:
            seen: list[str] = []
            service = CaptureService(
                settings=CaptureSettings(minimum_confidence=0.70),
                transcript_classifier=lambda text, value=confidence: (
                    seen.append(text) or value
                ),
            )
            completion = self._completion(service)
            result = service.transcribe(
                completion,
                lambda _cancelled: _Transcriber("not command intent"),
            )

            self.assertEqual(result.transcript.disposition, expected)
            self.assertEqual(result.transcript.confidence, confidence)
            self.assertEqual(seen, ["not command intent"])

    def test_stale_turn_is_rejected_without_constructing_stt(self) -> None:
        service = CaptureService()
        old_completion = self._completion(service)
        service.toggle()  # A new generation makes the completed take stale.
        factory = mock.Mock()

        result = service.transcribe(old_completion, factory)

        factory.assert_not_called()
        self.assertEqual(
            result.transcript.disposition, TranscriptDisposition.STALE_GENERATION
        )
        self.assertFalse(result.transcript.may_deliver)

    def test_lazy_cancellation_returns_no_deliverable_partial_transcript(self) -> None:
        service = CaptureService()
        completion = self._completion(service)
        cancelled = {"value": False}

        def segments():
            yield SimpleNamespace(text="partial")
            cancelled["value"] = True
            yield SimpleNamespace(text="must not be accepted")

        model = mock.Mock()
        model.transcribe.return_value = (segments(), None)

        def factory(probe):
            with mock.patch.object(
                listen, "detect_tier", return_value=CPU_TIER
            ), mock.patch.object(
                listen.UtteranceTranscriber, "_load", return_value=model
            ):
                return listen.UtteranceTranscriber("cpu", cancelled=probe)

        result = service.transcribe(
            completion,
            factory,
            cancelled=lambda: cancelled["value"],
        )

        self.assertEqual(result.transcript.disposition, TranscriptDisposition.CANCELLED)
        self.assertFalse(result.transcript.may_deliver)
        self.assertEqual(result.transcript.text, "")


class WhisperCancellationBoundaryTests(unittest.TestCase):
    def test_model_construction_honors_cancellation_requested_while_loading(self) -> None:
        cancelled = {"value": False}

        def load(_tier):
            cancelled["value"] = True
            return mock.Mock()

        with mock.patch.object(listen, "detect_tier", return_value=CPU_TIER), mock.patch.object(
            listen.UtteranceTranscriber, "_load", side_effect=load
        ):
            with self.assertRaises(CaptureCancelled):
                listen.UtteranceTranscriber(
                    "cpu", cancelled=lambda: cancelled["value"]
                )

    def test_lazy_segment_iteration_checks_cancellation_after_each_next(self) -> None:
        cancelled = {"value": False}
        consumed: list[str] = []

        def segments():
            consumed.append("first")
            yield SimpleNamespace(text="first")
            cancelled["value"] = True
            consumed.append("second")
            yield SimpleNamespace(text="second")

        model = mock.Mock()
        model.transcribe.return_value = (segments(), None)
        with mock.patch.object(listen, "detect_tier", return_value=CPU_TIER), mock.patch.object(
            listen.UtteranceTranscriber, "_load", return_value=model
        ):
            transcriber = listen.UtteranceTranscriber(
                "cpu", cancelled=lambda: cancelled["value"]
            )
            with self.assertRaises(CaptureCancelled):
                transcriber.transcribe(object())

        self.assertEqual(consumed, ["first", "second"])


if __name__ == "__main__":
    unittest.main()
