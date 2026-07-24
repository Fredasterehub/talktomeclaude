"""Virtual integration tests for the capture-to-Windows-delivery boundary."""

from __future__ import annotations

import gc
import threading
import unittest
import weakref
from collections import deque
from dataclasses import dataclass

from talktomeclaude.capture import (
    CaptureMode,
    CapturePhase,
    CaptureService,
    CaptureSettings,
    SnapshotCallableAdapter,
    TranscriptDisposition,
    Transcription,
)
from talktomeclaude.companion.capture_delivery import (
    CaptureDeliveryCode,
    CaptureDeliveryCoordinator,
)
from talktomeclaude.core import RuntimeCoordinator, RuntimePhase, RuntimeState
from talktomeclaude.platform.contracts import (
    DeliveryCode,
    DeliveryMode,
    DeliveryResult,
    RestoreStatus,
)
from talktomeclaude.platform.windows.clipboard import ClipboardCode, ClipboardOperation
from talktomeclaude.platform.windows.injector import (
    KeySendCode,
    KeySendOutcome,
    TextInjector,
)
from talktomeclaude.platform.windows.target import (
    TargetCode,
    TargetEvidence,
    TargetResolution,
    TargetValidation,
)


EVIDENCE = TargetEvidence(
    101,
    202,
    "WindowsTerminal.exe",
    "CASCADIA_HOSTING_WINDOW_CLASS",
    "windows_terminal",
)
OTHER_EVIDENCE = TargetEvidence(
    303,
    404,
    "pwsh.exe",
    "ConsoleWindowClass",
    "console_host",
)
VALID = TargetValidation(True, TargetCode.VALID)


class _Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


class _Resolver:
    def __init__(
        self,
        events: list[str],
        *,
        resolutions: list[TargetResolution] | None = None,
        validations: list[TargetValidation] | None = None,
    ) -> None:
        self.events = events
        self.resolutions = deque(
            resolutions or [TargetResolution(EVIDENCE, TargetCode.VALID)]
        )
        self.validations = deque(validations or [VALID, VALID, VALID])
        self.seen: list[TargetEvidence] = []
        self.snapshot_calls = 0

    def snapshot(self) -> TargetResolution:
        self.events.append("target.snapshot")
        self.snapshot_calls += 1
        return self.resolutions.popleft()

    def validate(self, evidence: TargetEvidence) -> TargetValidation:
        self.events.append("target.validate")
        self.seen.append(evidence)
        return self.validations.popleft()


class _Clipboard:
    def __init__(self, events: list[str]) -> None:
        self.events = events
        self.text: str | None = None

    def snapshot(self) -> ClipboardOperation:
        self.events.append("clipboard.snapshot")
        return ClipboardOperation(ClipboardCode.OK)

    def set_text(self, text: str) -> ClipboardOperation:
        self.events.append("clipboard.set")
        self.text = text
        return ClipboardOperation(ClipboardCode.OK)

    def restore(self) -> RestoreStatus:
        self.events.append("clipboard.restore")
        return RestoreStatus.RESTORED


class _Keyboard:
    def __init__(self, events: list[str]) -> None:
        self.events = events

    def send_paste(self) -> KeySendOutcome:
        self.events.append("keyboard.paste")
        return KeySendOutcome(KeySendCode.SENT, 4, 4, 0.001)

    def send_enter(self) -> KeySendOutcome:
        self.events.append("keyboard.enter")
        return KeySendOutcome(KeySendCode.SENT, 2, 2, 0.001)


class _Transcriber:
    def __init__(self, result: object, events: list[str]) -> None:
        self.result = result
        self.events = events

    def transcribe(self, audio: object) -> object:
        self.events.append("stt.transcribe")
        return self.result


def _factory(result: object, events: list[str]):
    def create(_cancelled):
        events.append("stt.construct")
        return _Transcriber(result, events)

    return create


def _vertical(
    *,
    transcript: object = Transcription("hello", 0.9),
    resolutions: list[TargetResolution] | None = None,
    validations: list[TargetValidation] | None = None,
    settings: CaptureSettings | None = None,
    clock: _Clock | None = None,
):
    events: list[str] = []
    resolver = _Resolver(
        events,
        resolutions=resolutions,
        validations=validations,
    )
    clipboard = _Clipboard(events)
    injector = TextInjector(
        resolver=resolver,  # type: ignore[arg-type]
        keyboard=_Keyboard(events),
        clipboard_factory=lambda: clipboard,  # type: ignore[arg-type]
    )
    capture = CaptureService(
        settings=settings,
        clock=clock or _Clock(),
        snapshot_resolver=SnapshotCallableAdapter(injector.snapshot_target),
    )
    coordinator = CaptureDeliveryCoordinator(capture, injector)
    return coordinator, resolver, clipboard, events, _factory(transcript, events)


class CaptureDeliveryIntegrationTests(unittest.TestCase):
    def test_finish_snapshot_precedes_stt_and_same_evidence_reaches_all_boundaries(self) -> None:
        secret = "never-log-مرحبا-🙂-e\u0301"
        coordinator, resolver, clipboard, events, factory = _vertical(
            transcript=Transcription(secret, 0.99)
        )
        coordinator.start()
        coordinator.add_audio(b"audio")

        result = coordinator.finish_toggle(
            factory,
            mode=DeliveryMode.ASSISTANT,
            auto_submit=True,
        )

        self.assertEqual(result.code, CaptureDeliveryCode.DELIVERED)
        self.assertEqual(result.runtime_phase, RuntimePhase.WAITING_FOR_CLAUDE)
        self.assertEqual(clipboard.text, secret)
        self.assertEqual(
            events,
            [
                "target.snapshot",
                "stt.construct",
                "stt.transcribe",
                "target.validate",
                "clipboard.snapshot",
                "clipboard.set",
                "target.validate",
                "keyboard.paste",
                "target.validate",
                "keyboard.enter",
                "clipboard.restore",
            ],
        )
        self.assertEqual(resolver.snapshot_calls, 1)
        self.assertEqual(len(resolver.seen), 3)
        self.assertTrue(all(item is EVIDENCE for item in resolver.seen))
        self.assertNotIn(secret, repr(result))
        self.assertNotIn(secret, repr(result.diagnostics))
        self.assertIsNone(result.transcript)

    def test_generic_and_assistant_off_pass_policy_without_enter(self) -> None:
        for mode, auto_submit in (
            (DeliveryMode.GENERIC, True),
            (DeliveryMode.ASSISTANT, False),
        ):
            with self.subTest(mode=mode, auto_submit=auto_submit):
                coordinator, _, _, events, factory = _vertical()
                coordinator.start()
                coordinator.add_audio(b"audio")
                result = coordinator.finish_toggle(
                    factory,
                    mode=mode,
                    auto_submit=auto_submit,
                )
                self.assertTrue(result.succeeded)
                self.assertEqual(events.count("keyboard.paste"), 1)
                self.assertNotIn("keyboard.enter", events)
                self.assertEqual(
                    result.runtime_phase,
                    RuntimePhase.IDLE
                    if mode is DeliveryMode.GENERIC
                    else RuntimePhase.WAITING_FOR_CLAUDE,
                )

    def test_consecutive_generic_turns_return_to_idle_between_deliveries(self) -> None:
        coordinator, resolver, _, events, factory = _vertical(
            resolutions=[
                TargetResolution(EVIDENCE, TargetCode.VALID),
                TargetResolution(OTHER_EVIDENCE, TargetCode.VALID),
            ],
            validations=[VALID, VALID, VALID, VALID],
        )

        first = coordinator.start()
        coordinator.add_audio(b"one")
        first_result = coordinator.finish_toggle(
            factory,
            mode=DeliveryMode.GENERIC,
            auto_submit=True,
        )
        second = coordinator.start()
        coordinator.add_audio(b"two")
        second_result = coordinator.finish_toggle(
            factory,
            mode=DeliveryMode.GENERIC,
            auto_submit=False,
        )

        self.assertGreater(second, first)
        self.assertEqual(first_result.runtime_phase, RuntimePhase.IDLE)
        self.assertEqual(second_result.runtime_phase, RuntimePhase.IDLE)
        self.assertEqual(resolver.snapshot_calls, 2)
        self.assertEqual(events.count("keyboard.paste"), 2)
        self.assertNotIn("keyboard.enter", events)

    def test_hold_release_snapshots_before_stt_and_delivers(self) -> None:
        coordinator, resolver, _, events, factory = _vertical()
        coordinator.start(CaptureMode.HOLD_TO_TALK)
        coordinator.add_audio(b"held")

        result = coordinator.release_hold(
            factory,
            mode=DeliveryMode.GENERIC,
            auto_submit=False,
        )

        self.assertTrue(result.succeeded)
        self.assertEqual(result.runtime_phase, RuntimePhase.IDLE)
        self.assertEqual(resolver.snapshot_calls, 1)
        self.assertLess(events.index("target.snapshot"), events.index("stt.construct"))

    def test_invalid_snapshot_fails_closed_without_entering_injector(self) -> None:
        events: list[str] = []

        class Injector:
            def snapshot_target(self):
                events.append("target.snapshot")
                # Even malformed evidence cannot override a non-VALID code.
                return TargetResolution(EVIDENCE, TargetCode.UNSUPPORTED)

            def deliver(self, *_args, **_kwargs):
                events.append("injector.deliver")
                raise AssertionError("invalid initial target must not enter injector")

        injector = Injector()
        capture = CaptureService(
            snapshot_resolver=SnapshotCallableAdapter(injector.snapshot_target)
        )
        coordinator = CaptureDeliveryCoordinator(capture, injector)
        coordinator.start()
        coordinator.add_audio(b"audio")

        result = coordinator.finish_toggle(
            _factory(Transcription("hello", 0.9), events),
            mode=DeliveryMode.ASSISTANT,
            auto_submit=True,
        )

        self.assertEqual(result.code, CaptureDeliveryCode.DELIVERY_FAILED)
        self.assertEqual(result.delivery.code, DeliveryCode.INVALID_TARGET)
        self.assertEqual(
            events, ["target.snapshot", "stt.construct", "stt.transcribe"]
        )

    def test_empty_and_low_confidence_are_visible_but_never_injected(self) -> None:
        secret = "private-low-confidence-text"
        for transcription, disposition in (
            (Transcription("   ", 1.0), TranscriptDisposition.EMPTY),
            (Transcription(secret, 0.1), TranscriptDisposition.LOW_CONFIDENCE),
        ):
            with self.subTest(disposition=disposition):
                coordinator, _, _, events, factory = _vertical(
                    transcript=transcription
                )
                coordinator.start()
                coordinator.add_audio(b"audio")
                result = coordinator.finish_toggle(
                    factory,
                    mode=DeliveryMode.ASSISTANT,
                    auto_submit=True,
                )
                self.assertEqual(result.code, CaptureDeliveryCode.REVIEW_REQUIRED)
                self.assertEqual(result.runtime_phase, RuntimePhase.AWAITING_CONFIRMATION)
                self.assertEqual(result.transcript.disposition, disposition)
                self.assertEqual(result.transcript.text, transcription.text)
                self.assertNotIn("target.validate", events)
                self.assertNotIn("clipboard.snapshot", events)
                self.assertNotIn("keyboard.paste", events)
                self.assertNotIn(secret, repr(result))
                self.assertNotIn(secret, repr(result.diagnostics))

    def test_safety_ceiling_preserves_transcript_for_fresh_recovery_without_injection(self) -> None:
        clock = _Clock()
        settings = CaptureSettings(
            safety_ceiling_seconds=10,
            warning_before_seconds=2,
        )
        coordinator, _, _, events, factory = _vertical(
            settings=settings,
            clock=clock,
        )
        coordinator.start()
        coordinator.add_audio(b"preserved")
        clock.now = 10
        progress = coordinator._capture.poll()
        assert progress.completion is not None

        result = coordinator.process_completion(
            progress.completion,
            factory,
            mode=DeliveryMode.ASSISTANT,
            auto_submit=True,
        )

        self.assertEqual(result.code, CaptureDeliveryCode.REVIEW_REQUIRED)
        self.assertEqual(result.runtime_phase, RuntimePhase.AWAITING_CONFIRMATION)
        self.assertEqual(coordinator._capture.phase, CapturePhase.IDLE)
        self.assertNotIn("target.validate", events)
        self.assertNotIn("keyboard.paste", events)

        next_turn = coordinator.start()
        self.assertEqual(coordinator.runtime.state.phase, RuntimePhase.RECORDING)
        self.assertEqual(coordinator._capture.phase, CapturePhase.RECORDING)
        self.assertGreater(next_turn, progress.completion.audio.turn_id)

    def test_cancelled_safety_recovery_discards_preserved_take_and_returns_idle(self) -> None:
        clock = _Clock()
        coordinator, _, _, events, factory = _vertical(
            settings=CaptureSettings(
                safety_ceiling_seconds=10,
                warning_before_seconds=2,
            ),
            clock=clock,
        )
        coordinator.start()
        coordinator.add_audio(b"discard me")
        clock.now = 10
        progress = coordinator._capture.poll()
        assert progress.completion is not None

        result = coordinator.process_completion(
            progress.completion,
            factory,
            mode=DeliveryMode.ASSISTANT,
            auto_submit=True,
            cancelled=lambda: True,
        )

        self.assertEqual(result.code, CaptureDeliveryCode.CANCELLED)
        self.assertEqual(result.runtime_phase, RuntimePhase.IDLE)
        self.assertEqual(coordinator._capture.phase, CapturePhase.IDLE)
        self.assertNotIn("stt.construct", events)
        self.assertNotIn("target.validate", events)
        self.assertNotIn("clipboard.snapshot", events)
        self.assertNotIn("keyboard.paste", events)

    def test_cancel_stale_and_stt_failure_never_invoke_delivery(self) -> None:
        # Explicit cancellation.
        events: list[str] = []
        injector = _SpyInjector(events)
        capture = CaptureService()
        runtime = RuntimeCoordinator(RuntimeState(RuntimePhase.RECORDING, 1))
        capture.start()
        cancelled = capture.cancel()
        coordinator = CaptureDeliveryCoordinator(capture, injector, runtime)
        result = coordinator.process_completion(
            cancelled,
            _factory(Transcription("secret", 0.9), events),
            mode=DeliveryMode.ASSISTANT,
            auto_submit=True,
        )
        self.assertEqual(result.code, CaptureDeliveryCode.CANCELLED)
        self.assertEqual(injector.deliveries, [])
        self.assertNotIn("secret", repr(result))
        self.assertEqual(capture.phase, CapturePhase.IDLE)
        self.assertEqual(coordinator.runtime.state.phase, RuntimePhase.IDLE)

        # Stale completion after a newer turn started.
        events = []
        injector = _SpyInjector(events)
        capture = CaptureService()
        capture.start()
        old = capture.toggle()
        capture.start()
        coordinator = CaptureDeliveryCoordinator(
            capture,
            injector,
            RuntimeCoordinator(RuntimeState(RuntimePhase.RECORDING, 1)),
        )
        result = coordinator.process_completion(
            old,  # type: ignore[arg-type]
            _factory(Transcription("stale-secret", 0.9), events),
            mode=DeliveryMode.ASSISTANT,
            auto_submit=True,
        )
        self.assertEqual(result.code, CaptureDeliveryCode.STALE)
        self.assertEqual(injector.deliveries, [])
        self.assertNotIn("stale-secret", repr(result))
        self.assertEqual(capture.phase, CapturePhase.RECORDING)
        self.assertEqual(coordinator.runtime.state.phase, RuntimePhase.RECORDING)

        # Model construction failure.
        events = []
        injector = _SpyInjector(events)
        capture = CaptureService()
        coordinator = CaptureDeliveryCoordinator(capture, injector)
        coordinator.start()
        coordinator.add_audio(b"audio")

        def fail(_probe):
            raise RuntimeError("model unavailable")

        result = coordinator.finish_toggle(
            fail,
            mode=DeliveryMode.ASSISTANT,
            auto_submit=True,
        )
        self.assertEqual(result.code, CaptureDeliveryCode.TRANSCRIPTION_FAILED)
        self.assertEqual(result.runtime_phase, RuntimePhase.RECOVERABLE_ERROR)
        self.assertEqual(injector.deliveries, [])

    def test_bounded_stt_timeout_and_tainted_boundary_are_propagated(self) -> None:
        events: list[str] = []
        injector = _SpyInjector(
            events,
            snapshots=[
                TargetResolution(EVIDENCE, TargetCode.VALID),
                TargetResolution(OTHER_EVIDENCE, TargetCode.VALID),
            ],
        )
        capture = CaptureService(
            snapshot_resolver=SnapshotCallableAdapter(injector.snapshot_target),
            stt_construction_timeout_seconds=0.01,
            stt_iteration_timeout_seconds=0.01,
        )
        coordinator = CaptureDeliveryCoordinator(capture, injector)
        release = threading.Event()

        def blocked_factory(_probe):
            release.wait(1)
            return _Transcriber(Transcription("late secret", 0.9), events)

        coordinator.start()
        coordinator.add_audio(b"audio")
        timed_out = coordinator.finish_toggle(
            blocked_factory,
            mode=DeliveryMode.ASSISTANT,
            auto_submit=True,
        )

        self.assertEqual(
            timed_out.code, CaptureDeliveryCode.TRANSCRIPTION_FAILED
        )
        self.assertEqual(timed_out.error_code, "stt_construction_timeout")
        self.assertEqual(
            timed_out.transcript_disposition,
            TranscriptDisposition.CONSTRUCTION_TIMEOUT,
        )
        self.assertTrue(timed_out.boundary_replacement_required)
        self.assertEqual(timed_out.runtime_phase, RuntimePhase.RECOVERABLE_ERROR)
        self.assertEqual(injector.deliveries, [])
        self.assertTrue(capture.boundary_replacement_required)

        # A later turn cannot reuse the tainted isolation boundary, and does
        # not invoke either the factory or injector.
        coordinator.start()
        coordinator.add_audio(b"new")

        def forbidden_factory(_probe):
            raise AssertionError("tainted boundary must reject before construction")

        tainted = coordinator.finish_toggle(
            forbidden_factory,
            mode=DeliveryMode.GENERIC,
            auto_submit=False,
        )
        release.set()

        self.assertEqual(tainted.error_code, "stt_boundary_tainted")
        self.assertEqual(
            tainted.transcript_disposition,
            TranscriptDisposition.BOUNDARY_TAINTED,
        )
        self.assertTrue(tainted.boundary_replacement_required)
        self.assertEqual(injector.deliveries, [])
        self.assertNotIn("late secret", repr(timed_out))
        self.assertNotIn("late secret", repr(timed_out.diagnostics))

    def test_bounded_stt_iteration_timeout_is_structured_and_never_injected(self) -> None:
        events: list[str] = []
        injector = _SpyInjector(events)
        capture = CaptureService(
            snapshot_resolver=SnapshotCallableAdapter(injector.snapshot_target),
            stt_construction_timeout_seconds=1,
            stt_iteration_timeout_seconds=0.01,
        )
        coordinator = CaptureDeliveryCoordinator(capture, injector)
        release = threading.Event()

        class BlockedTranscriber:
            def transcribe(self, _audio):
                release.wait(1)
                return Transcription("late iteration", 0.9)

        coordinator.start()
        coordinator.add_audio(b"audio")
        result = coordinator.finish_toggle(
            lambda _probe: BlockedTranscriber(),
            mode=DeliveryMode.ASSISTANT,
            auto_submit=True,
        )
        release.set()

        self.assertEqual(result.code, CaptureDeliveryCode.TRANSCRIPTION_FAILED)
        self.assertEqual(result.error_code, "stt_iteration_timeout")
        self.assertEqual(
            result.transcript_disposition,
            TranscriptDisposition.ITERATION_TIMEOUT,
        )
        self.assertTrue(result.boundary_replacement_required)
        self.assertEqual(injector.deliveries, [])

    def test_low_confidence_confirmation_resolves_a_fresh_target(self) -> None:
        events: list[str] = []
        injector = _SpyInjector(
            events,
            snapshots=[
                TargetResolution(EVIDENCE, TargetCode.VALID),
                TargetResolution(OTHER_EVIDENCE, TargetCode.VALID),
            ],
        )
        capture = CaptureService(
            snapshot_resolver=SnapshotCallableAdapter(injector.snapshot_target)
        )
        coordinator = CaptureDeliveryCoordinator(capture, injector)
        coordinator.start()
        coordinator.add_audio(b"audio")
        review = coordinator.finish_toggle(
            _factory(Transcription("edited text", 0.1), events),
            mode=DeliveryMode.ASSISTANT,
            auto_submit=True,
        )
        self.assertEqual(injector.deliveries, [])

        delivered = coordinator.confirm_or_recover(
            review.transcript,
            mode=DeliveryMode.GENERIC,
            auto_submit=True,
        )

        self.assertTrue(delivered.succeeded)
        self.assertEqual(injector.snapshot_calls, 2)
        self.assertEqual(len(injector.deliveries), 1)
        self.assertIs(injector.deliveries[0][1], OTHER_EVIDENCE)
        self.assertEqual(injector.deliveries[0][2:], (DeliveryMode.GENERIC, True))

    def test_delivery_failure_and_partial_recovery_never_reuse_old_target(self) -> None:
        for failure in (
            DeliveryCode.TARGET_CHANGED_PRE_PASTE,
            DeliveryCode.PASTED_NOT_SUBMITTED,
        ):
            with self.subTest(failure=failure):
                events: list[str] = []
                injector = _SpyInjector(
                    events,
                    snapshots=[
                        TargetResolution(EVIDENCE, TargetCode.VALID),
                        TargetResolution(OTHER_EVIDENCE, TargetCode.VALID),
                    ],
                    results=[
                        DeliveryResult(
                            failure,
                            pasted=failure is DeliveryCode.PASTED_NOT_SUBMITTED,
                        ),
                        DeliveryResult(DeliveryCode.DELIVERED, pasted=True),
                    ],
                )
                capture = CaptureService(
                    snapshot_resolver=SnapshotCallableAdapter(
                        injector.snapshot_target
                    )
                )
                coordinator = CaptureDeliveryCoordinator(capture, injector)
                coordinator.start()
                coordinator.add_audio(b"audio")
                failed = coordinator.finish_toggle(
                    _factory(Transcription("recover me", 0.9), events),
                    mode=DeliveryMode.ASSISTANT,
                    auto_submit=True,
                )
                self.assertEqual(failed.code, CaptureDeliveryCode.DELIVERY_FAILED)
                self.assertEqual(failed.runtime_phase, RuntimePhase.RECOVERABLE_ERROR)
                self.assertIs(injector.deliveries[0][1], EVIDENCE)

                recovered = coordinator.confirm_or_recover(
                    failed.transcript,
                    mode=DeliveryMode.ASSISTANT,
                    auto_submit=False,
                )
                self.assertTrue(recovered.succeeded)
                self.assertIs(injector.deliveries[1][1], OTHER_EVIDENCE)
                self.assertEqual(injector.snapshot_calls, 2)

    def test_coordinator_does_not_retain_completion_or_native_evidence(self) -> None:
        events: list[str] = []
        evidence_ref: weakref.ReferenceType[object] | None = None

        class Evidence:
            pass

        @dataclass
        class Resolution:
            evidence: object
            code: TargetCode = TargetCode.VALID

        class Injector:
            def snapshot_target(self):
                nonlocal evidence_ref
                evidence = Evidence()
                evidence_ref = weakref.ref(evidence)
                events.append("target.snapshot")
                return Resolution(evidence)

            def deliver(self, _text, _evidence, **_policy):
                events.append("injector.deliver")
                return DeliveryResult(DeliveryCode.DELIVERED, pasted=True)

        injector = Injector()
        capture = CaptureService(
            snapshot_resolver=SnapshotCallableAdapter(injector.snapshot_target)
        )
        coordinator = CaptureDeliveryCoordinator(capture, injector)
        coordinator.start()
        coordinator.add_audio(b"audio")
        result = coordinator.finish_toggle(
            _factory(Transcription("private", 0.9), events),
            mode=DeliveryMode.GENERIC,
            auto_submit=False,
        )

        gc.collect()
        assert evidence_ref is not None
        self.assertIsNone(evidence_ref())
        self.assertFalse(
            any(
                "completion" in name or "evidence" in name
                for name in vars(coordinator)
            )
        )
        self.assertNotIn("private", repr(result))
        self.assertNotIn("private", repr(result.diagnostics))

    def test_cancel_returns_only_after_admitted_side_effect_boundary_settles(self) -> None:
        entered = threading.Event()
        release = threading.Event()
        cancelled = threading.Event()
        cancel_returned = threading.Event()
        effects: list[str] = []

        class Injector:
            def snapshot_target(self) -> object:
                return EVIDENCE

            def deliver(self, *_args, **_policy) -> DeliveryResult:
                entered.set()
                release.wait(1)
                effects.append("paste")
                return DeliveryResult(DeliveryCode.DELIVERED, pasted=True)

        injector = Injector()
        coordinator = CaptureDeliveryCoordinator(
            CaptureService(
                snapshot_resolver=SnapshotCallableAdapter(
                    injector.snapshot_target
                )
            ),
            injector,
        )
        coordinator.start()
        coordinator.add_audio(b"audio")
        delivery = threading.Thread(
            target=lambda: coordinator.finish_toggle(
                _factory(Transcription("private", 0.99), []),
                mode=DeliveryMode.ASSISTANT,
                auto_submit=True,
                cancelled=cancelled.is_set,
            )
        )
        delivery.start()
        self.assertTrue(entered.wait(1))

        def cancel() -> None:
            cancelled.set()
            coordinator.cancel()
            effects.append("cancel_return")
            cancel_returned.set()

        cancellation = threading.Thread(target=cancel)
        cancellation.start()
        self.assertFalse(cancel_returned.wait(0.05))
        release.set()
        delivery.join(1)
        cancellation.join(1)

        self.assertTrue(cancel_returned.is_set())
        self.assertEqual(effects, ["paste", "cancel_return"])


class _SpyInjector:
    def __init__(
        self,
        events: list[str],
        *,
        snapshots: list[TargetResolution] | None = None,
        results: list[DeliveryResult] | None = None,
    ) -> None:
        self.events = events
        self.snapshots = deque(
            snapshots or [TargetResolution(EVIDENCE, TargetCode.VALID)]
        )
        self.results = deque(results or [DeliveryResult(DeliveryCode.DELIVERED)])
        self.deliveries: list[tuple[str, object, DeliveryMode, bool]] = []
        self.snapshot_calls = 0

    def snapshot_target(self) -> TargetResolution:
        self.snapshot_calls += 1
        self.events.append("target.snapshot")
        return self.snapshots.popleft()

    def deliver(
        self,
        text: str,
        evidence: object,
        *,
        mode: DeliveryMode,
        auto_submit: bool,
        cancelled: object = None,
    ) -> DeliveryResult:
        del cancelled
        self.events.append("injector.deliver")
        self.deliveries.append((text, evidence, mode, auto_submit))
        return self.results.popleft()


if __name__ == "__main__":
    unittest.main()
