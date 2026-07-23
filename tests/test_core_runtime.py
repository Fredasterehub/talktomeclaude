"""Generation admission tests for async companion effects."""

from __future__ import annotations

import unittest

from talktomeclaude.core import (
    EffectKind,
    EventKind,
    RuntimeCoordinator,
    RuntimeEvent,
    RuntimePhase,
    RuntimeState,
    TransitionCode,
)


class RuntimeGenerationTests(unittest.TestCase):
    _PHASE_FOR_EFFECT = {
        EffectKind.CAPTURE: RuntimePhase.TRANSCRIBING,
        EffectKind.REPLY: RuntimePhase.WAITING_FOR_CLAUDE,
        EffectKind.PLAN: RuntimePhase.PLANNING,
        EffectKind.SPEECH: RuntimePhase.SPEAKING,
    }
    _RESULT_EVENTS = {
        EffectKind.CAPTURE: {
            EventKind.FINISH_RECORDING,
            EventKind.TRANSCRIPT_ACCEPTED,
            EventKind.TRANSCRIPT_REVIEW_REQUIRED,
        },
        EffectKind.REPLY: {
            EventKind.REPLY_RECEIVED,
            EventKind.TRANSPORT_DISCONNECTED,
        },
        EffectKind.PLAN: {EventKind.PLAN_READY},
        EffectKind.SPEECH: {EventKind.SPEECH_FINISHED},
    }

    def test_each_async_effect_kind_is_rejected_after_cancel(self) -> None:
        for kind in EffectKind:
            with self.subTest(kind=kind):
                runtime = RuntimeCoordinator()
                runtime.dispatch(RuntimeEvent(EventKind.START_RECORDING))
                ticket = runtime.ticket(kind)
                runtime.dispatch(RuntimeEvent(EventKind.CANCEL))
                acceptance = runtime.accept(ticket)
                self.assertFalse(acceptance.accepted)
                self.assertEqual(TransitionCode.STALE_GENERATION, acceptance.code)

    def test_capture_result_is_rejected_after_new_turn_starts(self) -> None:
        runtime = RuntimeCoordinator()
        runtime.dispatch(RuntimeEvent(EventKind.START_RECORDING))
        old_capture = runtime.ticket(EffectKind.CAPTURE)
        runtime.dispatch(RuntimeEvent(EventKind.CANCEL))
        runtime.dispatch(RuntimeEvent(EventKind.START_RECORDING))

        result = runtime.dispatch_effect(
            old_capture, RuntimeEvent(EventKind.FINISH_RECORDING)
        )
        self.assertFalse(result.accepted)
        self.assertEqual(TransitionCode.STALE_GENERATION, result.code)
        self.assertEqual(RuntimePhase.RECORDING, runtime.state.phase)

    def test_reply_plan_and_speech_results_apply_only_in_current_generation(self) -> None:
        runtime = RuntimeCoordinator()
        runtime.dispatch(RuntimeEvent(EventKind.START_RECORDING))
        runtime.dispatch(RuntimeEvent(EventKind.FINISH_RECORDING))
        runtime.dispatch(RuntimeEvent(EventKind.TRANSCRIPT_ACCEPTED))
        runtime.dispatch(RuntimeEvent(EventKind.DELIVERY_SUCCEEDED))

        reply = runtime.ticket(EffectKind.REPLY)
        result = runtime.dispatch_effect(
            reply, RuntimeEvent(EventKind.REPLY_RECEIVED)
        )
        self.assertTrue(result.accepted)
        plan = runtime.ticket(EffectKind.PLAN)
        result = runtime.dispatch_effect(plan, RuntimeEvent(EventKind.PLAN_READY))
        self.assertTrue(result.accepted)
        speech = runtime.ticket(EffectKind.SPEECH)
        result = runtime.dispatch_effect(
            speech, RuntimeEvent(EventKind.SPEECH_FINISHED)
        )
        self.assertTrue(result.accepted)
        self.assertEqual(RuntimePhase.IDLE, runtime.state.phase)

    def test_shutdown_rejects_even_a_matching_generation_ticket(self) -> None:
        runtime = RuntimeCoordinator()
        runtime.dispatch(RuntimeEvent(EventKind.START_RECORDING))
        runtime.dispatch(RuntimeEvent(EventKind.STOP_REQUESTED))
        ticket = runtime.ticket(EffectKind.SPEECH)
        acceptance = runtime.accept(ticket)
        self.assertFalse(acceptance.accepted)
        self.assertEqual(TransitionCode.ILLEGAL_TRANSITION, acceptance.code)

    def test_completed_effect_loses_authority_even_before_next_generation(self) -> None:
        runtime = RuntimeCoordinator()
        runtime.dispatch(RuntimeEvent(EventKind.START_RECORDING))
        runtime.dispatch(RuntimeEvent(EventKind.FINISH_RECORDING))
        runtime.dispatch(RuntimeEvent(EventKind.TRANSCRIPT_ACCEPTED))
        runtime.dispatch(RuntimeEvent(EventKind.DELIVERY_SUCCEEDED))
        runtime.dispatch(RuntimeEvent(EventKind.REPLY_RECEIVED))
        runtime.dispatch(RuntimeEvent(EventKind.PLAN_READY))
        speech = runtime.ticket(EffectKind.SPEECH)
        runtime.dispatch_effect(speech, RuntimeEvent(EventKind.SPEECH_FINISHED))

        acceptance = runtime.accept(speech)
        self.assertFalse(acceptance.accepted)
        self.assertEqual(TransitionCode.ILLEGAL_TRANSITION, acceptance.code)

    def test_illegal_effect_does_not_mutate_coordinator(self) -> None:
        runtime = RuntimeCoordinator()
        ticket = runtime.ticket(EffectKind.PLAN)
        result = runtime.dispatch_effect(ticket, RuntimeEvent(EventKind.PLAN_READY))
        self.assertFalse(result.accepted)
        self.assertEqual(TransitionCode.ILLEGAL_TRANSITION, result.code)
        self.assertEqual(RuntimePhase.IDLE, runtime.state.phase)

    def test_every_cross_kind_result_event_is_rejected_before_reduction(self) -> None:
        all_result_events = set().union(*self._RESULT_EVENTS.values())
        for kind, own_events in self._RESULT_EVENTS.items():
            for event_kind in all_result_events - own_events:
                with self.subTest(ticket=kind, event=event_kind):
                    initial = RuntimeState(self._PHASE_FOR_EFFECT[kind], generation=6)
                    runtime = RuntimeCoordinator(initial)
                    ticket = runtime.ticket(kind)
                    result = runtime.dispatch_effect(
                        ticket, RuntimeEvent(event_kind)
                    )
                    self.assertFalse(result.accepted)
                    self.assertEqual(TransitionCode.INVALID_EVENT, result.code)
                    self.assertIs(initial, runtime.state)

    def test_speech_ticket_cannot_finish_capture(self) -> None:
        initial = RuntimeState(RuntimePhase.RECORDING, generation=4)
        runtime = RuntimeCoordinator(initial)
        speech = runtime.ticket(EffectKind.SPEECH)
        result = runtime.dispatch_effect(
            speech, RuntimeEvent(EventKind.FINISH_RECORDING)
        )
        self.assertFalse(result.accepted)
        self.assertEqual(TransitionCode.INVALID_EVENT, result.code)
        self.assertIs(initial, runtime.state)


if __name__ == "__main__":
    unittest.main()
