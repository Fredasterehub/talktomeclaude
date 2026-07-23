"""Named deadline and recoverable timeout mapping tests."""

from __future__ import annotations

import unittest

from talktomeclaude.core import (
    DEFAULT_DEADLINES,
    DeadlineName,
    DeadlinePolicy,
    DeadlineSpec,
    EventKind,
    RuntimeEvent,
    RuntimePhase,
    RuntimeState,
    TransitionCode,
    reduce_state,
)


class FakeClock:
    def __init__(self, now: float = 100.0) -> None:
        self.now = now

    def monotonic(self) -> float:
        return self.now


class DeadlineTests(unittest.TestCase):
    _OWNER_PHASE = {
        DeadlineName.CAPTURE_SAFETY_CEILING: RuntimePhase.RECORDING,
        DeadlineName.STT_CONSTRUCTION: RuntimePhase.TRANSCRIBING,
        DeadlineName.STT_ITERATION: RuntimePhase.TRANSCRIBING,
        DeadlineName.DELIVERY: RuntimePhase.DELIVERING,
        DeadlineName.RECONNECT: RuntimePhase.DISCONNECTED,
        DeadlineName.PLANNING: RuntimePhase.PLANNING,
        DeadlineName.SYNTHESIS: RuntimePhase.SPEAKING,
        DeadlineName.PLAYBACK_STOP: RuntimePhase.PAUSED,
        DeadlineName.WORKER_SHUTDOWN: RuntimePhase.STOPPING,
    }

    def _owner_state(self, name: DeadlineName, generation: int) -> RuntimeState:
        phase = self._OWNER_PHASE[name]
        if phase is RuntimePhase.DISCONNECTED:
            return RuntimeState(
                phase,
                generation,
                resume_phase=RuntimePhase.WAITING_FOR_CLAUDE,
            )
        return RuntimeState(phase, generation)

    def test_every_named_deadline_has_a_distinct_recoverable_error_mapping(self) -> None:
        clock = FakeClock()
        policy = DeadlinePolicy(monotonic=clock.monotonic)
        error_codes: set[str] = set()

        for index, name in enumerate(DeadlineName):
            with self.subTest(name=name):
                state = self._owner_state(name, index)
                deadline = policy.start(name, state)
                spec = policy.spec(name)
                self.assertEqual(clock.now, deadline.started_at)
                self.assertEqual(clock.now + spec.seconds, deadline.expires_at)
                self.assertFalse(deadline.expired(clock.now))
                clock.now = deadline.expires_at
                self.assertTrue(deadline.expired(clock.now))
                self.assertEqual(0.0, deadline.remaining(clock.now))
                result = policy.apply_timeout(state, deadline)
                self.assertTrue(result.accepted)
                self.assertEqual(TransitionCode.APPLIED, result.code)
                self.assertEqual(RuntimePhase.RECOVERABLE_ERROR, result.current.phase)
                self.assertEqual(spec.error_code, result.current.error_code)
                self.assertEqual(spec.recover_to, result.current.resume_phase)
                error_codes.add(spec.error_code)
                clock.now += 1

        self.assertEqual(len(DeadlineName), len(error_codes))

    def test_stale_deadline_cannot_fault_a_new_generation(self) -> None:
        clock = FakeClock()
        policy = DeadlinePolicy(monotonic=clock.monotonic)
        deadline = policy.start(
            DeadlineName.SYNTHESIS,
            RuntimeState(RuntimePhase.SPEAKING, generation=3),
        )
        state = RuntimeState(RuntimePhase.SPEAKING, generation=4)
        result = policy.apply_timeout(state, deadline)
        self.assertFalse(result.accepted)
        self.assertEqual(TransitionCode.STALE_GENERATION, result.code)
        self.assertIs(state, result.current)

    def test_unexpired_deadline_is_explicitly_not_applied(self) -> None:
        clock = FakeClock()
        policy = DeadlinePolicy(monotonic=clock.monotonic)
        state = RuntimeState(RuntimePhase.DELIVERING, generation=2)
        deadline = policy.start(DeadlineName.DELIVERY, state)

        result = policy.apply_timeout(state, deadline)

        self.assertFalse(result.accepted)
        self.assertEqual(TransitionCode.DEADLINE_NOT_EXPIRED, result.code)
        self.assertIs(state, result.current)

    def test_late_same_generation_deadline_cannot_fault_a_later_phase(self) -> None:
        clock = FakeClock()
        policy = DeadlinePolicy(monotonic=clock.monotonic)
        owner = RuntimeState(RuntimePhase.DELIVERING, generation=9)
        deadline = policy.start(DeadlineName.DELIVERY, owner)
        clock.now = deadline.expires_at
        later = RuntimeState(RuntimePhase.WAITING_FOR_CLAUDE, generation=9)

        result = policy.apply_timeout(later, deadline)

        self.assertFalse(result.accepted)
        self.assertEqual(TransitionCode.DEADLINE_NOT_APPLICABLE, result.code)
        self.assertIs(later, result.current)
        replay = policy.apply_timeout(owner, deadline)
        self.assertFalse(replay.accepted)
        self.assertEqual(TransitionCode.DEADLINE_NOT_OWNER, replay.code)

    def test_new_deadline_replaces_old_ownership(self) -> None:
        clock = FakeClock()
        policy = DeadlinePolicy(monotonic=clock.monotonic)
        state = RuntimeState(RuntimePhase.PLANNING, generation=5)
        old = policy.start(DeadlineName.PLANNING, state)
        new = policy.start(DeadlineName.PLANNING, state)
        clock.now = new.expires_at

        rejected = policy.apply_timeout(state, old)
        self.assertFalse(rejected.accepted)
        self.assertEqual(TransitionCode.DEADLINE_NOT_OWNER, rejected.code)
        applied = policy.apply_timeout(state, new)
        self.assertTrue(applied.accepted)

    def test_deadline_cannot_start_outside_its_admissible_phase(self) -> None:
        clock = FakeClock()
        policy = DeadlinePolicy(monotonic=clock.monotonic)
        with self.assertRaises(ValueError):
            policy.start(
                DeadlineName.DELIVERY,
                RuntimeState(RuntimePhase.WAITING_FOR_CLAUDE),
            )

    def test_reconnect_timeout_requires_explicit_reconnected_event(self) -> None:
        clock = FakeClock()
        policy = DeadlinePolicy(monotonic=clock.monotonic)
        disconnected = RuntimeState(
            RuntimePhase.DISCONNECTED,
            generation=12,
            resume_phase=RuntimePhase.WAITING_FOR_CLAUDE,
        )
        deadline = policy.start(DeadlineName.RECONNECT, disconnected)
        clock.now = deadline.expires_at

        timed_out = policy.apply_timeout(disconnected, deadline)
        self.assertEqual(RuntimePhase.RECOVERABLE_ERROR, timed_out.current.phase)
        self.assertEqual(RuntimePhase.DISCONNECTED, timed_out.current.resume_phase)
        retried = reduce_state(timed_out.current, RuntimeEvent(EventKind.RETRY))
        self.assertEqual(RuntimePhase.DISCONNECTED, retried.current.phase)
        self.assertEqual(
            RuntimePhase.WAITING_FOR_CLAUDE, retried.current.resume_phase
        )
        reconnected = reduce_state(
            retried.current,
            RuntimeEvent(EventKind.TRANSPORT_RECONNECTED),
        )
        self.assertEqual(RuntimePhase.WAITING_FOR_CLAUDE, reconnected.current.phase)

    def test_policy_requires_complete_positive_specs(self) -> None:
        clock = FakeClock()
        with self.assertRaises(ValueError):
            DeadlinePolicy(specs={}, monotonic=clock.monotonic)
        with self.assertRaises(ValueError):
            DeadlineSpec(0, "bad", RuntimePhase.IDLE)
        with self.assertRaises(ValueError):
            DeadlineSpec(1, "", RuntimePhase.IDLE)
        with self.assertRaises(ValueError):
            DeadlineSpec(1, "bad", RuntimePhase.STOPPING)

    def test_default_specs_cover_enum_exactly(self) -> None:
        self.assertEqual(set(DeadlineName), set(DEFAULT_DEADLINES))


if __name__ == "__main__":
    unittest.main()
