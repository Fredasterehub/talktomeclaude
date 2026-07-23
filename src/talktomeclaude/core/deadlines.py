"""Named, injectable deadlines and their recoverable outcomes."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Callable, Mapping

from .contracts import (
    EventKind,
    RuntimeEvent,
    RuntimePhase,
    RuntimeState,
    TransitionCode,
    TransitionResult,
)
from .state import reduce_state


class DeadlineName(str, Enum):
    CAPTURE_SAFETY_CEILING = "capture_safety_ceiling"
    STT_CONSTRUCTION = "stt_construction"
    STT_ITERATION = "stt_iteration"
    DELIVERY = "delivery"
    RECONNECT = "reconnect"
    PLANNING = "planning"
    SYNTHESIS = "synthesis"
    PLAYBACK_STOP = "playback_stop"
    WORKER_SHUTDOWN = "worker_shutdown"


DEADLINE_PHASES: Mapping[DeadlineName, frozenset[RuntimePhase]] = {
    DeadlineName.CAPTURE_SAFETY_CEILING: frozenset({RuntimePhase.RECORDING}),
    DeadlineName.STT_CONSTRUCTION: frozenset({RuntimePhase.TRANSCRIBING}),
    DeadlineName.STT_ITERATION: frozenset({RuntimePhase.TRANSCRIBING}),
    DeadlineName.DELIVERY: frozenset({RuntimePhase.DELIVERING}),
    DeadlineName.RECONNECT: frozenset({RuntimePhase.DISCONNECTED}),
    DeadlineName.PLANNING: frozenset({RuntimePhase.PLANNING}),
    DeadlineName.SYNTHESIS: frozenset({RuntimePhase.SPEAKING}),
    DeadlineName.PLAYBACK_STOP: frozenset(
        {RuntimePhase.SPEAKING, RuntimePhase.PAUSED, RuntimePhase.STOPPING}
    ),
    DeadlineName.WORKER_SHUTDOWN: frozenset({RuntimePhase.STOPPING}),
}


def _rejected(
    state: RuntimeState,
    event: RuntimeEvent,
    code: TransitionCode,
) -> TransitionResult:
    return TransitionResult(
        accepted=False,
        code=code,
        previous=state,
        current=state,
        event=event,
    )


@dataclass(frozen=True, slots=True)
class DeadlineSpec:
    seconds: float
    error_code: str
    recover_to: RuntimePhase

    def __post_init__(self) -> None:
        if self.seconds <= 0:
            raise ValueError("deadline seconds must be positive")
        if not self.error_code:
            raise ValueError("deadline error code must not be empty")
        if self.recover_to in {
            RuntimePhase.RECOVERABLE_ERROR,
            RuntimePhase.STOPPING,
        }:
            raise ValueError("deadline recovery destination must be actionable")


DEFAULT_DEADLINES: Mapping[DeadlineName, DeadlineSpec] = {
    DeadlineName.CAPTURE_SAFETY_CEILING: DeadlineSpec(
        900.0, "capture_safety_ceiling", RuntimePhase.AWAITING_CONFIRMATION
    ),
    DeadlineName.STT_CONSTRUCTION: DeadlineSpec(
        180.0, "stt_construction_timeout", RuntimePhase.AWAITING_CONFIRMATION
    ),
    DeadlineName.STT_ITERATION: DeadlineSpec(
        60.0, "stt_iteration_timeout", RuntimePhase.AWAITING_CONFIRMATION
    ),
    DeadlineName.DELIVERY: DeadlineSpec(
        5.0, "delivery_timeout", RuntimePhase.AWAITING_CONFIRMATION
    ),
    DeadlineName.RECONNECT: DeadlineSpec(
        30.0, "reconnect_timeout", RuntimePhase.DISCONNECTED
    ),
    DeadlineName.PLANNING: DeadlineSpec(
        3.0, "planning_timeout", RuntimePhase.SPEAKING
    ),
    DeadlineName.SYNTHESIS: DeadlineSpec(
        120.0, "synthesis_timeout", RuntimePhase.PAUSED
    ),
    DeadlineName.PLAYBACK_STOP: DeadlineSpec(
        0.25, "playback_stop_timeout", RuntimePhase.PAUSED
    ),
    DeadlineName.WORKER_SHUTDOWN: DeadlineSpec(
        2.0, "worker_shutdown_timeout", RuntimePhase.IDLE
    ),
}


@dataclass(frozen=True, slots=True)
class Deadline:
    name: DeadlineName
    started_at: float
    expires_at: float
    generation: int
    owner_phase: RuntimePhase
    ownership: int

    def expired(self, now: float) -> bool:
        return now >= self.expires_at

    def remaining(self, now: float) -> float:
        return max(0.0, self.expires_at - now)


class DeadlinePolicy:
    """Starts named deadlines using an injectable monotonic clock."""

    def __init__(
        self,
        specs: Mapping[DeadlineName, DeadlineSpec] | None = None,
        *,
        monotonic: Callable[[], float],
    ) -> None:
        self._specs = dict(DEFAULT_DEADLINES if specs is None else specs)
        missing = set(DeadlineName) - self._specs.keys()
        if missing:
            raise ValueError(f"missing deadline specs: {sorted(x.value for x in missing)}")
        self._monotonic = monotonic
        self._next_ownership = 0
        self._active: dict[DeadlineName, int] = {}

    def spec(self, name: DeadlineName) -> DeadlineSpec:
        return self._specs[name]

    def start(self, name: DeadlineName, state: RuntimeState) -> Deadline:
        if state.phase not in DEADLINE_PHASES[name]:
            raise ValueError(
                f"{name.value} is not admissible in {state.phase.value}"
            )
        now = self._monotonic()
        self._next_ownership += 1
        ownership = self._next_ownership
        self._active[name] = ownership
        return Deadline(
            name=name,
            started_at=now,
            expires_at=now + self.spec(name).seconds,
            generation=state.generation,
            owner_phase=state.phase,
            ownership=ownership,
        )

    def timeout_event(self, deadline: Deadline) -> RuntimeEvent:
        spec = self.spec(deadline.name)
        return RuntimeEvent(
            EventKind.ERROR_OCCURRED,
            generation=deadline.generation,
            error_code=spec.error_code,
            recover_to=spec.recover_to,
        )

    def apply_timeout(
        self,
        state: RuntimeState,
        deadline: Deadline,
    ):
        """Map a deadline to a generation-checked recoverable transition."""

        event = self.timeout_event(deadline)
        if deadline.generation != state.generation:
            self._release(deadline)
            return _rejected(state, event, TransitionCode.STALE_GENERATION)
        if self._active.get(deadline.name) != deadline.ownership:
            return _rejected(state, event, TransitionCode.DEADLINE_NOT_OWNER)
        now = self._monotonic()
        if not deadline.expired(now):
            return _rejected(state, event, TransitionCode.DEADLINE_NOT_EXPIRED)
        if (
            state.phase is not deadline.owner_phase
            or state.phase not in DEADLINE_PHASES[deadline.name]
        ):
            self._release(deadline)
            return _rejected(state, event, TransitionCode.DEADLINE_NOT_APPLICABLE)
        self._release(deadline)
        return reduce_state(state, event)

    def _release(self, deadline: Deadline) -> None:
        if self._active.get(deadline.name) == deadline.ownership:
            self._active.pop(deadline.name)
