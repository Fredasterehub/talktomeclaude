"""Thread-safe state ownership and generation-safe effect admission."""

from __future__ import annotations

import threading

from .contracts import (
    EffectAcceptance,
    EffectKind,
    EffectTicket,
    EventKind,
    RuntimeEvent,
    RuntimePhase,
    RuntimeState,
    TransitionCode,
    TransitionResult,
)
from .state import reduce_state


_EFFECT_PHASES: dict[EffectKind, frozenset[RuntimePhase]] = {
    EffectKind.CAPTURE: frozenset(
        {RuntimePhase.RECORDING, RuntimePhase.TRANSCRIBING}
    ),
    EffectKind.REPLY: frozenset(
        {RuntimePhase.WAITING_FOR_CLAUDE, RuntimePhase.DISCONNECTED}
    ),
    # A validated late plan can still be considered at an unspoken boundary.
    EffectKind.PLAN: frozenset(
        {RuntimePhase.PLANNING, RuntimePhase.SPEAKING, RuntimePhase.PAUSED}
    ),
    EffectKind.SPEECH: frozenset(
        {RuntimePhase.SPEAKING, RuntimePhase.PAUSED}
    ),
}

_EFFECT_RESULTS: dict[EffectKind, frozenset[EventKind]] = {
    EffectKind.CAPTURE: frozenset(
        {
            EventKind.FINISH_RECORDING,
            EventKind.TRANSCRIPT_ACCEPTED,
            EventKind.TRANSCRIPT_REVIEW_REQUIRED,
            EventKind.ERROR_OCCURRED,
        }
    ),
    EffectKind.REPLY: frozenset(
        {
            EventKind.REPLY_RECEIVED,
            EventKind.TRANSPORT_DISCONNECTED,
            EventKind.ERROR_OCCURRED,
        }
    ),
    EffectKind.PLAN: frozenset(
        {EventKind.PLAN_READY, EventKind.ERROR_OCCURRED}
    ),
    EffectKind.SPEECH: frozenset(
        {EventKind.SPEECH_FINISHED, EventKind.ERROR_OCCURRED}
    ),
}


class RuntimeCoordinator:
    """Own state while keeping all transition decisions in the pure reducer."""

    def __init__(self, initial: RuntimeState | None = None) -> None:
        self._state = initial or RuntimeState()
        self._lock = threading.RLock()

    @property
    def state(self) -> RuntimeState:
        with self._lock:
            return self._state

    def ticket(self, kind: EffectKind) -> EffectTicket:
        with self._lock:
            return EffectTicket(kind=kind, generation=self._state.generation)

    def accept(self, ticket: EffectTicket) -> EffectAcceptance:
        with self._lock:
            generation_matches = ticket.generation == self._state.generation
            phase_accepts = self._state.phase in _EFFECT_PHASES[ticket.kind]
            accepted = generation_matches and phase_accepts
            if accepted:
                code = TransitionCode.APPLIED
            elif not generation_matches:
                code = TransitionCode.STALE_GENERATION
            else:
                code = TransitionCode.ILLEGAL_TRANSITION
            return EffectAcceptance(
                accepted=accepted,
                code=code,
                current_generation=self._state.generation,
                ticket=ticket,
            )

    def dispatch(self, event: RuntimeEvent) -> TransitionResult:
        with self._lock:
            result = reduce_state(self._state, event)
            if result.accepted:
                self._state = result.current
            return result

    def dispatch_effect(
        self,
        ticket: EffectTicket,
        event: RuntimeEvent,
    ) -> TransitionResult:
        """Atomically admit a result and apply its semantic event."""

        with self._lock:
            tagged = RuntimeEvent(
                event.kind,
                generation=ticket.generation,
                error_code=event.error_code,
                recover_to=event.recover_to,
            )
            if ticket.generation != self._state.generation:
                return TransitionResult(
                    accepted=False,
                    code=TransitionCode.STALE_GENERATION,
                    previous=self._state,
                    current=self._state,
                    event=tagged,
                )
            if event.kind not in _EFFECT_RESULTS[ticket.kind]:
                return TransitionResult(
                    accepted=False,
                    code=TransitionCode.INVALID_EVENT,
                    previous=self._state,
                    current=self._state,
                    event=tagged,
                )
            if self._state.phase not in _EFFECT_PHASES[ticket.kind]:
                return TransitionResult(
                    accepted=False,
                    code=TransitionCode.ILLEGAL_TRANSITION,
                    previous=self._state,
                    current=self._state,
                    event=tagged,
                )
            result = reduce_state(self._state, tagged)
            if result.accepted:
                self._state = result.current
            return result
