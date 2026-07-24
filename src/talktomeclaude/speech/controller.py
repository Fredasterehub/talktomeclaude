"""Canonical reply to durable oral-session and speech-pipeline composition."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Protocol

from talktomeclaude.reply import ReplyEvent

from .canonical import canonicalize
from .planner import UnitKind, deterministic_plan
from .session import (
    Control,
    FrozenAnswerState,
    NavigationResult,
    OralSessionStore,
    OralStatus,
    PreviewEffectState,
)


class SpeechQueue(Protocol):
    def offer(
        self,
        unit_id: str,
        text: str,
        *,
        effect_id: str | None = None,
    ) -> bool: ...

    def stop(self) -> object: ...


@dataclass(frozen=True, slots=True)
class SpeechAcceptance:
    answer_id: str
    roadmap_created: bool
    units_scheduled: int
    status: OralStatus


@dataclass(frozen=True, slots=True)
class InterruptionResult:
    answer_id: str
    parked: FrozenAnswerState
    requires_new_turn: bool = True


class CanonicalSpeechController:
    """Attach exactly one durable oral cursor to canonical reply events."""

    def __init__(self, session: OralSessionStore, pipeline: SpeechQueue) -> None:
        self._session = session
        self._pipeline = pipeline
        self._lock = threading.RLock()
        self._active_answer_id: str | None = None
        self._offered_unit_ids: set[str] = set()

    def accept(self, event: ReplyEvent) -> SpeechAcceptance:
        """Freeze the answer before the first speech unit can be offered."""

        answer = canonicalize(event.event_id, event.answer)
        candidate = deterministic_plan(answer)
        with self._lock:
            prior_answer_id = self._session.active_answer_id()
            if prior_answer_id is not None and prior_answer_id != event.event_id:
                stopped = self._pipeline.stop()
                if getattr(stopped, "silence_confirmed", True) is not True:
                    raise RuntimeError("prior speech could not stop safely")
                self._offered_unit_ids.clear()
                self._session.park_for_interruption(prior_answer_id)
                self._active_answer_id = None
            frozen = self._session.freeze(answer, candidate)
            state = frozen.state
            if self._active_answer_id != event.event_id:
                self._active_answer_id = event.event_id
                self._offered_unit_ids.clear()
            scheduled = self._schedule_locked(state)
            return SpeechAcceptance(
                event.event_id,
                frozen.created,
                scheduled,
                state.status,
            )

    def recover_after_prior_controller_exit(self) -> int:
        """Recover claims only after the caller confirms the prior owner died."""

        with self._lock:
            return self._session.recover_preview_claims()

    def _schedule_locked(self, state: FrozenAnswerState) -> int:
        if state.status is not OralStatus.ACTIVE:
            return 0
        scheduled = 0
        for unit in state.roadmap.units[state.cursor :]:
            if (
                unit.unit_id in state.spoken_unit_ids
                or unit.unit_id in self._offered_unit_ids
            ):
                continue
            effect_id: str | None = None
            claim = None
            if unit.kind is UnitKind.PREVIEW:
                preview_state = self._session.preview_effect_state(
                    state.roadmap.answer_id
                )
                if preview_state is PreviewEffectState.CLAIMED:
                    break
                if preview_state is PreviewEffectState.PENDING:
                    claim = self._session.claim_preview(state.roadmap.answer_id)
                    if claim is None:
                        break
                    effect_id = claim.effect_id
            try:
                accepted = self._pipeline.offer(
                    unit.unit_id,
                    unit.wording,
                    effect_id=effect_id,
                )
            except BaseException:
                if claim is not None:
                    self._session.release_preview(claim)
                raise
            if not accepted:
                if claim is not None:
                    self._session.release_preview(claim)
                break
            if claim is not None:
                self._session.ack_preview(claim)
            self._offered_unit_ids.add(unit.unit_id)
            scheduled += 1
        return scheduled

    def unit_completed(self, unit_id: str) -> bool:
        """Persist an admitted completion, then refill the bounded pipeline."""

        with self._lock:
            answer_id = self._active_answer_id
            if answer_id is None or unit_id not in self._offered_unit_ids:
                return False
            state = self._session.complete_unit(answer_id, unit_id)
            self._offered_unit_ids.remove(unit_id)
            self._schedule_locked(state)
            return True

    def pause(self) -> FrozenAnswerState:
        with self._lock:
            answer_id = self._require_active_locked()
            self._offered_unit_ids.clear()
            self._pipeline.stop()
            return self._session.pause(answer_id)

    def continue_explicitly(self) -> FrozenAnswerState:
        with self._lock:
            answer_id = self._require_active_locked()
            state = self._session.continue_explicitly(answer_id)
            self._schedule_locked(state)
            return state

    def interrupt(self) -> InterruptionResult:
        """Invalidate speech first and park only the completed durable boundary."""

        with self._lock:
            answer_id = self._require_active_locked()
            self._offered_unit_ids.clear()
            self._pipeline.stop()
            parked = self._session.park_for_interruption(answer_id)
            self._active_answer_id = None
            return InterruptionResult(answer_id, parked)

    def go_back(self) -> NavigationResult:
        with self._lock:
            result = self._session.go_back()
            assert result.state is not None
            self._active_answer_id = result.state.roadmap.answer_id
            self._offered_unit_ids.clear()
            self._schedule_locked(result.state)
            return result

    def navigate(
        self, control: Control, *, target: str | None = None
    ) -> NavigationResult:
        with self._lock:
            answer_id = self._require_active_locked()
            if control is Control.GO_BACK:
                return self.go_back()
            if control in {
                Control.BACK,
                Control.NEXT,
                Control.REPEAT,
                Control.JUMP,
                Control.STOP,
            }:
                self._offered_unit_ids.clear()
                self._pipeline.stop()
            result = self._session.navigate(answer_id, control, target=target)
            if result.state is not None:
                self._schedule_locked(result.state)
            return result

    def _require_active_locked(self) -> str:
        if self._active_answer_id is None:
            raise RuntimeError("there is no active oral answer")
        return self._active_answer_id


__all__ = [
    "CanonicalSpeechController",
    "InterruptionResult",
    "SpeechAcceptance",
    "SpeechQueue",
]
