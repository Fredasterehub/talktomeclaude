"""Durable frozen-roadmap, preview admission, and oral navigation state."""

from __future__ import annotations

import copy
import hashlib
import secrets
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from enum import StrEnum
from pathlib import Path
from typing import Any

from talktomeclaude.storage import AtomicJsonTransaction

from .canonical import CanonicalAnswer
from .planner import OralRoadmap, OralUnit, UnitKind, refine_unsaid, validate_roadmap

SESSION_VERSION = 2
MAX_RECAP_CHARS = 240


class OralSessionError(RuntimeError):
    """Durable oral state is corrupt or a transition is not permitted."""


class OralStatus(StrEnum):
    ACTIVE = "active"
    PAUSED = "paused"
    PARKED = "parked"
    STOPPED = "stopped"
    COMPLETE = "complete"


class Control(StrEnum):
    PAUSE = "pause"
    CONTINUE = "continue"
    REPEAT = "repeat"
    BACK = "back"
    NEXT = "next"
    TOPICS = "topics"
    SUMMARIZE = "summarize"
    DEEPER = "deeper"
    JUMP = "jump"
    WHERE = "where"
    GO_BACK = "go_back"
    KEEP_GOING = "keep_going"
    STOP = "stop"
    VOICE_OFF = "voice_off"
    HELP = "help"


class PreviewEffectState(StrEnum):
    PENDING = "pending"
    CLAIMED = "claimed"
    DELIVERED = "delivered"


_CONTROL_SYNONYMS: Mapping[Control, frozenset[str]] = {
    Control.PAUSE: frozenset({"pause", "hold", "hold on"}),
    Control.CONTINUE: frozenset({"continue", "resume"}),
    Control.REPEAT: frozenset({"repeat", "say that again"}),
    Control.BACK: frozenset({"back", "previous"}),
    Control.NEXT: frozenset({"next", "skip"}),
    Control.TOPICS: frozenset({"topics", "list topics"}),
    Control.SUMMARIZE: frozenset({"summarize", "summary"}),
    Control.DEEPER: frozenset({"deeper", "more detail"}),
    Control.JUMP: frozenset({"jump", "jump to"}),
    Control.WHERE: frozenset({"where were you", "where-were-you", "where"}),
    Control.GO_BACK: frozenset({"go back", "go-back", "return to that"}),
    Control.KEEP_GOING: frozenset({"keep going", "keep-going", "carry on"}),
    Control.STOP: frozenset({"stop talking", "stop-talking", "stop"}),
    Control.VOICE_OFF: frozenset({"voice off", "voice-off", "mute voice"}),
    Control.HELP: frozenset({"help", "voice help"}),
}


@dataclass(frozen=True, slots=True)
class ControlCommand:
    """A deterministic control plus optional content-bearing topic target."""

    control: Control
    target: str | None = field(default=None, repr=False)


def parse_control_command(text: str) -> ControlCommand | None:
    normalized = " ".join(text.casefold().strip().split())
    for control, synonyms in _CONTROL_SYNONYMS.items():
        if normalized in synonyms:
            return ControlCommand(control)
    jump_prefix = "jump to "
    if normalized.startswith(jump_prefix):
        target = normalized[len(jump_prefix) :].strip()
        if target:
            return ControlCommand(Control.JUMP, target)
    return None


def parse_control(text: str) -> Control | None:
    command = parse_control_command(text)
    return command.control if command is not None else None


@dataclass(frozen=True, slots=True)
class FrozenAnswerState:
    roadmap: OralRoadmap = field(repr=False)
    cursor: int
    spoken_unit_ids: frozenset[str]
    deferred_block_ids: frozenset[str]
    status: OralStatus

    @property
    def current_unit(self) -> OralUnit | None:
        if 0 <= self.cursor < len(self.roadmap.units):
            return self.roadmap.units[self.cursor]
        return None


@dataclass(frozen=True, slots=True)
class FreezeResult:
    state: FrozenAnswerState = field(repr=False)
    created: bool


@dataclass(frozen=True, slots=True)
class PreviewClaim:
    answer_id: str
    effect_id: str
    unit_id: str
    claim_token: str = field(repr=False)
    unit: OralUnit = field(repr=False)


@dataclass(frozen=True, slots=True)
class NavigationResult:
    control: Control
    state: FrozenAnswerState | None = field(repr=False)
    unit: OralUnit | None = field(default=None, repr=False)
    response: str = field(default="", repr=False)
    requires_new_turn: bool = False


@dataclass(frozen=True, slots=True)
class _PreviewRecord:
    effect_id: str
    unit_id: str
    state: PreviewEffectState
    claim_token: str | None = field(default=None, repr=False)

    def to_dict(self) -> dict[str, object]:
        return {
            "claim_token": self.claim_token,
            "effect_id": self.effect_id,
            "state": self.state.value,
            "unit_id": self.unit_id,
        }


def _empty_root() -> dict[str, Any]:
    return {
        "active_answer_id": None,
        "answers": {},
        "parked_answer_ids": [],
        "version": SESSION_VERSION,
    }


def _root(value: dict[str, Any]) -> dict[str, Any]:
    if not value:
        return _empty_root()
    if set(value) != {
        "active_answer_id",
        "answers",
        "parked_answer_ids",
        "version",
    }:
        raise OralSessionError("durable oral session schema is invalid")
    active = value["active_answer_id"]
    answers = value["answers"]
    parked = value["parked_answer_ids"]
    if (
        value["version"] != SESSION_VERSION
        or (active is not None and (not isinstance(active, str) or not active))
        or not isinstance(answers, dict)
        or any(not isinstance(key, str) or not key for key in answers)
        or not isinstance(parked, list)
        or any(not isinstance(item, str) or not item for item in parked)
        or len(set(parked)) != len(parked)
        or (active is not None and active not in answers)
        or any(item not in answers for item in parked)
        or active in parked
    ):
        raise OralSessionError("durable oral session state is invalid")
    return value


def _preview_effect_id(roadmap: OralRoadmap, unit_id: str) -> str:
    digest = hashlib.sha256(
        f"{roadmap.answer_digest}\0{unit_id}\0preview".encode("utf-8")
    ).hexdigest()
    return f"preview-{digest}"


def _preview_from_record(
    value: object,
    roadmap: OralRoadmap,
) -> _PreviewRecord | None:
    preview_units = tuple(unit for unit in roadmap.units if unit.kind is UnitKind.PREVIEW)
    if not preview_units:
        if value is not None:
            raise OralSessionError("simple answer carries a preview effect")
        return None
    if len(preview_units) != 1 or not isinstance(value, dict) or set(value) != {
        "claim_token",
        "effect_id",
        "state",
        "unit_id",
    }:
        raise OralSessionError("preview effect record is invalid")
    effect_id = value["effect_id"]
    unit_id = value["unit_id"]
    raw_state = value["state"]
    claim_token = value["claim_token"]
    if (
        not isinstance(effect_id, str)
        or not isinstance(unit_id, str)
        or not isinstance(raw_state, str)
        or (claim_token is not None and not isinstance(claim_token, str))
    ):
        raise OralSessionError("preview effect values are invalid")
    try:
        state = PreviewEffectState(raw_state)
    except ValueError as exc:
        raise OralSessionError("preview effect state is invalid") from exc
    preview = preview_units[0]
    if (
        unit_id != preview.unit_id
        or effect_id != _preview_effect_id(roadmap, unit_id)
        or (state is PreviewEffectState.CLAIMED) != bool(claim_token)
    ):
        raise OralSessionError("preview effect identity is invalid")
    return _PreviewRecord(effect_id, unit_id, state, claim_token)


def _state_from_record(record: object) -> FrozenAnswerState:
    if not isinstance(record, dict) or set(record) != {
        "cursor",
        "deferred_block_ids",
        "oral_roadmap_frozen",
        "preview_effect",
        "spoken_unit_ids",
        "status",
    }:
        raise OralSessionError("frozen answer record is invalid")
    try:
        roadmap = OralRoadmap.from_dict(record["oral_roadmap_frozen"])
        cursor = record["cursor"]
        spoken = record["spoken_unit_ids"]
        deferred = record["deferred_block_ids"]
        raw_status = record["status"]
        if (
            type(cursor) is not int
            or cursor < 0
            or cursor > len(roadmap.units)
            or not isinstance(spoken, list)
            or any(not isinstance(item, str) or not item for item in spoken)
            or not isinstance(deferred, list)
            or any(not isinstance(item, str) or not item for item in deferred)
            or not isinstance(raw_status, str)
        ):
            raise ValueError
        status = OralStatus(raw_status)
    except (KeyError, TypeError, ValueError) as exc:
        raise OralSessionError("frozen answer record is invalid") from exc
    known_units = frozenset(unit.unit_id for unit in roadmap.units)
    known_blocks = frozenset(item.block_id for item in roadmap.block_dispositions)
    spoken_set = frozenset(spoken)
    deferred_set = frozenset(deferred)
    if (
        len(spoken_set) != len(spoken)
        or len(deferred_set) != len(deferred)
        or not spoken_set <= known_units
        or not deferred_set <= known_blocks
        or (status is OralStatus.COMPLETE) != (cursor == len(roadmap.units))
    ):
        raise OralSessionError("frozen answer cursor/disposition state is invalid")
    _preview_from_record(record["preview_effect"], roadmap)
    return FrozenAnswerState(roadmap, cursor, spoken_set, deferred_set, status)


def _preview_record(record: dict[str, Any], roadmap: OralRoadmap) -> _PreviewRecord | None:
    return _preview_from_record(record["preview_effect"], roadmap)


def _record(
    state: FrozenAnswerState,
    preview: _PreviewRecord | None,
) -> dict[str, Any]:
    return {
        "cursor": state.cursor,
        "deferred_block_ids": sorted(state.deferred_block_ids),
        "oral_roadmap_frozen": state.roadmap.to_dict(),
        "preview_effect": None if preview is None else preview.to_dict(),
        "spoken_unit_ids": sorted(state.spoken_unit_ids),
        "status": state.status.value,
    }


def _new_preview(roadmap: OralRoadmap) -> _PreviewRecord | None:
    units = tuple(unit for unit in roadmap.units if unit.kind is UnitKind.PREVIEW)
    if not units:
        return None
    unit = units[0]
    return _PreviewRecord(
        _preview_effect_id(roadmap, unit.unit_id),
        unit.unit_id,
        PreviewEffectState.PENDING,
    )


class OralSessionStore:
    """Cross-process durable roadmap CAS, preview outbox, and navigation owner."""

    def __init__(
        self,
        path: str | Path,
        *,
        phase_hook: Callable[[str], None] | None = None,
    ) -> None:
        self.path = Path(path)
        self._transaction = AtomicJsonTransaction(
            self.path,
            purpose="oral-roadmap-frozen",
            phase_hook=phase_hook,
        )

    def active_answer_id(self) -> str | None:
        """Return the durable active answer without exposing its content."""

        return _root(self._transaction.read())["active_answer_id"]

    def freeze(self, answer: CanonicalAnswer, candidate: OralRoadmap) -> FreezeResult:
        """Compare-and-set one immutable roadmap; an existing winner is restored."""

        validate_roadmap(answer, candidate)
        result: list[FreezeResult] = []

        def update(current: dict[str, Any]) -> dict[str, Any]:
            state = copy.deepcopy(_root(current))
            answers = state["answers"]
            existing = answers.get(answer.answer_id)
            if existing is not None:
                restored = _state_from_record(existing)
                validate_roadmap(answer, restored.roadmap)
                result.append(FreezeResult(restored, False))
                return state
            active = state["active_answer_id"]
            if active is not None:
                active_state = _state_from_record(answers[active])
                if active_state.status not in (OralStatus.COMPLETE, OralStatus.STOPPED):
                    raise OralSessionError("active answer must be parked before a new freeze")
            frozen = FrozenAnswerState(
                candidate,
                0,
                frozenset(),
                frozenset(),
                OralStatus.ACTIVE,
            )
            answers[answer.answer_id] = _record(frozen, _new_preview(candidate))
            state["active_answer_id"] = answer.answer_id
            result.append(FreezeResult(frozen, True))
            return state

        try:
            self._transaction.update(update)
        except OralSessionError:
            raise
        except Exception as exc:
            raise OralSessionError("oral roadmap freeze failed") from exc
        return result[0]

    def claim_preview(self, answer_id: str) -> PreviewClaim | None:
        """Atomically claim one pending preview admission for one controller."""

        result: list[PreviewClaim | None] = []
        claim_token = secrets.token_hex(16)

        def update(current: dict[str, Any]) -> dict[str, Any]:
            root = copy.deepcopy(_root(current))
            record = root["answers"].get(answer_id)
            if not isinstance(record, dict):
                raise OralSessionError("answer has no frozen roadmap")
            state = _state_from_record(record)
            preview = _preview_record(record, state.roadmap)
            if preview is None or preview.state is not PreviewEffectState.PENDING:
                result.append(None)
                return root
            claimed = replace(
                preview,
                state=PreviewEffectState.CLAIMED,
                claim_token=claim_token,
            )
            root["answers"][answer_id] = _record(state, claimed)
            result.append(
                PreviewClaim(
                    answer_id,
                    claimed.effect_id,
                    claimed.unit_id,
                    claim_token,
                    state.roadmap.unit(claimed.unit_id),
                )
            )
            return root

        self._transaction.update(update)
        return result[0]

    def preview_effect_state(
        self, answer_id: str
    ) -> PreviewEffectState | None:
        root = _root(self._transaction.read())
        record = root["answers"].get(answer_id)
        if not isinstance(record, dict):
            raise OralSessionError("answer has no frozen roadmap")
        state = _state_from_record(record)
        preview = _preview_record(record, state.roadmap)
        return None if preview is None else preview.state

    def ack_preview(self, claim: PreviewClaim) -> None:
        """Mark a preview delivered only after durable idempotent queue admission."""

        def update(current: dict[str, Any]) -> dict[str, Any]:
            root = copy.deepcopy(_root(current))
            record = root["answers"].get(claim.answer_id)
            if not isinstance(record, dict):
                raise OralSessionError("preview answer is missing")
            state = _state_from_record(record)
            preview = _preview_record(record, state.roadmap)
            if (
                preview is None
                or preview.effect_id != claim.effect_id
                or preview.unit_id != claim.unit_id
                or preview.state is not PreviewEffectState.CLAIMED
                or preview.claim_token != claim.claim_token
            ):
                raise OralSessionError("preview claim is no longer authoritative")
            delivered = replace(
                preview,
                state=PreviewEffectState.DELIVERED,
                claim_token=None,
            )
            root["answers"][claim.answer_id] = _record(state, delivered)
            return root

        self._transaction.update(update)

    def release_preview(self, claim: PreviewClaim) -> None:
        """Return a failed admission claim to the durable pending state."""

        def update(current: dict[str, Any]) -> dict[str, Any]:
            root = copy.deepcopy(_root(current))
            record = root["answers"].get(claim.answer_id)
            if not isinstance(record, dict):
                raise OralSessionError("preview answer is missing")
            state = _state_from_record(record)
            preview = _preview_record(record, state.roadmap)
            if (
                preview is None
                or preview.state is not PreviewEffectState.CLAIMED
                or preview.claim_token != claim.claim_token
            ):
                raise OralSessionError("preview claim is no longer authoritative")
            pending = replace(
                preview,
                state=PreviewEffectState.PENDING,
                claim_token=None,
            )
            root["answers"][claim.answer_id] = _record(state, pending)
            return root

        self._transaction.update(update)

    def recover_preview_claims(self) -> int:
        """Reset claims left by a confirmed-dead prior controller process."""

        recovered = [0]

        def update(current: dict[str, Any]) -> dict[str, Any]:
            root = copy.deepcopy(_root(current))
            for answer_id, record in tuple(root["answers"].items()):
                state = _state_from_record(record)
                preview = _preview_record(record, state.roadmap)
                if preview is not None and preview.state is PreviewEffectState.CLAIMED:
                    pending = replace(
                        preview,
                        state=PreviewEffectState.PENDING,
                        claim_token=None,
                    )
                    root["answers"][answer_id] = _record(state, pending)
                    recovered[0] += 1
            return root

        self._transaction.update(update)
        return recovered[0]

    def freeze_before_preview(
        self,
        answer: CanonicalAnswer,
        candidate: OralRoadmap,
        enqueue: Callable[[OralUnit], None],
    ) -> FreezeResult:
        """Compatibility helper: freeze, claim, enqueue, then acknowledge."""

        result = self.freeze(answer, candidate)
        claim = self.claim_preview(answer.answer_id)
        if claim is None:
            return result
        try:
            enqueue(claim.unit)
        except BaseException:
            self.release_preview(claim)
            raise
        self.ack_preview(claim)
        return result

    def restore(self, answer_id: str) -> FrozenAnswerState | None:
        root = _root(self._transaction.read())
        record = root["answers"].get(answer_id)
        return None if record is None else _state_from_record(record)

    def _transition(
        self,
        answer_id: str,
        transform: Callable[[FrozenAnswerState], FrozenAnswerState],
    ) -> FrozenAnswerState:
        result: list[FrozenAnswerState] = []

        def update(current: dict[str, Any]) -> dict[str, Any]:
            root = copy.deepcopy(_root(current))
            existing = root["answers"].get(answer_id)
            if not isinstance(existing, dict):
                raise OralSessionError("answer has no frozen roadmap")
            state = _state_from_record(existing)
            replacement = transform(state)
            preview = _preview_record(existing, state.roadmap)
            root["answers"][answer_id] = _record(replacement, preview)
            if replacement.status in (OralStatus.COMPLETE, OralStatus.STOPPED):
                if root["active_answer_id"] == answer_id:
                    root["active_answer_id"] = None
            result.append(replacement)
            return root

        self._transaction.update(update)
        return result[0]

    def complete_unit(self, answer_id: str, unit_id: str) -> FrozenAnswerState:
        def complete(state: FrozenAnswerState) -> FrozenAnswerState:
            if state.status is not OralStatus.ACTIVE:
                raise OralSessionError("only the active answer can complete speech")
            current = state.current_unit
            if current is None or current.unit_id != unit_id:
                raise OralSessionError("speech completion is not the current frozen unit")
            cursor = state.cursor + 1
            status = OralStatus.COMPLETE if cursor == len(state.roadmap.units) else state.status
            return replace(
                state,
                cursor=cursor,
                spoken_unit_ids=state.spoken_unit_ids | {unit_id},
                status=status,
            )

        return self._transition(answer_id, complete)

    def pause(self, answer_id: str) -> FrozenAnswerState:
        def pause_active(state: FrozenAnswerState) -> FrozenAnswerState:
            if state.status is not OralStatus.ACTIVE:
                raise OralSessionError("only the active answer can pause")
            return replace(state, status=OralStatus.PAUSED)

        return self._transition(answer_id, pause_active)

    def continue_explicitly(self, answer_id: str) -> FrozenAnswerState:
        def resume(state: FrozenAnswerState) -> FrozenAnswerState:
            if state.status is OralStatus.PARKED:
                raise OralSessionError("parked answers require explicit go back")
            if state.status in (OralStatus.STOPPED, OralStatus.COMPLETE):
                raise OralSessionError("answer cannot resume from its terminal state")
            return replace(state, status=OralStatus.ACTIVE)

        return self._transition(answer_id, resume)

    def park_for_interruption(self, answer_id: str) -> FrozenAnswerState:
        parked: list[FrozenAnswerState] = []

        def update(current: dict[str, Any]) -> dict[str, Any]:
            root = copy.deepcopy(_root(current))
            existing = root["answers"].get(answer_id)
            if not isinstance(existing, dict):
                raise OralSessionError("answer has no frozen roadmap")
            state = _state_from_record(existing)
            if state.status not in (OralStatus.ACTIVE, OralStatus.PAUSED):
                raise OralSessionError("answer cannot park from its current state")
            replacement = replace(state, status=OralStatus.PARKED)
            root["answers"][answer_id] = _record(
                replacement,
                _preview_record(existing, state.roadmap),
            )
            history = [item for item in root["parked_answer_ids"] if item != answer_id]
            history.append(answer_id)
            root["parked_answer_ids"] = history
            if root["active_answer_id"] == answer_id:
                root["active_answer_id"] = None
            parked.append(replacement)
            return root

        self._transaction.update(update)
        return parked[0]

    @staticmethod
    def _recap(state: FrozenAnswerState) -> str:
        spoken_topics = []
        for topic in state.roadmap.topics:
            if any(
                unit.unit_id in state.spoken_unit_ids and unit.topic_id == topic.topic_id
                for unit in state.roadmap.units
            ):
                spoken_topics.append(topic.label)
        current = state.current_unit
        text = (
            "We covered " + ", ".join(spoken_topics) + "."
            if spoken_topics
            else "We had not completed a section yet."
        )
        if current is not None and current.topic_id is not None:
            text += " Returning to " + state.roadmap.topic(current.topic_id).label + "."
        if len(text) > MAX_RECAP_CHARS:
            text = text[: MAX_RECAP_CHARS - 1].rstrip() + "…"
        return text

    def go_back(self, *, skip_answer_id: str | None = None) -> NavigationResult:
        result: list[NavigationResult] = []

        def update(current: dict[str, Any]) -> dict[str, Any]:
            root = copy.deepcopy(_root(current))
            history = root["parked_answer_ids"]
            if not history:
                raise OralSessionError("there is no parked answer")
            target_id = history.pop()
            skipped_id = None
            if target_id == skip_answer_id and history:
                skipped_id = target_id
                target_id = history.pop()
            target_record = root["answers"].get(target_id)
            if not isinstance(target_record, dict):
                raise OralSessionError("parked answer state is missing")
            active_id = root["active_answer_id"]
            if active_id is not None and active_id != target_id:
                active_record = root["answers"].get(active_id)
                if not isinstance(active_record, dict):
                    raise OralSessionError("active answer state is missing")
                active_state = _state_from_record(active_record)
                if active_state.status not in (OralStatus.ACTIVE, OralStatus.PAUSED):
                    raise OralSessionError("active answer cannot be parked for return")
                parked_active = replace(active_state, status=OralStatus.PARKED)
                root["answers"][active_id] = _record(
                    parked_active,
                    _preview_record(active_record, active_state.roadmap),
                )
                history[:] = [item for item in history if item != active_id]
                history.append(active_id)
            if skipped_id is not None:
                history[:] = [item for item in history if item != skipped_id]
                history.append(skipped_id)
            target_state = _state_from_record(target_record)
            restored = replace(target_state, status=OralStatus.ACTIVE)
            root["answers"][target_id] = _record(
                restored,
                _preview_record(target_record, target_state.roadmap),
            )
            root["active_answer_id"] = target_id
            result.append(
                NavigationResult(
                    Control.GO_BACK,
                    restored,
                    restored.current_unit,
                    self._recap(restored),
                )
            )
            return root

        self._transaction.update(update)
        return result[0]

    def apply_late_plan(
        self,
        answer: CanonicalAnswer,
        candidate: OralRoadmap,
    ) -> FrozenAnswerState:
        def refine(state: FrozenAnswerState) -> FrozenAnswerState:
            roadmap = refine_unsaid(
                answer,
                state.roadmap,
                candidate,
                spoken_unit_ids=state.spoken_unit_ids,
            )
            return replace(state, roadmap=roadmap)

        return self._transition(answer.answer_id, refine)

    @staticmethod
    def _deferred_for_units(units: tuple[OralUnit, ...]) -> frozenset[str]:
        return frozenset(
            block_id
            for unit in units
            if unit.kind is UnitKind.SECTION
            for block_id in unit.block_ids
        )

    def navigate(
        self,
        answer_id: str,
        control: Control,
        *,
        target: str | None = None,
    ) -> NavigationResult:
        if control is Control.GO_BACK:
            return self.go_back()
        if control is Control.PAUSE:
            state = self.pause(answer_id)
            return NavigationResult(control, state, state.current_unit)
        if control in (Control.CONTINUE, Control.KEEP_GOING):
            state = self.continue_explicitly(answer_id)
            return NavigationResult(control, state, state.current_unit)
        if control is Control.STOP:
            state = self._transition(
                answer_id,
                lambda value: replace(value, status=OralStatus.STOPPED),
            )
            return NavigationResult(control, state)
        if control is Control.VOICE_OFF:
            restored = self.restore(answer_id)
            return NavigationResult(control, restored)
        if control is Control.HELP:
            restored = self.restore(answer_id)
            return NavigationResult(
                control,
                restored,
                response="Pause, continue, repeat, back, next, topics, jump, summarize, deeper, or stop.",
            )

        def move(state: FrozenAnswerState) -> FrozenAnswerState:
            if state.status not in (OralStatus.ACTIVE, OralStatus.PAUSED):
                raise OralSessionError("answer cannot navigate from its current state")
            cursor = state.cursor
            deferred = state.deferred_block_ids
            if control in (Control.REPEAT, Control.BACK):
                cursor = max(0, cursor - 1)
            elif control is Control.NEXT:
                current = state.current_unit
                if current is None:
                    raise OralSessionError("answer has no next frozen unit")
                deferred |= self._deferred_for_units((current,))
                cursor += 1
            elif control is Control.JUMP:
                if not target:
                    raise OralSessionError("jump requires a frozen topic target")
                normalized = target.casefold()
                topic = next(
                    (
                        item
                        for item in state.roadmap.topics
                        if item.topic_id == target or item.label.casefold() == normalized
                    ),
                    None,
                )
                if topic is None:
                    raise OralSessionError("jump target is not a frozen topic")
                destination = next(
                    index
                    for index, unit in enumerate(state.roadmap.units)
                    if unit.kind is UnitKind.SECTION and unit.topic_id == topic.topic_id
                )
                if destination > cursor:
                    deferred |= self._deferred_for_units(
                        state.roadmap.units[cursor:destination]
                    )
                cursor = destination
            status = OralStatus.COMPLETE if cursor == len(state.roadmap.units) else state.status
            return replace(
                state,
                cursor=cursor,
                deferred_block_ids=deferred,
                status=status,
            )

        if control in (Control.REPEAT, Control.BACK, Control.NEXT, Control.JUMP):
            state = self._transition(answer_id, move)
            return NavigationResult(control, state, state.current_unit)

        restored = self.restore(answer_id)
        if restored is None:
            raise OralSessionError("answer has no frozen roadmap")
        if control is Control.TOPICS:
            response = "; ".join(topic.label for topic in restored.roadmap.topics)
        elif control is Control.SUMMARIZE:
            response = self._recap(restored)
        elif control is Control.WHERE:
            unit = restored.current_unit
            response = "The answer is complete." if unit is None else f"At {unit.kind.value}."
        elif control is Control.DEEPER:
            response = "Repeating the current frozen section with its exact details."
        else:
            raise OralSessionError("control is unsupported")
        return NavigationResult(control, restored, restored.current_unit, response)


__all__ = [
    "Control",
    "ControlCommand",
    "FreezeResult",
    "FrozenAnswerState",
    "MAX_RECAP_CHARS",
    "NavigationResult",
    "OralSessionError",
    "OralSessionStore",
    "OralStatus",
    "PreviewClaim",
    "PreviewEffectState",
    "parse_control",
    "parse_control_command",
]
