"""Deterministic, hash-sealed, preservation-checked oral roadmap planning."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from dataclasses import dataclass, field, replace
from enum import StrEnum
from typing import Any

from .canonical import BlockKind, CanonicalAnswer
from .preservation import (
    BlockDisposition,
    BlockDispositionKind,
    PreservationError,
    require_preservation,
    semantic_tokens_preserved,
)

ROADMAP_VERSION = 1


class RoadmapError(ValueError):
    """An oral roadmap is invalid for its canonical answer."""


class StructuralMutationError(RoadmapError):
    """A late plan attempted to change frozen navigation structure."""


class UnitKind(StrEnum):
    OUTCOME = "outcome"
    PREVIEW = "preview"
    SECTION = "section"
    CHECKPOINT = "checkpoint"


def _strict_dict(value: object, keys: frozenset[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or frozenset(value) != keys:
        raise RoadmapError(f"{label} schema is invalid")
    return value


def _strict_string_list(
    value: object,
    label: str,
    *,
    allow_empty: bool = False,
) -> tuple[str, ...]:
    if not isinstance(value, list) or any(
        not isinstance(item, str) or not item for item in value
    ) or (not allow_empty and not value):
        raise RoadmapError(f"{label} is invalid")
    return tuple(value)


@dataclass(frozen=True, slots=True)
class OralTopic:
    topic_id: str
    label: str = field(repr=False)
    block_ids: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "block_ids": list(self.block_ids),
            "label": self.label,
            "topic_id": self.topic_id,
        }

    @classmethod
    def from_dict(cls, value: object) -> "OralTopic":
        raw = _strict_dict(
            value,
            frozenset({"block_ids", "label", "topic_id"}),
            "oral topic",
        )
        topic_id = raw["topic_id"]
        label = raw["label"]
        if (
            not isinstance(topic_id, str)
            or not topic_id
            or not isinstance(label, str)
            or not label
        ):
            raise RoadmapError("oral topic values are invalid")
        return cls(topic_id, label, _strict_string_list(raw["block_ids"], "topic blocks"))


@dataclass(frozen=True, slots=True)
class OralUnit:
    unit_id: str
    kind: UnitKind
    topic_id: str | None
    block_ids: tuple[str, ...]
    wording: str = field(repr=False)

    def to_dict(self) -> dict[str, Any]:
        return {
            "block_ids": list(self.block_ids),
            "kind": self.kind.value,
            "topic_id": self.topic_id,
            "unit_id": self.unit_id,
            "wording": self.wording,
        }

    @classmethod
    def from_dict(cls, value: object) -> "OralUnit":
        raw = _strict_dict(
            value,
            frozenset({"block_ids", "kind", "topic_id", "unit_id", "wording"}),
            "oral unit",
        )
        unit_id = raw["unit_id"]
        raw_kind = raw["kind"]
        topic_id = raw["topic_id"]
        wording = raw["wording"]
        if (
            not isinstance(unit_id, str)
            or not unit_id
            or not isinstance(raw_kind, str)
            or (topic_id is not None and (not isinstance(topic_id, str) or not topic_id))
            or not isinstance(wording, str)
        ):
            raise RoadmapError("oral unit values are invalid")
        try:
            kind = UnitKind(raw_kind)
        except ValueError as exc:
            raise RoadmapError("oral unit kind is invalid") from exc
        return cls(
            unit_id,
            kind,
            topic_id,
            _strict_string_list(
                raw["block_ids"],
                "unit blocks",
                allow_empty=True,
            ),
            wording,
        )


@dataclass(frozen=True, slots=True)
class OralRoadmap:
    version: int
    answer_id: str
    answer_digest: str
    complex: bool
    topics: tuple[OralTopic, ...]
    units: tuple[OralUnit, ...]
    checkpoint_sequence: tuple[str, ...]
    block_dispositions: tuple[BlockDisposition, ...]
    roadmap_hash: str

    def unit(self, unit_id: str) -> OralUnit:
        for unit in self.units:
            if unit.unit_id == unit_id:
                return unit
        raise KeyError(unit_id)

    def topic(self, topic_id: str) -> OralTopic:
        for topic in self.topics:
            if topic.topic_id == topic_id:
                return topic
        raise KeyError(topic_id)

    def disposition(self, block_id: str) -> BlockDisposition:
        for disposition in self.block_dispositions:
            if disposition.block_id == block_id:
                return disposition
        raise KeyError(block_id)

    def _payload_dict(self) -> dict[str, Any]:
        return {
            "answer_digest": self.answer_digest,
            "answer_id": self.answer_id,
            "block_dispositions": [
                disposition.to_dict() for disposition in self.block_dispositions
            ],
            "checkpoint_sequence": list(self.checkpoint_sequence),
            "complex": self.complex,
            "topics": [topic.to_dict() for topic in self.topics],
            "units": [unit.to_dict() for unit in self.units],
            "version": self.version,
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self._payload_dict(), "roadmap_hash": self.roadmap_hash}

    @classmethod
    def from_dict(cls, value: object) -> "OralRoadmap":
        raw = _strict_dict(
            value,
            frozenset(
                {
                    "answer_digest",
                    "answer_id",
                    "block_dispositions",
                    "checkpoint_sequence",
                    "complex",
                    "roadmap_hash",
                    "topics",
                    "units",
                    "version",
                }
            ),
            "oral roadmap",
        )
        if (
            type(raw["version"]) is not int
            or not isinstance(raw["answer_id"], str)
            or not raw["answer_id"]
            or not isinstance(raw["answer_digest"], str)
            or not raw["answer_digest"]
            or type(raw["complex"]) is not bool
            or not isinstance(raw["roadmap_hash"], str)
            or not raw["roadmap_hash"]
            or not isinstance(raw["topics"], list)
            or not isinstance(raw["units"], list)
            or not isinstance(raw["block_dispositions"], list)
        ):
            raise RoadmapError("oral roadmap values are invalid")
        try:
            dispositions = tuple(
                BlockDisposition.from_dict(item)
                for item in raw["block_dispositions"]
            )
        except PreservationError as exc:
            raise RoadmapError("oral roadmap dispositions are invalid") from exc
        roadmap = cls(
            raw["version"],
            raw["answer_id"],
            raw["answer_digest"],
            raw["complex"],
            tuple(OralTopic.from_dict(item) for item in raw["topics"]),
            tuple(OralUnit.from_dict(item) for item in raw["units"]),
            _strict_string_list(
                raw["checkpoint_sequence"],
                "checkpoint sequence",
                allow_empty=True,
            ),
            dispositions,
            raw["roadmap_hash"],
        )
        if roadmap.roadmap_hash != compute_roadmap_hash(roadmap):
            raise RoadmapError("oral roadmap hash is invalid")
        _validate_structure(roadmap)
        return roadmap

    def structural_signature(self) -> tuple[object, ...]:
        return (
            self.version,
            self.answer_id,
            self.answer_digest,
            self.complex,
            tuple((item.topic_id, item.label, item.block_ids) for item in self.topics),
            tuple(
                (item.unit_id, item.kind.value, item.topic_id, item.block_ids)
                for item in self.units
            ),
            self.checkpoint_sequence,
            tuple(
                (item.block_id, item.kind.value, item.unit_id)
                for item in self.block_dispositions
            ),
        )


def compute_roadmap_hash(roadmap: OralRoadmap) -> str:
    encoded = json.dumps(
        roadmap._payload_dict(),
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8", errors="strict")
    return hashlib.sha256(encoded).hexdigest()


def seal_roadmap(roadmap: OralRoadmap) -> OralRoadmap:
    """Return a roadmap carrying the hash of its complete immutable payload."""

    unsigned = replace(roadmap, roadmap_hash="")
    return replace(unsigned, roadmap_hash=compute_roadmap_hash(unsigned))


def _stable_id(prefix: str, answer: CanonicalAnswer, ordinal: int, material: str) -> str:
    digest = hashlib.sha256(
        f"{answer.digest}\0{ordinal}\0{material}".encode("utf-8")
    ).hexdigest()[:16]
    return f"{prefix}-{ordinal:02d}-{digest}"


def _label(answer: CanonicalAnswer, block_ids: tuple[str, ...], ordinal: int) -> str:
    blocks = [answer.block(block_id) for block_id in block_ids]
    for block in blocks:
        if block.kind is BlockKind.HEADING:
            value = re.sub(r"^\s*#{1,6}\s*", "", block.text).strip()
            if value:
                return value[:80]
    words = " ".join("".join(block.text for block in blocks).split()).split()
    stem = " ".join(words[:7]).rstrip(".,:;!?")
    return stem[:80] if stem else f"Part {ordinal + 1}"


def _groups(answer: CanonicalAnswer) -> tuple[tuple[str, ...], ...]:
    blocks = answer.blocks
    if len(blocks) < 2:
        return (tuple(block.block_id for block in blocks),)
    heading_starts = [
        index for index, block in enumerate(blocks) if block.kind is BlockKind.HEADING
    ]
    groups: list[tuple[str, ...]] = []
    if heading_starts:
        starts = ([0] if heading_starts[0] else []) + heading_starts
        starts = sorted(set(starts))
        for position, start in enumerate(starts):
            end = starts[position + 1] if position + 1 < len(starts) else len(blocks)
            groups.append(tuple(block.block_id for block in blocks[start:end]))
    if len(groups) < 2:
        desired = min(4, max(2, math.ceil(len(blocks) / 3)))
        groups = []
        for ordinal in range(desired):
            start = ordinal * len(blocks) // desired
            end = (ordinal + 1) * len(blocks) // desired
            groups.append(tuple(block.block_id for block in blocks[start:end]))
    if len(groups) > 4:
        groups = groups[:3] + [tuple(item for group in groups[3:] for item in group)]
    return tuple(group for group in groups if group)


def _is_complex(answer: CanonicalAnswer) -> bool:
    if len(answer.blocks) < 2:
        return False
    headings = sum(block.kind is BlockKind.HEADING for block in answer.blocks)
    return len(answer.blocks) >= 3 or headings >= 2 or len(answer.text) > 400


def _outcome(answer: CanonicalAnswer) -> str:
    source = next(
        (
            block.text.strip()
            for block in answer.blocks
            if block.kind in (BlockKind.PROSE, BlockKind.QUOTE) and block.text.strip()
        ),
        answer.blocks[0].text.strip(),
    )
    sentences = [item for item in re.split(r"(?<=[.!?])\s+", source) if item]
    return " ".join(sentences[:2])


def _sentence_count(value: str) -> int:
    return len([item for item in re.split(r"(?<=[.!?])\s+", value.strip()) if item])


def _validate_structure(roadmap: OralRoadmap) -> None:
    if roadmap.version != ROADMAP_VERSION or not roadmap.topics or not roadmap.units:
        raise RoadmapError("roadmap identity or content is invalid")
    topic_ids = tuple(topic.topic_id for topic in roadmap.topics)
    if (
        len(set(topic_ids)) != len(topic_ids)
        or any(not topic.label or not topic.block_ids for topic in roadmap.topics)
    ):
        raise RoadmapError("roadmap topics are invalid")
    owned = tuple(block_id for topic in roadmap.topics for block_id in topic.block_ids)
    if len(set(owned)) != len(owned):
        raise RoadmapError("roadmap topic ownership is not one-to-one")
    unit_ids = tuple(unit.unit_id for unit in roadmap.units)
    if len(set(unit_ids)) != len(unit_ids):
        raise RoadmapError("roadmap unit identities are not unique")

    if roadmap.complex:
        if not (2 <= len(roadmap.topics) <= 4):
            raise RoadmapError("complex roadmap must contain two to four topics")
        expected_length = 2 + 2 * len(roadmap.topics)
        if len(roadmap.units) != expected_length:
            raise RoadmapError("complex roadmap unit sequence is invalid")
        outcome, preview = roadmap.units[:2]
        if (
            outcome.kind is not UnitKind.OUTCOME
            or outcome.topic_id is not None
            or outcome.block_ids
            or not outcome.wording
            or not (1 <= _sentence_count(outcome.wording) <= 2)
            or preview.kind is not UnitKind.PREVIEW
            or preview.topic_id is not None
            or preview.block_ids
            or not preview.wording
        ):
            raise RoadmapError("complex roadmap framing is invalid")
        body = roadmap.units[2:]
    else:
        if len(roadmap.topics) != 1 or len(roadmap.units) != 1:
            raise RoadmapError("simple roadmap must contain one direct section")
        body = roadmap.units

    checkpoints: list[str] = []
    section_by_topic: dict[str, OralUnit] = {}
    for ordinal, topic in enumerate(roadmap.topics):
        offset = ordinal * 2 if roadmap.complex else ordinal
        section = body[offset]
        if (
            section.kind is not UnitKind.SECTION
            or section.topic_id != topic.topic_id
            or section.block_ids != topic.block_ids
        ):
            raise RoadmapError("roadmap section ownership/order is invalid")
        section_by_topic[topic.topic_id] = section
        if roadmap.complex:
            checkpoint = body[offset + 1]
            if (
                checkpoint.kind is not UnitKind.CHECKPOINT
                or checkpoint.topic_id != topic.topic_id
                or checkpoint.block_ids
                or not checkpoint.wording
            ):
                raise RoadmapError("roadmap checkpoint order is invalid")
            checkpoints.append(checkpoint.unit_id)
    if tuple(checkpoints) != roadmap.checkpoint_sequence:
        raise RoadmapError("checkpoint sequence is invalid")

    dispositions = roadmap.block_dispositions
    disposition_ids = tuple(item.block_id for item in dispositions)
    if Counter(disposition_ids) != Counter(owned):
        raise RoadmapError("roadmap block dispositions are not one-to-one")
    by_block = {item.block_id: item for item in dispositions}
    for topic in roadmap.topics:
        section = section_by_topic[topic.topic_id]
        spoken_wording: list[str] = []
        for block_id in topic.block_ids:
            disposition = by_block[block_id]
            if disposition.kind is BlockDispositionKind.SPOKEN:
                if disposition.unit_id != section.unit_id:
                    raise RoadmapError("spoken block points at the wrong section")
                spoken_wording.append(disposition.wording)
            elif disposition.unit_id is not None or disposition.wording:
                raise RoadmapError("non-spoken block carries speech authority")
        expected_wording = "".join(spoken_wording)
        if not expected_wording or section.wording != expected_wording:
            raise RoadmapError("section wording does not match block dispositions")


def validate_roadmap(answer: CanonicalAnswer, roadmap: OralRoadmap) -> None:
    if roadmap.roadmap_hash != compute_roadmap_hash(roadmap):
        raise RoadmapError("oral roadmap hash is invalid")
    _validate_structure(roadmap)
    if roadmap.answer_id != answer.answer_id or roadmap.answer_digest != answer.digest:
        raise RoadmapError("roadmap identity does not match the canonical answer")
    expected = tuple(block.block_id for block in answer.blocks)
    owned = tuple(block_id for topic in roadmap.topics for block_id in topic.block_ids)
    if Counter(owned) != Counter(expected):
        raise RoadmapError("roadmap topics do not own every canonical block exactly once")
    if tuple(item.block_id for item in roadmap.block_dispositions) != expected:
        raise RoadmapError("roadmap dispositions do not follow canonical block order")
    for ordinal, topic in enumerate(roadmap.topics):
        if topic.label != _label(answer, topic.block_ids, ordinal):
            raise RoadmapError("roadmap topic label is not canonical")
    if roadmap.complex:
        if not semantic_tokens_preserved(
            _outcome(answer), roadmap.units[0].wording
        ):
            raise RoadmapError("roadmap outcome is not canonical")
        expected_preview = (
            "I'll cover "
            + "; ".join(topic.label for topic in roadmap.topics)
            + "."
        )
        if not semantic_tokens_preserved(
            expected_preview, roadmap.units[1].wording
        ):
            raise RoadmapError("roadmap preview is not canonical")
        body = roadmap.units[2:]
        for ordinal, topic in enumerate(roadmap.topics):
            checkpoint = body[ordinal * 2 + 1]
            if not semantic_tokens_preserved(
                f"That completes {topic.label}.", checkpoint.wording
            ):
                raise RoadmapError("roadmap checkpoint is not canonical")
    try:
        require_preservation(answer, roadmap.block_dispositions)
    except PreservationError as exc:
        raise RoadmapError("roadmap does not preserve canonical content") from exc


def deterministic_plan(answer: CanonicalAnswer) -> OralRoadmap:
    """Build a stable simple/complex plan without inference or side effects."""

    complex_answer = _is_complex(answer)
    groups = _groups(answer) if complex_answer else (
        tuple(block.block_id for block in answer.blocks),
    )
    topics = tuple(
        OralTopic(
            _stable_id("topic", answer, ordinal, "\0".join(group)),
            _label(answer, group, ordinal),
            group,
        )
        for ordinal, group in enumerate(groups)
    )
    units: list[OralUnit] = []
    if complex_answer:
        units.extend(
            (
                OralUnit(
                    _stable_id("unit", answer, 0, "outcome"),
                    UnitKind.OUTCOME,
                    None,
                    (),
                    _outcome(answer),
                ),
                OralUnit(
                    _stable_id("unit", answer, 1, "preview"),
                    UnitKind.PREVIEW,
                    None,
                    (),
                    "I'll cover " + "; ".join(topic.label for topic in topics) + ".",
                ),
            )
        )
    dispositions: dict[str, BlockDisposition] = {}
    checkpoint_ids: list[str] = []
    for topic in topics:
        section_id = _stable_id("unit", answer, len(units), topic.topic_id + ":section")
        for block_id in topic.block_ids:
            dispositions[block_id] = BlockDisposition(
                block_id,
                BlockDispositionKind.SPOKEN,
                section_id,
                answer.block(block_id).text,
            )
        units.append(
            OralUnit(
                section_id,
                UnitKind.SECTION,
                topic.topic_id,
                topic.block_ids,
                "".join(dispositions[item].wording for item in topic.block_ids),
            )
        )
        if complex_answer:
            checkpoint_id = _stable_id(
                "unit", answer, len(units), topic.topic_id + ":checkpoint"
            )
            checkpoint_ids.append(checkpoint_id)
            units.append(
                OralUnit(
                    checkpoint_id,
                    UnitKind.CHECKPOINT,
                    topic.topic_id,
                    (),
                    f"That completes {topic.label}.",
                )
            )
    roadmap = seal_roadmap(
        OralRoadmap(
            ROADMAP_VERSION,
            answer.answer_id,
            answer.digest,
            complex_answer,
            topics,
            tuple(units),
            tuple(checkpoint_ids),
            tuple(dispositions[block.block_id] for block in answer.blocks),
            "",
        )
    )
    validate_roadmap(answer, roadmap)
    return roadmap


def refine_unsaid(
    answer: CanonicalAnswer,
    frozen: OralRoadmap,
    candidate: OralRoadmap,
    *,
    spoken_unit_ids: frozenset[str],
) -> OralRoadmap:
    """Accept wording-only changes inside unsaid frozen unit boundaries."""

    validate_roadmap(answer, frozen)
    try:
        validate_roadmap(answer, candidate)
    except RoadmapError as exc:
        raise StructuralMutationError("late roadmap is not preservation-safe") from exc
    if frozen.structural_signature() != candidate.structural_signature():
        raise StructuralMutationError("late roadmap changed frozen structure")
    frozen_units = {unit.unit_id: unit for unit in frozen.units}
    frozen_dispositions = {
        item.block_id: item for item in frozen.block_dispositions
    }
    for unit in candidate.units:
        if unit.unit_id in spoken_unit_ids and unit.wording != frozen_units[unit.unit_id].wording:
            raise StructuralMutationError("late roadmap changed already spoken wording")
    for disposition in candidate.block_dispositions:
        if (
            disposition.unit_id in spoken_unit_ids
            and disposition.wording
            != frozen_dispositions[disposition.block_id].wording
        ):
            raise StructuralMutationError("late roadmap changed already spoken block wording")
    refined = seal_roadmap(
        replace(
            frozen,
            units=candidate.units,
            block_dispositions=candidate.block_dispositions,
            roadmap_hash="",
        )
    )
    validate_roadmap(answer, refined)
    return refined


__all__ = [
    "OralRoadmap",
    "OralTopic",
    "OralUnit",
    "ROADMAP_VERSION",
    "RoadmapError",
    "StructuralMutationError",
    "UnitKind",
    "compute_roadmap_hash",
    "deterministic_plan",
    "refine_unsaid",
    "seal_roadmap",
    "validate_roadmap",
]
