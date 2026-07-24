"""Exact block accounting and protected-value preservation for oral roadmaps."""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, cast

from .canonical import CanonicalAnswer, ProtectedValue


class PreservationError(ValueError):
    """A proposed oral representation drops or changes canonical content."""


def _semantic_tokens(value: str) -> tuple[str, ...]:
    """Conservative wording proof: facts remain in the same lexical order."""

    # Whitespace and sentence cadence may change. Operators, emoji, decision
    # marks, path separators, quotes, and other symbols remain authoritative.
    return tuple(
        re.findall(r"\w+|[^\w\s.,;!?]", value, flags=re.UNICODE)
    )


def semantic_tokens_preserved(source: str, candidate: str) -> bool:
    return _semantic_tokens(candidate) == _semantic_tokens(source)


class BlockDispositionKind(StrEnum):
    SPOKEN = "spoken"
    VISIBLE_ONLY = "visible_only"
    DEFERRED = "deferred"


@dataclass(frozen=True, slots=True)
class BlockDisposition:
    """One explicit, immutable delivery decision for one canonical block."""

    block_id: str
    kind: BlockDispositionKind
    unit_id: str | None
    wording: str = field(default="", repr=False)

    def to_dict(self) -> dict[str, Any]:
        return {
            "block_id": self.block_id,
            "kind": self.kind.value,
            "unit_id": self.unit_id,
            "wording": self.wording,
        }

    @classmethod
    def from_dict(cls, value: object) -> "BlockDisposition":
        if not isinstance(value, dict) or set(value) != {
            "block_id",
            "kind",
            "unit_id",
            "wording",
        }:
            raise PreservationError("block disposition schema is invalid")
        block_id = value["block_id"]
        raw_kind = value["kind"]
        unit_id = value["unit_id"]
        wording = value["wording"]
        if (
            not isinstance(block_id, str)
            or not block_id
            or not isinstance(raw_kind, str)
            or (unit_id is not None and (not isinstance(unit_id, str) or not unit_id))
            or not isinstance(wording, str)
        ):
            raise PreservationError("block disposition values are invalid")
        try:
            kind = BlockDispositionKind(raw_kind)
        except ValueError as exc:
            raise PreservationError("block disposition kind is invalid") from exc
        if kind is BlockDispositionKind.SPOKEN:
            if unit_id is None or not wording:
                raise PreservationError("spoken block requires a unit and wording")
        elif unit_id is not None or wording:
            raise PreservationError(
                "visible-only and deferred blocks cannot carry spoken wording"
            )
        return cls(block_id, kind, unit_id, wording)


@dataclass(frozen=True, slots=True)
class PreservationReport:
    missing_block_ids: tuple[str, ...]
    duplicate_block_ids: tuple[str, ...]
    unknown_block_ids: tuple[str, ...]
    invalid_block_ids: tuple[str, ...]
    missing_values: tuple[tuple[str, ProtectedValue], ...]

    @property
    def valid(self) -> bool:
        return not (
            self.missing_block_ids
            or self.duplicate_block_ids
            or self.unknown_block_ids
            or self.invalid_block_ids
            or self.missing_values
        )


def _missing_protected_values(
    block_id: str,
    values: tuple[ProtectedValue, ...],
    wording: str,
) -> list[tuple[str, ProtectedValue]]:
    """Report missing occurrences, not just missing distinct strings."""

    by_value: dict[str, list[ProtectedValue]] = {}
    for value in values:
        by_value.setdefault(value.value, []).append(value)
    missing: list[tuple[str, ProtectedValue]] = []
    for exact, occurrences in by_value.items():
        actual = wording.count(exact)
        if actual < len(occurrences):
            missing.extend(
                (block_id, value) for value in occurrences[actual:]
            )
    return missing


def inspect_preservation(
    answer: CanonicalAnswer,
    dispositions: Iterable[BlockDisposition] | Iterable[str],
    legacy_wording_by_block: Mapping[str, Iterable[str]] | None = None,
) -> PreservationReport:
    """Require exactly one explicit disposition for every canonical block."""

    if legacy_wording_by_block is not None:
        items = tuple(
            BlockDisposition(
                block_id,
                BlockDispositionKind.SPOKEN,
                "legacy-section",
                "\n".join(legacy_wording_by_block.get(block_id, ())),
            )
            for block_id in dispositions
            if isinstance(block_id, str)
        )
    else:
        items = tuple(cast(Iterable[BlockDisposition], dispositions))
    if any(not isinstance(item, BlockDisposition) for item in items):
        raise PreservationError("oral roadmap block dispositions are invalid")
    expected = tuple(block.block_id for block in answer.blocks)
    expected_set = frozenset(expected)
    counts = Counter(item.block_id for item in items)
    missing = tuple(block_id for block_id in expected if counts[block_id] == 0)
    duplicate = tuple(block_id for block_id in expected if counts[block_id] > 1)
    unknown = tuple(sorted(block_id for block_id in counts if block_id not in expected_set))
    by_block = {item.block_id: item for item in items if counts[item.block_id] == 1}
    invalid: list[str] = []
    missing_values: list[tuple[str, ProtectedValue]] = []
    for block in answer.blocks:
        disposition = by_block.get(block.block_id)
        if disposition is None:
            continue
        if disposition.kind is BlockDispositionKind.SPOKEN:
            if disposition.unit_id is None or not disposition.wording:
                invalid.append(block.block_id)
                continue
            if not semantic_tokens_preserved(block.text, disposition.wording):
                invalid.append(block.block_id)
                continue
            missing_values.extend(
                _missing_protected_values(
                    block.block_id,
                    block.protected_values,
                    disposition.wording,
                )
            )
        elif disposition.unit_id is not None or disposition.wording:
            invalid.append(block.block_id)
    return PreservationReport(
        missing,
        duplicate,
        unknown,
        tuple(invalid),
        tuple(missing_values),
    )


def require_preservation(
    answer: CanonicalAnswer,
    dispositions: Iterable[BlockDisposition] | Iterable[str],
    legacy_wording_by_block: Mapping[str, Iterable[str]] | None = None,
) -> None:
    report = inspect_preservation(answer, dispositions, legacy_wording_by_block)
    if not report.valid:
        raise PreservationError(
            "oral roadmap does not preserve complete canonical block/value coverage"
        )


def wording_by_block(
    assignments: Iterable[tuple[Iterable[str], str]],
) -> dict[str, tuple[str, ...]]:
    """Compatibility collector retained while callers migrate to dispositions."""

    collected: defaultdict[str, list[str]] = defaultdict(list)
    for block_ids, wording in assignments:
        for block_id in block_ids:
            collected[block_id].append(wording)
    return {block_id: tuple(values) for block_id, values in collected.items()}


__all__ = [
    "BlockDisposition",
    "BlockDispositionKind",
    "PreservationError",
    "PreservationReport",
    "inspect_preservation",
    "require_preservation",
    "semantic_tokens_preserved",
    "wording_by_block",
]
