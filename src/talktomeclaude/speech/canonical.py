"""Immutable canonical answers with stable, lossless source blocks."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from enum import StrEnum

CANONICAL_VERSION = 1


class CanonicalError(ValueError):
    """An answer cannot be represented by the canonical speech contract."""


class BlockKind(StrEnum):
    HEADING = "heading"
    PROSE = "prose"
    FENCE = "fence"
    TABLE = "table"
    LIST = "list"
    QUOTE = "quote"


class ProtectedKind(StrEnum):
    CITATION = "citation"
    PATH = "path"
    COMMAND = "command"
    NUMBER = "number"
    UNCERTAINTY = "uncertainty"
    RISK = "risk"


@dataclass(frozen=True, slots=True)
class ProtectedValue:
    kind: ProtectedKind
    value: str = field(repr=False)
    start: int
    end: int


@dataclass(frozen=True, slots=True)
class CanonicalBlock:
    block_id: str
    ordinal: int
    kind: BlockKind
    text: str = field(repr=False)
    protected_values: tuple[ProtectedValue, ...] = ()


@dataclass(frozen=True, slots=True)
class CanonicalAnswer:
    version: int
    answer_id: str
    digest: str
    text: str = field(repr=False)
    blocks: tuple[CanonicalBlock, ...]

    def __post_init__(self) -> None:
        if self.version != CANONICAL_VERSION:
            raise CanonicalError("canonical answer version is unsupported")
        if not _safe_identifier(self.answer_id):
            raise CanonicalError("answer identity is invalid")
        expected = hashlib.sha256(self.text.encode("utf-8", errors="strict")).hexdigest()
        if self.digest != expected:
            raise CanonicalError("canonical answer digest is invalid")
        if not self.blocks or "".join(block.text for block in self.blocks) != self.text:
            raise CanonicalError("canonical blocks do not cover the answer exactly")
        if tuple(block.ordinal for block in self.blocks) != tuple(range(len(self.blocks))):
            raise CanonicalError("canonical block ordinals are invalid")
        if len({block.block_id for block in self.blocks}) != len(self.blocks):
            raise CanonicalError("canonical block identities are not unique")

    def block(self, block_id: str) -> CanonicalBlock:
        for block in self.blocks:
            if block.block_id == block_id:
                return block
        raise KeyError(block_id)


_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,127}$")
_HEADING = re.compile(r"^\s{0,3}#{1,6}\s+\S")
_FENCE = re.compile(r"^\s{0,3}(`{3,}|~{3,})")
_LIST = re.compile(r"^\s*(?:[-+*]|\d+[.)])\s+\S")
_QUOTE = re.compile(r"^\s*>\s?.*")
_TABLE_SEPARATOR = re.compile(
    r"^\s*\|?\s*:?-{3,}:?\s*(?:\|\s*:?-{3,}:?\s*)+\|?\s*$"
)

_PROTECTED_PATTERNS: tuple[tuple[ProtectedKind, re.Pattern[str]], ...] = (
    (
        ProtectedKind.CITATION,
        re.compile(r"\[[^\]\r\n]+\]\([^\)\r\n]+\)|\[(?:\d+|[A-Z][A-Za-z.-]*\s*\d*)\]"),
    ),
    (
        ProtectedKind.PATH,
        re.compile(
            r"(?<![\w])(?:[A-Za-z]:\\(?:[^\s<>:\"|?*]+\\?)+|/(?:[^\s/]+/)*[^\s/]+)"
        ),
    ),
    (
        ProtectedKind.COMMAND,
        re.compile(r"`[^`\r\n]+`|^\s*[$>]\s+[^\r\n]+", re.MULTILINE),
    ),
    (
        ProtectedKind.NUMBER,
        re.compile(
            r"(?<![\w])[-+]?\d+(?:[.,]\d+)*(?:%|\s?(?:ms|s|MB|GB|Hz|kHz|MHz|GHz|°C|°F))?(?![\w])"
        ),
    ),
    (
        ProtectedKind.UNCERTAINTY,
        re.compile(r"(?i)\b(?:may|might|could|likely|unlikely|probably|uncertain)\b"),
    ),
    (
        ProtectedKind.RISK,
        re.compile(r"(?i)\b(?:risk|warning|danger|caution|must|never|do not)\b"),
    ),
)


def _safe_identifier(value: object) -> bool:
    return isinstance(value, str) and _IDENTIFIER.fullmatch(value) is not None


def _line_kind(lines: list[str], index: int) -> BlockKind:
    line = lines[index]
    if _FENCE.match(line):
        return BlockKind.FENCE
    if _HEADING.match(line):
        return BlockKind.HEADING
    if _LIST.match(line):
        return BlockKind.LIST
    if _QUOTE.match(line):
        return BlockKind.QUOTE
    if "|" in line and index + 1 < len(lines) and _TABLE_SEPARATOR.match(lines[index + 1]):
        return BlockKind.TABLE
    return BlockKind.PROSE


def _consume_block(lines: list[str], start: int, kind: BlockKind) -> int:
    if kind is BlockKind.FENCE:
        opening = _FENCE.match(lines[start])
        assert opening is not None
        marker = opening.group(1)
        closing = re.compile(rf"^\s{{0,3}}{re.escape(marker[0])}{{{len(marker)},}}\s*$")
        cursor = start + 1
        while cursor < len(lines):
            if closing.match(lines[cursor].rstrip("\r\n")):
                return cursor + 1
            cursor += 1
        return len(lines)
    if kind is BlockKind.HEADING:
        return start + 1
    if kind is BlockKind.TABLE:
        cursor = start + 2
        while cursor < len(lines) and "|" in lines[cursor] and lines[cursor].strip():
            cursor += 1
        return cursor
    if kind in (BlockKind.LIST, BlockKind.QUOTE):
        matcher = _LIST if kind is BlockKind.LIST else _QUOTE
        cursor = start + 1
        while cursor < len(lines):
            line = lines[cursor]
            if not line.strip():
                return cursor + 1
            if matcher.match(line) or line.startswith((" ", "\t")):
                cursor += 1
                continue
            break
        return cursor

    cursor = start + 1
    while cursor < len(lines):
        if not lines[cursor].strip():
            cursor += 1
            break
        if _line_kind(lines, cursor) is not BlockKind.PROSE:
            break
        cursor += 1
    return cursor


def protected_values(text: str) -> tuple[ProtectedValue, ...]:
    """Return deterministic, non-overlapping exact values that speech must preserve."""

    candidates: list[ProtectedValue] = []
    for kind, pattern in _PROTECTED_PATTERNS:
        for match in pattern.finditer(text):
            candidates.append(ProtectedValue(kind, match.group(0), match.start(), match.end()))
    candidates.sort(key=lambda item: (item.start, -(item.end - item.start), item.kind.value))
    accepted: list[ProtectedValue] = []
    end = -1
    for item in candidates:
        if item.start < end:
            continue
        accepted.append(item)
        end = item.end
    return tuple(accepted)


def canonicalize(answer_id: str, text: str) -> CanonicalAnswer:
    """Create an immutable, byte-for-byte-covered canonical answer."""

    if not _safe_identifier(answer_id):
        raise CanonicalError("answer identity is invalid")
    if not isinstance(text, str) or not text:
        raise CanonicalError("answer text is empty")
    try:
        encoded = text.encode("utf-8", errors="strict")
    except UnicodeError as exc:
        raise CanonicalError("answer text is not valid Unicode") from exc
    digest = hashlib.sha256(encoded).hexdigest()
    lines = text.splitlines(keepends=True)
    if not lines:
        lines = [text]
    blocks: list[CanonicalBlock] = []
    cursor = 0
    while cursor < len(lines):
        kind = _line_kind(lines, cursor)
        end = _consume_block(lines, cursor, kind)
        block_text = "".join(lines[cursor:end])
        ordinal = len(blocks)
        identity_material = f"{digest}\0{ordinal}\0{block_text}".encode("utf-8")
        identity = hashlib.sha256(identity_material).hexdigest()[:16]
        blocks.append(
            CanonicalBlock(
                block_id=f"block-{ordinal:04d}-{identity}",
                ordinal=ordinal,
                kind=kind,
                text=block_text,
                protected_values=protected_values(block_text),
            )
        )
        cursor = end
    return CanonicalAnswer(CANONICAL_VERSION, answer_id, digest, text, tuple(blocks))
