"""Extract Claude's spoken dialogue from Claude Code transcripts.

A Claude Code transcript is a JSONL file of typed entries. Only assistant
prose is ever spoken: tool calls, tool results, thinking blocks, sidechain
(subagent) traffic, and code — fenced blocks and inline spans alike — are
silently dropped, and remaining markdown is flattened into speakable text.
"""

import json
import re
from typing import Iterable, Iterator

_FENCE_RE = re.compile(r"^\s*(```+|~~~+)")
_TABLE_SEPARATOR_RE = re.compile(r"^\s*\|?[\s|:-]+\|[\s|:-]*$")
_HORIZONTAL_RULE_RE = re.compile(r"^\s{0,3}([-*_])\s*(?:\1\s*){2,}$")
_HEADER_RE = re.compile(r"^\s{0,3}#{1,6}\s+")
_BULLET_RE = re.compile(r"^\s*(?:[-*+]|\d{1,3}[.)])\s+")
_BLOCKQUOTE_RE = re.compile(r"^\s{0,3}(?:>\s?)+")
_IMAGE_RE = re.compile(r"!\[([^\]]*)\]\([^)]*\)")
_LINK_RE = re.compile(r"\[([^\]]+)\]\([^)]*\)")
_INLINE_CODE_RE = re.compile(r"`([^`\n]*)`")
_BOLD_RE = re.compile(r"(\*\*|__)(.+?)\1")
_ITALIC_RE = re.compile(r"(?<!\w)\*([^*\n]+)\*(?!\w)")


def _strip_fenced_blocks(text: str) -> str:
    """Remove fenced code blocks, including everything after an unclosed fence."""
    kept = []
    in_fence = False
    for line in text.splitlines():
        if _FENCE_RE.match(line):
            in_fence = not in_fence
            continue
        if not in_fence:
            kept.append(line)
    return "\n".join(kept)


def speakable(text: str) -> str:
    """Flatten assistant markdown prose into text safe to hand to a voice."""
    lines = []
    for line in _strip_fenced_blocks(text).splitlines():
        stripped = line.strip()
        if not stripped:
            lines.append("")
            continue
        if stripped.startswith("|") or _TABLE_SEPARATOR_RE.match(stripped):
            continue
        if _HORIZONTAL_RULE_RE.match(stripped):
            continue
        line = _HEADER_RE.sub("", line)
        line = _BLOCKQUOTE_RE.sub("", line)
        line = _BULLET_RE.sub("", line)
        line = _IMAGE_RE.sub(r"\1", line)
        line = _LINK_RE.sub(r"\1", line)
        line = _INLINE_CODE_RE.sub("", line)  # code is never spoken, even inline
        line = _BOLD_RE.sub(r"\2", line)
        line = _ITALIC_RE.sub(r"\1", line)
        line = line.replace("`", "")
        line = re.sub(r" {2,}", " ", line)
        lines.append(line.strip())
    text = "\n".join(lines)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _text_blocks(entry: dict) -> Iterator[str]:
    """Yield the raw text blocks of one assistant transcript entry."""
    message = entry.get("message")
    if not isinstance(message, dict):
        return
    content = message.get("content")
    if isinstance(content, str):
        yield content
        return
    if not isinstance(content, list):
        return
    for block in content:
        if isinstance(block, dict) and block.get("type") == "text":
            text = block.get("text")
            if isinstance(text, str):
                yield text


def iter_dialogue(lines: Iterable[str]) -> Iterator[str]:
    """Yield speakable assistant dialogue from JSONL transcript lines.

    Everything that is not main-chain assistant prose — user turns, tool
    calls, tool results, thinking, sidechains, malformed lines — is dropped.
    """
    for raw in lines:
        raw = raw.strip()
        if not raw:
            continue
        try:
            entry = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if not isinstance(entry, dict):
            continue
        if entry.get("type") != "assistant" or entry.get("isSidechain"):
            continue
        for text in _text_blocks(entry):
            spoken = speakable(text)
            if spoken:
                yield spoken
