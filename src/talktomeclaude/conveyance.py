"""Chunk spoken answers and persist their per-session delivery position."""

from __future__ import annotations

import json
import re
from pathlib import Path

MAX_CHUNK_WORDS = 75
SCRATCHPAD_KEYS = ("cursor", "heading", "status")

_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+")


def chunk(text: str) -> list[str]:
    """Split *text* into sentence-aligned chunks under the spoken-word cap."""
    sentences = _SENTENCE_BOUNDARY.split(text.strip())
    chunks: list[str] = []
    current: list[str] = []
    current_words = 0

    for sentence in sentences:
        words = sentence.split()
        if not words:
            continue

        if len(words) > MAX_CHUNK_WORDS:
            if current:
                chunks.append(" ".join(current))
                current = []
                current_words = 0
            chunks.extend(
                " ".join(words[start : start + MAX_CHUNK_WORDS])
                for start in range(0, len(words), MAX_CHUNK_WORDS)
            )
            continue

        if current and current_words + len(words) > MAX_CHUNK_WORDS:
            chunks.append(" ".join(current))
            current = []
            current_words = 0

        current.append(sentence)
        current_words += len(words)

    if current:
        chunks.append(" ".join(current))
    return chunks


def scratchpad_path(session_id: str) -> Path:
    """Return the working-directory-relative voice scratchpad path."""
    return (
        Path(".omc")
        / "state"
        / "sessions"
        / session_id
        / "voice-conveyance.json"
    )


def write_scratchpad(
    session_id: str, *, cursor: int, heading: str, status: str
) -> None:
    """Atomically persist the locked voice-conveyance fields."""
    path = scratchpad_path(session_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    swap = path.with_name(path.name + ".tmp")
    data = {"cursor": cursor, "heading": heading, "status": status}
    swap.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    swap.replace(path)


def read_scratchpad(session_id: str) -> dict:
    """Read known scratchpad fields, returning empty state on any read failure."""
    try:
        with scratchpad_path(session_id).open(encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, UnicodeError, json.JSONDecodeError):
        return {}
    if not isinstance(data, dict):
        return {}
    return {key: data[key] for key in SCRATCHPAD_KEYS if key in data}
