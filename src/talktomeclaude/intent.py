"""Pure helpers for resolving voice-command intent."""

from __future__ import annotations

from dataclasses import dataclass
import json
import shutil
from typing import Any


MAX_SPOKEN_ALTERNATIVES = 3


@dataclass(frozen=True, slots=True)
class IntentResponse:
    """The locked response contract returned by intent classification."""

    command_id: Any
    args: Any
    missing_slots: Any
    confidence: Any
    alternatives: Any


def sanitize_missing_slots(raw) -> list[str]:
    """Normalize untrusted missing_slots to unique nonempty strings."""
    if not isinstance(raw, list):
        return []
    slots: list[str] = []
    for item in raw:
        if isinstance(item, str):
            slot = item.strip()
            if slot and slot not in slots:
                slots.append(slot)
    return slots


def sanitize_alternatives(raw, catalog: list[dict]) -> list[str]:
    """Reduce untrusted alternatives to canonical qualified command
    identities (``namespace:id``, bare id at top level), deduped and capped
    for speech; bare names resolve only when they identify one command."""
    if not isinstance(raw, list):
        return []
    counts: dict[str, int] = {}
    for entry in catalog:
        counts[entry["id"]] = counts.get(entry["id"], 0) + 1
    canonical: dict[str, str] = {}
    for entry in catalog:
        namespace = entry["namespace"]
        qualified = f"{namespace}:{entry['id']}" if namespace else str(entry["id"])
        canonical[qualified.casefold()] = qualified
        if counts[entry["id"]] == 1:
            canonical.setdefault(str(entry["id"]).casefold(), qualified)
    alternatives: list[str] = []
    for item in raw:
        if not isinstance(item, str):
            continue
        identity = canonical.get(item.strip().casefold())
        if identity is not None and identity not in alternatives:
            alternatives.append(identity)
            if len(alternatives) == MAX_SPOKEN_ALTERNATIVES:
                break
    return alternatives


def keyword_prefilter(utterance: str, catalog: list[dict]) -> str | None:
    """Return the command id when *utterance* exactly identifies a command."""
    candidate = utterance.strip().casefold()
    for entry in catalog:
        command_id = entry["id"]
        namespace = entry["namespace"]
        names = [command_id]
        if namespace:
            names.append(f"{namespace}:{command_id}")
        if candidate in (name.casefold() for name in names):
            return command_id
    return None


def intent_subcall_command(prompt: str, model: str) -> list[str]:
    """Build an isolated Claude intent-classification command."""
    claude = shutil.which("claude") or "claude"
    return [
        claude,
        "-p",
        prompt,
        "--model",
        model,
        "--output-format",
        "json",
    ]


def parse_intent_response(json_str: str) -> IntentResponse:
    """Parse intent-classification JSON into its immutable value type."""
    payload = json.loads(json_str)
    return IntentResponse(
        command_id=payload["command_id"],
        args=payload["args"],
        missing_slots=payload["missing_slots"],
        confidence=payload["confidence"],
        alternatives=payload["alternatives"],
    )
