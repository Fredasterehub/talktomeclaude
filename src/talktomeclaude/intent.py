"""Pure helpers for resolving voice-command intent."""

from __future__ import annotations

from dataclasses import dataclass
import json
import shutil
from typing import Any


@dataclass(frozen=True, slots=True)
class IntentResponse:
    """The locked response contract returned by intent classification."""

    command_id: Any
    args: Any
    missing_slots: Any
    confidence: Any
    alternatives: Any


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
