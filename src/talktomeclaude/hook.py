"""Claude Code Stop-hook event handling.

A Stop event arrives as one JSON object on stdin. Per the platform contract,
the text of Claude's final reply is taken from ``last_assistant_message`` —
never recovered from ``transcript_path``, which is written asynchronously
and may not yet contain the final message when the hook fires.
"""

import json
from typing import TextIO

from talktomeclaude.transcript import speakable


def read_stop_event(stream: TextIO) -> dict | None:
    """Parse the hook event JSON from *stream*; None if it is unusable."""
    try:
        event = json.loads(stream.read())
    except (OSError, UnicodeDecodeError, ValueError):
        return None
    return event if isinstance(event, dict) else None


def stop_dialogue(event: dict) -> str:
    """Speakable dialogue of the turn's final assistant message.

    Empty when the event carries no ``last_assistant_message`` (older
    Claude Code versions, or turns that ended without a reply) or when
    nothing speakable remains after filtering.
    """
    message = event.get("last_assistant_message")
    if not isinstance(message, str):
        return ""
    return speakable(message)
