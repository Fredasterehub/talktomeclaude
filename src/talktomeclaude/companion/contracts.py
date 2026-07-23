"""Small, UI-independent contracts shared by companion presentations.

The desktop shell and headless recovery path consume these values.  They do
not own capture, delivery, or speech behavior; those services will eventually
publish snapshots and accept intents behind this boundary.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from talktomeclaude.core import RuntimeState


class IntentKind(str, Enum):
    """User actions accepted by companion controllers."""

    STATUS = "status"
    START_RECORDING = "start-recording"
    FINISH_RECORDING = "finish-recording"
    CANCEL = "cancel"
    OPEN_SETTINGS = "open-settings"
    QUIT = "quit"


@dataclass(frozen=True, slots=True)
class CompanionIntent:
    """A presentation-neutral request from a user-facing controller."""

    kind: IntentKind


@dataclass(frozen=True, slots=True)
class CompanionSnapshot:
    """Content-safe state published to a companion presentation."""

    runtime: RuntimeState
    detail: str = ""
