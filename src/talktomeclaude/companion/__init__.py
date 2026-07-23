"""Presentation contracts for the TalkToMeClaude companion."""

from talktomeclaude.companion.contracts import (
    CompanionIntent,
    CompanionSnapshot,
    IntentKind,
)
from talktomeclaude.companion.viewmodel import CompanionViewModel, to_view_model

__all__ = [
    "CompanionIntent",
    "CompanionSnapshot",
    "CompanionViewModel",
    "IntentKind",
    "to_view_model",
]
