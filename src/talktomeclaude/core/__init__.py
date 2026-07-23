"""Deterministic, platform-neutral companion core."""

from .backoff import BackoffPolicy, JitteredBackoff
from .contracts import (
    EffectAcceptance,
    EffectKind,
    EffectTicket,
    EventKind,
    RuntimeEvent,
    RuntimePhase,
    RuntimeState,
    TransitionCode,
    TransitionResult,
)
from .deadlines import (
    DEFAULT_DEADLINES,
    Deadline,
    DeadlineName,
    DeadlinePolicy,
    DeadlineSpec,
)
from .runtime import RuntimeCoordinator
from .state import legal_events, reduce_state
from .workers import BoundedWorker, ShutdownResult

__all__ = [
    "BackoffPolicy",
    "BoundedWorker",
    "DEFAULT_DEADLINES",
    "Deadline",
    "DeadlineName",
    "DeadlinePolicy",
    "DeadlineSpec",
    "EffectAcceptance",
    "EffectKind",
    "EffectTicket",
    "EventKind",
    "JitteredBackoff",
    "RuntimeCoordinator",
    "RuntimeEvent",
    "RuntimePhase",
    "RuntimeState",
    "ShutdownResult",
    "TransitionCode",
    "TransitionResult",
    "legal_events",
    "reduce_state",
]
