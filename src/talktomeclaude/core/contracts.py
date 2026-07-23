"""Platform-neutral contracts for the companion runtime.

The core deliberately carries no transcript, answer, window, process, or voice
content.  Adapters own those values and exchange only opaque generation tickets
and semantic lifecycle events with this package.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class RuntimePhase(str, Enum):
    """Semantic states exposed by every companion controller."""

    IDLE = "idle"
    RECORDING = "recording"
    TRANSCRIBING = "transcribing"
    AWAITING_CONFIRMATION = "awaiting_confirmation"
    DELIVERING = "delivering"
    WAITING_FOR_CLAUDE = "waiting_for_claude"
    PLANNING = "planning"
    SPEAKING = "speaking"
    PAUSED = "paused"
    STOPPING = "stopping"
    DISCONNECTED = "disconnected"
    RECOVERABLE_ERROR = "recoverable_error"


class EventKind(str, Enum):
    """Inputs understood by the pure state reducer."""

    START_RECORDING = "start_recording"
    FINISH_RECORDING = "finish_recording"
    TRANSCRIPT_ACCEPTED = "transcript_accepted"
    TRANSCRIPT_REVIEW_REQUIRED = "transcript_review_required"
    CONFIRM_TRANSCRIPT = "confirm_transcript"
    DELIVERY_SUCCEEDED = "delivery_succeeded"
    REPLY_RECEIVED = "reply_received"
    PLAN_READY = "plan_ready"
    PAUSE_SPEECH = "pause_speech"
    RESUME_SPEECH = "resume_speech"
    SPEECH_FINISHED = "speech_finished"
    TRANSPORT_DISCONNECTED = "transport_disconnected"
    TRANSPORT_RECONNECTED = "transport_reconnected"
    ERROR_OCCURRED = "error_occurred"
    RETRY = "retry"
    CANCEL = "cancel"
    STOP_REQUESTED = "stop_requested"
    STOPPED = "stopped"


class EffectKind(str, Enum):
    """Kinds of asynchronous work whose results must not cross generations."""

    CAPTURE = "capture"
    REPLY = "reply"
    PLAN = "plan"
    SPEECH = "speech"


class TransitionCode(str, Enum):
    APPLIED = "applied"
    ILLEGAL_TRANSITION = "illegal_transition"
    INVALID_EVENT = "invalid_event"
    STALE_GENERATION = "stale_generation"
    DEADLINE_NOT_EXPIRED = "deadline_not_expired"
    DEADLINE_NOT_OWNER = "deadline_not_owner"
    DEADLINE_NOT_APPLICABLE = "deadline_not_applicable"


@dataclass(frozen=True, slots=True)
class RuntimeState:
    """The complete reducer state.

    ``resume_phase`` is populated only while disconnected or in a recoverable
    error.  It is an explicit recovery destination rather than inferred history.
    """

    phase: RuntimePhase = RuntimePhase.IDLE
    generation: int = 0
    resume_phase: RuntimePhase | None = None
    error_code: str | None = None

    def __post_init__(self) -> None:
        if self.generation < 0:
            raise ValueError("generation must be non-negative")
        if self.phase is RuntimePhase.DISCONNECTED:
            if self.resume_phase is None:
                raise ValueError("disconnected state requires a resume phase")
            if self.resume_phase in {
                RuntimePhase.DISCONNECTED,
                RuntimePhase.RECOVERABLE_ERROR,
                RuntimePhase.STOPPING,
            }:
                raise ValueError("disconnected resume phase must be actionable")
            if self.error_code is not None:
                raise ValueError("only recoverable error state carries an error code")
            return
        if self.phase is RuntimePhase.RECOVERABLE_ERROR:
            if self.resume_phase is None or not self.error_code:
                raise ValueError(
                    "recoverable error requires an error code and resume phase"
                )
            if self.resume_phase in {
                RuntimePhase.RECOVERABLE_ERROR,
                RuntimePhase.STOPPING,
            }:
                raise ValueError("error resume phase must be actionable")
            return
        if self.resume_phase is not None or self.error_code is not None:
            raise ValueError(
                "resume phase and error code are valid only for recovery states"
            )


@dataclass(frozen=True, slots=True)
class RuntimeEvent:
    kind: EventKind
    generation: int | None = None
    error_code: str | None = None
    recover_to: RuntimePhase | None = None


@dataclass(frozen=True, slots=True)
class TransitionResult:
    accepted: bool
    code: TransitionCode
    previous: RuntimeState
    current: RuntimeState
    event: RuntimeEvent


@dataclass(frozen=True, slots=True)
class EffectTicket:
    """Authority for one asynchronous effect in one runtime generation."""

    kind: EffectKind
    generation: int


@dataclass(frozen=True, slots=True)
class EffectAcceptance:
    accepted: bool
    code: TransitionCode
    current_generation: int
    ticket: EffectTicket
