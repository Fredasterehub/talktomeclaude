"""Platform-neutral contracts for companion audio capture.

The capture package intentionally knows nothing about HWNDs, clipboard access,
or terminal injection.  It emits one opaque foreground-snapshot request at the
operator's finish-toggle boundary and leaves validation and delivery to the
Windows platform adapter.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Protocol, TypeAlias


class CaptureMode(str, Enum):
    """Explicit companion capture modes.

    ``PUSH_TOGGLE`` is the default.  Silence is deliberately absent from the
    contract: only a second toggle, cancellation, or the configured safety
    ceiling can finish a toggle-mode take.
    """

    PUSH_TOGGLE = "push-toggle"
    # Preserve the existing config/CLI vocabulary while giving the companion
    # contract the clearer semantic name.
    HOLD_TO_TALK = "push-to-talk"


class CapturePhase(str, Enum):
    IDLE = "idle"
    RECORDING = "recording"
    PRESERVED = "preserved"


class CaptureEnd(str, Enum):
    FINISH_TOGGLE = "finish_toggle"
    KEY_RELEASE = "key_release"
    SAFETY_CEILING = "safety_ceiling"
    CANCELLED = "cancelled"


class SafetyNoticeCode(str, Enum):
    CEILING_APPROACHING = "ceiling_approaching"
    CEILING_REACHED = "ceiling_reached"


class TranscriptDisposition(str, Enum):
    ACCEPTED = "accepted"
    EMPTY = "empty"
    LOW_CONFIDENCE = "low_confidence"
    CANCELLED = "cancelled"
    STALE_GENERATION = "stale_generation"
    CONSTRUCTION_TIMEOUT = "construction_timeout"
    ITERATION_TIMEOUT = "iteration_timeout"
    BOUNDARY_TAINTED = "boundary_tainted"


class CaptureContractError(RuntimeError):
    """Raised when an intent is invalid for the current capture phase."""


class CaptureCancelled(RuntimeError):
    """Raised at an owned boundary after cancellation was requested."""


@dataclass(frozen=True, slots=True)
class CaptureSettings:
    """User-configurable capture policy.

    The default ceiling is intentionally high enough for the required
    ten-minute paused recording.  It is a safety stop rather than speech/VAD
    segmentation and never drops the collected chunks.
    """

    mode: CaptureMode = CaptureMode.PUSH_TOGGLE
    # Unified with core DeadlineName.CAPTURE_SAFETY_CEILING.
    safety_ceiling_seconds: float = 900.0
    warning_before_seconds: float = 60.0
    minimum_confidence: float = 0.50

    def __post_init__(self) -> None:
        if self.safety_ceiling_seconds <= 0:
            raise ValueError("safety ceiling must be positive")
        if not 0 <= self.warning_before_seconds < self.safety_ceiling_seconds:
            raise ValueError("warning must be within the safety ceiling")
        if not 0 <= self.minimum_confidence <= 1:
            raise ValueError("minimum confidence must be between 0 and 1")


@dataclass(frozen=True, slots=True)
class SafetyNotice:
    code: SafetyNoticeCode
    elapsed_seconds: float
    remaining_seconds: float


@dataclass(frozen=True, slots=True)
class ForegroundSnapshotRequest:
    """One ephemeral resolution request for one completed toggle take."""

    turn_id: int
    ephemeral: bool = True


ForegroundSnapshot: TypeAlias = Any


class ForegroundSnapshotResolver(Protocol):
    def resolve_foreground(
        self, request: ForegroundSnapshotRequest
    ) -> ForegroundSnapshot: ...


@dataclass(frozen=True, slots=True)
class SnapshotCallableAdapter:
    """Adapt a platform resolver's zero-argument ``snapshot`` method.

    For example, ``SnapshotCallableAdapter(windows_resolver.snapshot)`` wires
    the Win32 target adapter into capture without either package importing the
    other.  The request remains available to capture diagnostics but is never
    persisted by this adapter.
    """

    snapshot: Callable[[], ForegroundSnapshot]

    def resolve_foreground(
        self, request: ForegroundSnapshotRequest
    ) -> ForegroundSnapshot:
        del request
        return self.snapshot()


@dataclass(frozen=True, slots=True)
class CapturedAudio:
    """An immutable, recoverable take.

    Chunks remain opaque so virtual tests can use bytes while the live adapter
    can retain numpy blocks without this layer depending on numpy.
    ``record_start_diagnostic`` is never supplied to the finish resolver and
    cannot authorize or veto delivery.
    """

    turn_id: int
    mode: CaptureMode
    chunks: tuple[Any, ...] = field(repr=False)
    started_at: float
    finished_at: float
    ended_by: CaptureEnd
    notices: tuple[SafetyNotice, ...] = ()
    record_start_diagnostic: Any | None = field(default=None, repr=False)


@dataclass(frozen=True, slots=True)
class CaptureCompletion:
    audio: CapturedAudio
    finish_snapshot: ForegroundSnapshot | None = field(default=None, repr=False)
    snapshot_request: ForegroundSnapshotRequest | None = None


@dataclass(frozen=True, slots=True)
class CaptureProgress:
    phase: CapturePhase
    notices: tuple[SafetyNotice, ...] = ()
    completion: CaptureCompletion | None = None


@dataclass(frozen=True, slots=True)
class Transcription:
    text: str = field(repr=False)
    confidence: float | None = None

    def __post_init__(self) -> None:
        if self.confidence is not None and not 0 <= self.confidence <= 1:
            raise ValueError("confidence must be between 0 and 1")


@dataclass(frozen=True, slots=True)
class TranscriptAcceptance:
    disposition: TranscriptDisposition
    text: str = field(repr=False)
    confidence: float | None
    reason: str | None = None

    @property
    def may_deliver(self) -> bool:
        return self.disposition is TranscriptDisposition.ACCEPTED


@dataclass(frozen=True, slots=True)
class CaptureTurnResult:
    completion: CaptureCompletion = field(repr=False)
    transcript: TranscriptAcceptance = field(repr=False)
    boundary_replacement_required: bool = False
    error_code: str | None = None


class Transcriber(Protocol):
    def transcribe(
        self, audio: Any
    ) -> str | Transcription | tuple[str, float | None]: ...


CancellationProbe: TypeAlias = Callable[[], bool]
TranscriberFactory: TypeAlias = Callable[[CancellationProbe], Transcriber]
TranscriptClassifier: TypeAlias = Callable[[str], float | None]
