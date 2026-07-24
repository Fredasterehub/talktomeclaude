"""One-turn capture-to-delivery orchestration for the Windows companion.

The coordinator intentionally retains no capture completion or native target
evidence.  A normal finish-toggle owns one transaction; confirmation and
recovery always resolve a new foreground target.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import threading
from typing import Any, Protocol

from talktomeclaude.capture import (
    CaptureCompletion,
    CaptureEnd,
    CaptureMode,
    CapturePhase,
    CaptureService,
    CaptureTurnResult,
    TranscriptAcceptance,
    TranscriptDisposition,
)
from talktomeclaude.capture.contracts import CancellationProbe, TranscriberFactory
from talktomeclaude.core import (
    EventKind,
    RuntimeCoordinator,
    RuntimeEvent,
    RuntimePhase,
)
from talktomeclaude.platform.contracts import (
    DeliveryCode,
    DeliveryMode,
    DeliveryResult,
)


class CaptureDeliveryCode(str, Enum):
    DELIVERED = "delivered"
    REVIEW_REQUIRED = "review_required"
    CANCELLED = "cancelled"
    STALE = "stale"
    TRANSCRIPTION_FAILED = "transcription_failed"
    DELIVERY_FAILED = "delivery_failed"


class TextDelivery(Protocol):
    """Narrow boundary implemented by the Windows text injector."""

    def snapshot_target(self) -> Any: ...

    def deliver(
        self,
        text: str,
        evidence: Any,
        *,
        mode: DeliveryMode,
        auto_submit: bool,
        cancelled: CancellationProbe = lambda: False,
    ) -> DeliveryResult: ...


@dataclass(frozen=True, slots=True)
class TranscriptReview:
    """Editable transcript payload whose representation is content-safe."""

    disposition: TranscriptDisposition
    text: str = field(repr=False)
    confidence: float | None = None
    reason: str | None = None

    @classmethod
    def from_acceptance(
        cls, acceptance: TranscriptAcceptance
    ) -> TranscriptReview:
        return cls(
            acceptance.disposition,
            acceptance.text,
            acceptance.confidence,
            acceptance.reason,
        )


@dataclass(frozen=True, slots=True)
class CaptureDeliveryResult:
    """Content-bearing UI result with no capture or native target evidence."""

    code: CaptureDeliveryCode
    transcript: TranscriptReview | None
    delivery: DeliveryResult | None
    runtime_phase: RuntimePhase
    fresh_snapshot_required: bool = False
    boundary_replacement_required: bool = False
    error_code: str | None = None
    transcript_disposition: TranscriptDisposition | None = None

    @property
    def succeeded(self) -> bool:
        return self.code is CaptureDeliveryCode.DELIVERED

    @property
    def diagnostics(self) -> dict[str, object]:
        """Return explicitly content-free fields safe for logs/telemetry."""

        return {
            "code": self.code.value,
            "transcript_disposition": (
                self.transcript.disposition.value
                if self.transcript
                else (
                    self.transcript_disposition.value
                    if self.transcript_disposition
                    else None
                )
            ),
            "delivery_code": (
                self.delivery.code.value if self.delivery else None
            ),
            "runtime_phase": self.runtime_phase.value,
            "fresh_snapshot_required": self.fresh_snapshot_required,
            "boundary_replacement_required": self.boundary_replacement_required,
            "error_code": self.error_code,
        }


class CaptureDeliveryCoordinator:
    """Join capture, transcript admission, and one fail-closed delivery.

    The active completion and its finish-time evidence are method locals only.
    Neither is stored on this object or returned in :class:`CaptureDeliveryResult`.
    """

    def __init__(
        self,
        capture: CaptureService,
        injector: TextDelivery,
        runtime: RuntimeCoordinator | None = None,
    ) -> None:
        self._capture = capture
        self._injector = injector
        self._runtime = runtime or RuntimeCoordinator()
        self._delivery_lock = threading.RLock()

    @property
    def runtime(self) -> RuntimeCoordinator:
        return self._runtime

    def start(self, mode: CaptureMode | None = None) -> int:
        """Start one capture generation and expose recording state."""

        transition = self._runtime.dispatch(
            RuntimeEvent(EventKind.START_RECORDING)
        )
        if not transition.accepted:
            raise RuntimeError(
                f"capture cannot start while {transition.current.phase.value}"
            )
        try:
            return self._capture.start(mode)
        except BaseException:
            self._runtime.dispatch(RuntimeEvent(EventKind.CANCEL))
            raise

    def add_audio(self, chunk: Any) -> None:
        self._capture.add_audio(chunk)

    def cancel(self) -> None:
        """Cancel owned capture state and discard its opaque audio chunks."""

        # Serialize with the complete synchronous injection transaction.  If a
        # Win32 SendInput call was already admitted, cancellation returns only
        # after that call and all later cancellation checks have settled; no
        # clipboard/key side effect can occur after this method returns.
        with self._delivery_lock:
            if self._capture.phase is CapturePhase.RECORDING:
                self._capture.cancel()
            self._runtime.dispatch(RuntimeEvent(EventKind.CANCEL))

    def finish_toggle(
        self,
        factory: TranscriberFactory,
        *,
        mode: DeliveryMode,
        auto_submit: bool,
        cancelled: CancellationProbe = lambda: False,
    ) -> CaptureDeliveryResult:
        """Snapshot at toggle completion, then transcribe and maybe deliver."""

        completion = self.begin_finish_toggle()
        return self.process_completion(
            completion,
            factory,
            mode=mode,
            auto_submit=auto_submit,
            cancelled=cancelled,
            runtime_transitioned=True,
        )

    def begin_finish_toggle(self) -> CaptureCompletion:
        """Close toggle capture and expose TRANSCRIBING before slow STT work.

        Graphical callers use this split boundary so the finish intent can
        update its semantic state synchronously, then run
        :meth:`process_completion` on a worker without blocking Tk.
        """

        completion = self._capture.toggle()
        if not isinstance(completion, CaptureCompletion):
            raise RuntimeError("finish-toggle started a new capture unexpectedly")
        self._begin_completion_transition()
        return completion

    def release_hold(
        self,
        factory: TranscriberFactory,
        *,
        mode: DeliveryMode,
        auto_submit: bool,
        cancelled: CancellationProbe = lambda: False,
    ) -> CaptureDeliveryResult:
        """Finish hold-to-talk at key release and process its snapshot."""

        completion = self.begin_release_hold()
        return self.process_completion(
            completion,
            factory,
            mode=mode,
            auto_submit=auto_submit,
            cancelled=cancelled,
            runtime_transitioned=True,
        )

    def begin_release_hold(self) -> CaptureCompletion:
        """Close hold-to-talk capture at key release before slow STT work."""

        completion = self._capture.release()
        self._begin_completion_transition()
        return completion

    def process_completion(
        self,
        completion: CaptureCompletion,
        factory: TranscriberFactory,
        *,
        mode: DeliveryMode,
        auto_submit: bool,
        cancelled: CancellationProbe = lambda: False,
        runtime_transitioned: bool = False,
    ) -> CaptureDeliveryResult:
        """Process an already-finished take without retaining its evidence."""

        # Reject an old completion before changing the current generation.  If
        # a newer capture is recording, both owners remain in RECORDING.
        if completion.audio.turn_id != self._capture.turn_id:
            if self._capture.phase is not CapturePhase.RECORDING:
                self._runtime.dispatch(RuntimeEvent(EventKind.CANCEL))
            return CaptureDeliveryResult(
                CaptureDeliveryCode.STALE,
                None,
                None,
                self._runtime.state.phase,
                fresh_snapshot_required=True,
                transcript_disposition=TranscriptDisposition.STALE_GENERATION,
            )

        # Poll leaves a safety-ceiling take in PRESERVED.  Consume that owned
        # reference before transcription so every exit releases audio/evidence
        # and a subsequent explicit recording can start.
        if (
            completion.audio.ended_by is CaptureEnd.SAFETY_CEILING
            and self._capture.phase is CapturePhase.PRESERVED
        ):
            completion = self._capture.consume_preserved(completion)

        if runtime_transitioned:
            if self._runtime.state.phase is not RuntimePhase.TRANSCRIBING:
                raise RuntimeError("capture completion does not own transcribing state")
        else:
            self._begin_completion_transition()

        try:
            turn = self._capture.transcribe(
                completion,
                factory,
                cancelled=cancelled,
            )
        except Exception as exc:
            exception_code = type(exc).__name__
            self._runtime.dispatch(
                RuntimeEvent(
                    EventKind.ERROR_OCCURRED,
                    error_code=exception_code,
                    recover_to=RuntimePhase.AWAITING_CONFIRMATION,
                )
            )
            return CaptureDeliveryResult(
                CaptureDeliveryCode.TRANSCRIPTION_FAILED,
                None,
                None,
                self._runtime.state.phase,
                fresh_snapshot_required=True,
                error_code=exception_code,
            )

        raw_error_code = getattr(turn, "error_code", None)
        error_code = raw_error_code if isinstance(raw_error_code, str) else None
        if error_code:
            self._runtime.dispatch(
                RuntimeEvent(
                    EventKind.ERROR_OCCURRED,
                    error_code=error_code,
                    recover_to=RuntimePhase.AWAITING_CONFIRMATION,
                )
            )
            return CaptureDeliveryResult(
                CaptureDeliveryCode.TRANSCRIPTION_FAILED,
                None,
                None,
                self._runtime.state.phase,
                fresh_snapshot_required=True,
                boundary_replacement_required=(
                    turn.boundary_replacement_required
                ),
                error_code=error_code,
                transcript_disposition=turn.transcript.disposition,
            )

        disposition = turn.transcript.disposition
        # A safety stop preserves usable content for an explicit recovery
        # action.  It never turns an elapsed timer into permission to inject,
        # but cancellation/stale outcomes still take their normal discard path.
        if (
            completion.audio.ended_by is CaptureEnd.SAFETY_CEILING
            and disposition
            in {
                TranscriptDisposition.ACCEPTED,
                TranscriptDisposition.EMPTY,
                TranscriptDisposition.LOW_CONFIDENCE,
            }
        ):
            return self._review(turn)

        if disposition is TranscriptDisposition.ACCEPTED:
            return self._deliver_turn(
                turn,
                mode=mode,
                auto_submit=auto_submit,
                cancelled=cancelled,
            )
        if disposition in {
            TranscriptDisposition.EMPTY,
            TranscriptDisposition.LOW_CONFIDENCE,
        }:
            return self._review(turn)

        self._runtime.dispatch(RuntimeEvent(EventKind.CANCEL))
        code = (
            CaptureDeliveryCode.STALE
            if disposition is TranscriptDisposition.STALE_GENERATION
            else CaptureDeliveryCode.CANCELLED
        )
        return CaptureDeliveryResult(
            code,
            None,
            None,
            self._runtime.state.phase,
            fresh_snapshot_required=True,
            boundary_replacement_required=turn.boundary_replacement_required,
            transcript_disposition=disposition,
        )

    def _begin_completion_transition(self) -> None:
        transition = self._runtime.dispatch(RuntimeEvent(EventKind.FINISH_RECORDING))
        if not transition.accepted:
            raise RuntimeError(
                f"capture cannot finish while {transition.current.phase.value}"
            )

    def confirm_or_recover(
        self,
        transcript: TranscriptReview,
        *,
        mode: DeliveryMode,
        auto_submit: bool,
        cancelled: CancellationProbe = lambda: False,
    ) -> CaptureDeliveryResult:
        """Deliver reviewed text using new confirm-time target evidence.

        The caller must return the visible/editable transcript.  The coordinator
        does not cache it, and this method never reuses the earlier completion.
        """

        if not transcript.text.strip():
            return CaptureDeliveryResult(
                CaptureDeliveryCode.REVIEW_REQUIRED,
                transcript,
                None,
                self._runtime.state.phase,
                fresh_snapshot_required=True,
            )

        if self._runtime.state.phase is RuntimePhase.RECOVERABLE_ERROR:
            recovered = self._runtime.dispatch(RuntimeEvent(EventKind.RETRY))
            if not recovered.accepted:
                raise RuntimeError("capture recovery state could not be resumed")
        confirmed = self._runtime.dispatch(
            RuntimeEvent(EventKind.CONFIRM_TRANSCRIPT)
        )
        if not confirmed.accepted:
            raise RuntimeError(
                f"transcript cannot be confirmed while {confirmed.current.phase.value}"
            )

        if cancelled():
            return CaptureDeliveryResult(
                CaptureDeliveryCode.CANCELLED,
                None,
                None,
                self._runtime.state.phase,
                fresh_snapshot_required=True,
            )
        resolution = self._injector.snapshot_target()
        evidence = _resolution_evidence(resolution)
        return self._deliver(
            transcript,
            evidence,
            mode=mode,
            auto_submit=auto_submit,
            cancelled=cancelled,
        )

    def _review(self, turn: CaptureTurnResult) -> CaptureDeliveryResult:
        transition = self._runtime.dispatch(
            RuntimeEvent(EventKind.TRANSCRIPT_REVIEW_REQUIRED)
        )
        if not transition.accepted:
            raise RuntimeError("runtime rejected transcript review state")
        return CaptureDeliveryResult(
            CaptureDeliveryCode.REVIEW_REQUIRED,
            _review_payload(turn.transcript),
            None,
            self._runtime.state.phase,
            fresh_snapshot_required=True,
            boundary_replacement_required=turn.boundary_replacement_required,
            transcript_disposition=turn.transcript.disposition,
        )

    def _deliver_turn(
        self,
        turn: CaptureTurnResult,
        *,
        mode: DeliveryMode,
        auto_submit: bool,
        cancelled: CancellationProbe,
    ) -> CaptureDeliveryResult:
        transition = self._runtime.dispatch(
            RuntimeEvent(EventKind.TRANSCRIPT_ACCEPTED)
        )
        if not transition.accepted:
            raise RuntimeError("runtime rejected accepted transcript")

        resolution = turn.completion.finish_snapshot
        evidence = _resolution_evidence(resolution)
        return self._deliver(
            turn.transcript,
            evidence,
            mode=mode,
            auto_submit=auto_submit,
            cancelled=cancelled,
        )

    def _deliver(
        self,
        transcript: TranscriptAcceptance | TranscriptReview,
        evidence: Any,
        *,
        mode: DeliveryMode,
        auto_submit: bool,
        cancelled: CancellationProbe = lambda: False,
    ) -> CaptureDeliveryResult:
        with self._delivery_lock:
            return self._deliver_serialized(
                transcript,
                evidence,
                mode=mode,
                auto_submit=auto_submit,
                cancelled=cancelled,
            )

    def _deliver_serialized(
        self,
        transcript: TranscriptAcceptance | TranscriptReview,
        evidence: Any,
        *,
        mode: DeliveryMode,
        auto_submit: bool,
        cancelled: CancellationProbe,
    ) -> CaptureDeliveryResult:
        # Invalid initial resolution is a coordinator-level fail-closed path:
        # do not even enter the platform mutation transaction.
        if cancelled():
            delivery = DeliveryResult(DeliveryCode.CANCELLED)
        elif evidence is None:
            delivery = DeliveryResult(DeliveryCode.INVALID_TARGET)
        else:
            delivery = self._injector.deliver(
                transcript.text,
                evidence,
                mode=mode,
                auto_submit=auto_submit,
                cancelled=cancelled,
            )

        if delivery.succeeded:
            event_kind = (
                EventKind.DICTATION_DELIVERED
                if mode is DeliveryMode.GENERIC
                else EventKind.DELIVERY_SUCCEEDED
            )
            transition = self._runtime.dispatch(
                RuntimeEvent(event_kind)
            )
            if not transition.accepted:
                raise RuntimeError("runtime rejected successful delivery")
            return CaptureDeliveryResult(
                CaptureDeliveryCode.DELIVERED,
                None,
                delivery,
                self._runtime.state.phase,
            )

        if delivery.code is DeliveryCode.CANCELLED:
            if self._runtime.state.phase is not RuntimePhase.STOPPING:
                self._runtime.dispatch(RuntimeEvent(EventKind.CANCEL))
            return CaptureDeliveryResult(
                CaptureDeliveryCode.CANCELLED,
                None,
                delivery,
                self._runtime.state.phase,
                fresh_snapshot_required=True,
            )

        self._runtime.dispatch(
            RuntimeEvent(
                EventKind.ERROR_OCCURRED,
                error_code=delivery.code.value,
                recover_to=RuntimePhase.AWAITING_CONFIRMATION,
            )
        )
        return CaptureDeliveryResult(
            CaptureDeliveryCode.DELIVERY_FAILED,
            _review_payload(transcript),
            delivery,
            self._runtime.state.phase,
            fresh_snapshot_required=True,
            error_code=delivery.code.value,
        )


def _resolution_evidence(resolution: Any) -> Any | None:
    """Unwrap only a valid platform resolution, preserving identity."""

    if resolution is None:
        return None
    if hasattr(resolution, "code"):
        code = getattr(resolution, "code")
        value = getattr(code, "value", code)
        if str(value).casefold() != "valid":
            return None
    return getattr(resolution, "evidence", resolution)


def _review_payload(
    transcript: TranscriptAcceptance | TranscriptReview,
) -> TranscriptReview:
    if isinstance(transcript, TranscriptReview):
        return transcript
    return TranscriptReview.from_acceptance(transcript)
