"""Deterministic capture orchestration with no platform side effects."""

from __future__ import annotations

import queue
import threading
from time import monotonic
from typing import Any, Callable

from talktomeclaude.core.deadlines import DEFAULT_DEADLINES, DeadlineName

from .contracts import (
    CancellationProbe,
    CaptureCancelled,
    CaptureCompletion,
    CaptureContractError,
    CaptureEnd,
    CaptureMode,
    CapturePhase,
    CaptureProgress,
    CaptureSettings,
    CapturedAudio,
    CaptureTurnResult,
    ForegroundSnapshotRequest,
    ForegroundSnapshotResolver,
    SafetyNotice,
    SafetyNoticeCode,
    TranscriptAcceptance,
    TranscriptClassifier,
    TranscriptDisposition,
    TranscriberFactory,
    Transcription,
)


_DEFAULT_STT_CONSTRUCTION_SECONDS = DEFAULT_DEADLINES[
    DeadlineName.STT_CONSTRUCTION
].seconds
_DEFAULT_STT_ITERATION_SECONDS = DEFAULT_DEADLINES[DeadlineName.STT_ITERATION].seconds
_BOUNDARY_POLL_SECONDS = 0.005
_CALL_COMPLETED = "completed"
_CALL_CANCELLED = "cancelled"
_CALL_TIMED_OUT = "timed_out"


class CaptureService:
    """Own one capture turn at a time.

    The service is driven by shell intents and audio callbacks.  It never uses
    an idle/silence timer.  ``poll`` exists only for the explicit safety policy,
    and returns the preserved take if that high ceiling is reached.
    """

    def __init__(
        self,
        *,
        settings: CaptureSettings | None = None,
        clock: Callable[[], float] = monotonic,
        snapshot_resolver: ForegroundSnapshotResolver | None = None,
        record_start_probe: Callable[[], Any] | None = None,
        transcript_classifier: TranscriptClassifier | None = None,
        audio_assembler: Callable[[tuple[Any, ...]], Any] | None = None,
        stt_construction_timeout_seconds: float = _DEFAULT_STT_CONSTRUCTION_SECONDS,
        stt_iteration_timeout_seconds: float = _DEFAULT_STT_ITERATION_SECONDS,
    ) -> None:
        if stt_construction_timeout_seconds <= 0:
            raise ValueError("STT construction timeout must be positive")
        if stt_iteration_timeout_seconds <= 0:
            raise ValueError("STT iteration timeout must be positive")
        self.settings = settings or CaptureSettings()
        self._clock = clock
        self._snapshot_resolver = snapshot_resolver
        self._record_start_probe = record_start_probe
        self._transcript_classifier = transcript_classifier
        self._audio_assembler = audio_assembler or _join_chunks
        self._stt_construction_timeout_seconds = stt_construction_timeout_seconds
        self._stt_iteration_timeout_seconds = stt_iteration_timeout_seconds
        self._lock = threading.RLock()
        self._boundary_replacement_required = False
        self._phase = CapturePhase.IDLE
        self._turn_id = 0
        self._mode = self.settings.mode
        self._started_at = 0.0
        self._chunks: list[Any] = []
        self._notices: list[SafetyNotice] = []
        self._warning_emitted = False
        self._record_start_diagnostic: Any | None = None
        self._preserved: CaptureCompletion | None = None

    @property
    def phase(self) -> CapturePhase:
        return self._phase

    @property
    def turn_id(self) -> int:
        return self._turn_id

    @property
    def boundary_replacement_required(self) -> bool:
        with self._lock:
            return self._boundary_replacement_required

    def start(self, mode: CaptureMode | None = None) -> int:
        with self._lock:
            if self._phase is not CapturePhase.IDLE:
                raise CaptureContractError(
                    "capture is already active or awaiting recovery"
                )
            self._turn_id += 1
            self._mode = mode or self.settings.mode
            self._started_at = self._clock()
            self._chunks = []
            self._notices = []
            self._warning_emitted = False
            self._record_start_diagnostic = (
                self._record_start_probe()
                if self._record_start_probe is not None
                else None
            )
            self._preserved = None
            self._phase = CapturePhase.RECORDING
            return self._turn_id

    def add_audio(self, chunk: Any) -> None:
        with self._lock:
            if self._phase is not CapturePhase.RECORDING:
                raise CaptureContractError("audio is accepted only while recording")
            self._chunks.append(chunk)

    def toggle(self) -> int | CaptureCompletion:
        """Start an idle toggle turn or finish the current toggle turn."""

        with self._lock:
            if self._phase is CapturePhase.IDLE:
                return self.start(CaptureMode.PUSH_TOGGLE)
            if self._phase is CapturePhase.PRESERVED:
                raise CaptureContractError(
                    "recover the preserved safety-ceiling take first"
                )
            if self._mode is not CaptureMode.PUSH_TOGGLE:
                raise CaptureContractError(
                    "toggle finish is valid only in push-toggle mode"
                )
            return self._finish(CaptureEnd.FINISH_TOGGLE, request_snapshot=True)

    def release(self) -> CaptureCompletion:
        with self._lock:
            if self._phase is not CapturePhase.RECORDING:
                raise CaptureContractError("no active hold-to-talk capture")
            if self._mode is not CaptureMode.HOLD_TO_TALK:
                raise CaptureContractError(
                    "key release is valid only in hold-to-talk mode"
                )
            return self._finish(CaptureEnd.KEY_RELEASE, request_snapshot=True)

    def cancel(self) -> CaptureCompletion:
        with self._lock:
            if self._phase is not CapturePhase.RECORDING:
                raise CaptureContractError("no active capture to cancel")
            return self._finish(CaptureEnd.CANCELLED, request_snapshot=False)

    def poll(self) -> CaptureProgress:
        """Emit warning/ceiling events without inspecting audio or silence."""

        with self._lock:
            return self._poll_locked()

    def _poll_locked(self) -> CaptureProgress:
        if self._phase is CapturePhase.PRESERVED:
            # The completion is emitted only on the transition to PRESERVED.
            # Consume/discard are the explicit, identity-checked recovery paths.
            return CaptureProgress(self._phase)
        if self._phase is not CapturePhase.RECORDING:
            return CaptureProgress(self._phase)

        now = self._clock()
        elapsed = max(0.0, now - self._started_at)
        emitted: list[SafetyNotice] = []
        warning_at = (
            self.settings.safety_ceiling_seconds - self.settings.warning_before_seconds
        )
        if not self._warning_emitted and elapsed >= warning_at:
            notice = SafetyNotice(
                SafetyNoticeCode.CEILING_APPROACHING,
                elapsed_seconds=elapsed,
                remaining_seconds=max(
                    0.0, self.settings.safety_ceiling_seconds - elapsed
                ),
            )
            self._warning_emitted = True
            self._notices.append(notice)
            emitted.append(notice)
        if elapsed < self.settings.safety_ceiling_seconds:
            return CaptureProgress(self._phase, tuple(emitted))

        reached = SafetyNotice(
            SafetyNoticeCode.CEILING_REACHED,
            elapsed_seconds=elapsed,
            remaining_seconds=0.0,
        )
        self._notices.append(reached)
        emitted.append(reached)
        completion = self._finish(
            CaptureEnd.SAFETY_CEILING,
            request_snapshot=False,
            finished_at=now,
        )
        self._preserved = completion
        self._phase = CapturePhase.PRESERVED
        return CaptureProgress(self._phase, tuple(emitted), completion)

    def consume_preserved(
        self, expected: CaptureCompletion
    ) -> CaptureCompletion:
        """Consume the exact emitted completion, retaining it for recovery."""

        with self._lock:
            self._verify_preserved_identity(expected)
            completion = self._preserved
            assert completion is not None
            self._preserved = None
            self._phase = CapturePhase.IDLE
            return completion

    def discard_preserved(self, expected: CaptureCompletion) -> int:
        """Clear the exact emitted completion without returning retained audio."""

        with self._lock:
            self._verify_preserved_identity(expected)
            turn_id = expected.audio.turn_id
            self._preserved = None
            self._phase = CapturePhase.IDLE
            return turn_id

    def _verify_preserved_identity(self, expected: CaptureCompletion) -> None:
        if self._phase is not CapturePhase.PRESERVED or self._preserved is None:
            raise CaptureContractError("there is no preserved take")
        if self._preserved is not expected:
            raise CaptureContractError("preserved completion identity does not match")

    def transcribe(
        self,
        completion: CaptureCompletion,
        factory: TranscriberFactory,
        *,
        cancelled: CancellationProbe = lambda: False,
    ) -> CaptureTurnResult:
        """Run STT behind bounded daemon isolation.

        Python cannot terminate a hung thread.  A timeout/cancel therefore
        taints this service instance permanently, discards any late queue
        result, and tells the coordinator to replace the isolation boundary.
        """

        if self.boundary_replacement_required:
            return _boundary_tainted_result(completion)
        if self._is_stale(completion):
            return CaptureTurnResult(completion, _stale_acceptance())
        if completion.audio.ended_by is CaptureEnd.CANCELLED or cancelled():
            return _cancelled_result(completion)

        # Model construction may be slow and is not safely interruptible.  The
        # checks on both sides ensure a stop requested during construction is
        # honored before transcription begins.
        construction_status, engine, construction_error = _run_bounded(
            lambda: factory(cancelled),
            timeout_seconds=self._stt_construction_timeout_seconds,
            cancelled=cancelled,
        )
        if construction_status == _CALL_TIMED_OUT:
            self._taint_boundary()
            return _timeout_result(
                completion,
                TranscriptDisposition.CONSTRUCTION_TIMEOUT,
                "stt_construction_timeout",
            )
        if construction_status == _CALL_CANCELLED:
            self._taint_boundary()
            return _cancelled_result(completion, boundary_replacement_required=True)
        if isinstance(construction_error, CaptureCancelled):
            return _cancelled_result(completion)
        if construction_error is not None:
            raise construction_error
        if engine is None:
            raise CaptureContractError("transcriber construction returned no engine")
        if self._is_stale(completion):
            return CaptureTurnResult(completion, _stale_acceptance())
        if cancelled():
            self._taint_boundary()
            return _cancelled_result(completion, boundary_replacement_required=True)
        iteration_status, raw, iteration_error = _run_bounded(
            lambda: engine.transcribe(
                self._audio_assembler(completion.audio.chunks)
            ),
            timeout_seconds=self._stt_iteration_timeout_seconds,
            cancelled=cancelled,
        )
        if iteration_status == _CALL_TIMED_OUT:
            self._taint_boundary()
            return _timeout_result(
                completion,
                TranscriptDisposition.ITERATION_TIMEOUT,
                "stt_iteration_timeout",
            )
        if iteration_status == _CALL_CANCELLED:
            self._taint_boundary()
            return _cancelled_result(completion, boundary_replacement_required=True)
        if isinstance(iteration_error, CaptureCancelled):
            return _cancelled_result(completion)
        if iteration_error is not None:
            raise iteration_error
        if raw is None:
            raise CaptureContractError("transcriber returned no result")
        if cancelled():
            self._taint_boundary()
            return _cancelled_result(completion, boundary_replacement_required=True)
        transcription = _normalize_transcription(raw)
        if self._is_stale(completion):
            return CaptureTurnResult(
                completion,
                _stale_acceptance(transcription.text, transcription.confidence),
            )
        if self._transcript_classifier is not None:
            # This classifier is capture-specific.  Command-intent confidence
            # is unrelated and is never imported or reused here.
            transcription = Transcription(
                transcription.text,
                self._transcript_classifier(transcription.text),
            )
            if cancelled():
                self._taint_boundary()
                return _cancelled_result(
                    completion, boundary_replacement_required=True
                )
            if self._is_stale(completion):
                return CaptureTurnResult(
                    completion,
                    _stale_acceptance(
                        transcription.text, transcription.confidence
                    ),
                )
        acceptance = assess_transcript(
            transcription,
            minimum_confidence=self.settings.minimum_confidence,
        )
        return CaptureTurnResult(completion, acceptance)

    def _taint_boundary(self) -> None:
        with self._lock:
            self._boundary_replacement_required = True

    def _is_stale(self, completion: CaptureCompletion) -> bool:
        with self._lock:
            return completion.audio.turn_id != self._turn_id

    def _finish(
        self,
        ended_by: CaptureEnd,
        *,
        request_snapshot: bool,
        finished_at: float | None = None,
    ) -> CaptureCompletion:
        if self._phase is not CapturePhase.RECORDING:
            raise CaptureContractError("capture is not recording")
        finished = self._clock() if finished_at is None else finished_at
        audio = CapturedAudio(
            turn_id=self._turn_id,
            mode=self._mode,
            chunks=tuple(self._chunks),
            started_at=self._started_at,
            finished_at=finished,
            ended_by=ended_by,
            notices=tuple(self._notices),
            record_start_diagnostic=self._record_start_diagnostic,
        )
        request = None
        snapshot = None
        if request_snapshot:
            request = ForegroundSnapshotRequest(self._turn_id)
            if self._snapshot_resolver is not None:
                # This is the sole foreground-resolution call in capture.  The
                # returned evidence remains opaque and ephemeral.
                snapshot = self._snapshot_resolver.resolve_foreground(request)
        self._chunks = []
        self._notices = []
        self._record_start_diagnostic = None
        self._phase = CapturePhase.IDLE
        return CaptureCompletion(audio, snapshot, request)


def assess_transcript(
    transcription: Transcription,
    *,
    minimum_confidence: float,
) -> TranscriptAcceptance:
    """Return a structured gate; rejected text is retained verbatim."""

    if not 0 <= minimum_confidence <= 1:
        raise ValueError("minimum confidence must be between 0 and 1")
    if not transcription.text.strip():
        return TranscriptAcceptance(
            TranscriptDisposition.EMPTY,
            transcription.text,
            transcription.confidence,
            "transcript is empty",
        )
    if (
        transcription.confidence is not None
        and transcription.confidence < minimum_confidence
    ):
        return TranscriptAcceptance(
            TranscriptDisposition.LOW_CONFIDENCE,
            transcription.text,
            transcription.confidence,
            "transcript confidence is below the configured threshold",
        )
    return TranscriptAcceptance(
        TranscriptDisposition.ACCEPTED,
        transcription.text,
        transcription.confidence,
    )


def _cancelled_acceptance() -> TranscriptAcceptance:
    return TranscriptAcceptance(
        TranscriptDisposition.CANCELLED,
        "",
        None,
        "capture or transcription was cancelled",
    )


def _cancelled_result(
    completion: CaptureCompletion,
    *,
    boundary_replacement_required: bool = False,
) -> CaptureTurnResult:
    """Discard late content and tell the owner when isolation is tainted."""

    return CaptureTurnResult(
        completion,
        _cancelled_acceptance(),
        boundary_replacement_required=boundary_replacement_required,
    )


def _timeout_result(
    completion: CaptureCompletion,
    disposition: TranscriptDisposition,
    error_code: str,
) -> CaptureTurnResult:
    return CaptureTurnResult(
        completion,
        TranscriptAcceptance(
            disposition,
            "",
            None,
            "transcription deadline expired",
        ),
        boundary_replacement_required=True,
        error_code=error_code,
    )


def _boundary_tainted_result(completion: CaptureCompletion) -> CaptureTurnResult:
    return CaptureTurnResult(
        completion,
        TranscriptAcceptance(
            TranscriptDisposition.BOUNDARY_TAINTED,
            "",
            None,
            "STT isolation boundary must be replaced before retry",
        ),
        boundary_replacement_required=True,
        error_code="stt_boundary_tainted",
    )


def _run_bounded(
    callback: Callable[[], Any],
    *,
    timeout_seconds: float,
    cancelled: CancellationProbe,
) -> tuple[str, Any | None, BaseException | None]:
    """Run one opaque operation without letting it block the coordinator."""

    result_queue: queue.Queue[tuple[Any | None, BaseException | None]] = queue.Queue(
        maxsize=1
    )

    def run() -> None:
        try:
            result_queue.put((callback(), None))
        except BaseException as exc:
            result_queue.put((None, exc))

    thread = threading.Thread(target=run, name="capture-stt-boundary", daemon=True)
    thread.start()
    deadline = monotonic() + timeout_seconds
    while True:
        if cancelled():
            return _CALL_CANCELLED, None, None
        remaining = deadline - monotonic()
        if remaining <= 0:
            return _CALL_TIMED_OUT, None, None
        try:
            value, error = result_queue.get(
                timeout=min(_BOUNDARY_POLL_SECONDS, remaining)
            )
        except queue.Empty:
            continue
        return _CALL_COMPLETED, value, error


def _stale_acceptance(
    text: str = "", confidence: float | None = None
) -> TranscriptAcceptance:
    return TranscriptAcceptance(
        TranscriptDisposition.STALE_GENERATION,
        text,
        confidence,
        "capture result belongs to a stale turn",
    )


def _normalize_transcription(
    raw: str | Transcription | tuple[str, float | None],
) -> Transcription:
    if isinstance(raw, Transcription):
        return raw
    if isinstance(raw, str):
        return Transcription(raw)
    text, confidence = raw
    return Transcription(text, confidence)


def _join_chunks(chunks: tuple[Any, ...]) -> Any:
    """Keep one chunk lossless; combine bytes; otherwise preserve the tuple."""

    if len(chunks) == 1:
        return chunks[0]
    if all(isinstance(chunk, bytes) for chunk in chunks):
        return b"".join(chunks)
    return chunks
