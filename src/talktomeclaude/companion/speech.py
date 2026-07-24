"""Mute-aware production composition for canonical companion speech."""

from __future__ import annotations

import threading
from collections.abc import Callable
from pathlib import Path
from typing import Protocol

from talktomeclaude.reply import ReplyEvent
from talktomeclaude.speech import (
    CanonicalSpeechController,
    OralSessionStore,
    OralStatus,
    PersistentSpeechRuntime,
    SoundDevicePlayback,
    SpeechPipeline,
    SynthesisWorker,
    production_synthesis_worker,
)


class _OfferQueue(Protocol):
    def offer(
        self,
        unit_id: str,
        text: str,
        *,
        effect_id: str | None = None,
    ) -> bool: ...

    def stop(self) -> object: ...


class _RuntimeOwner(Protocol):
    def shutdown(self) -> bool: ...


class _MuteGate:
    """Freeze muted answers while refusing every playback admission."""

    def __init__(self, pipeline: _OfferQueue, *, muted: bool) -> None:
        self._pipeline = pipeline
        self._muted = muted
        self._lock = threading.Lock()

    @property
    def muted(self) -> bool:
        with self._lock:
            return self._muted

    def set_muted(self, muted: bool) -> None:
        with self._lock:
            self._muted = muted

    def offer(
        self,
        unit_id: str,
        text: str,
        *,
        effect_id: str | None = None,
    ) -> bool:
        with self._lock:
            if self._muted:
                return False
        return self._pipeline.offer(unit_id, text, effect_id=effect_id)

    def stop(self) -> object:
        return self._pipeline.stop()


class CompanionSpeech:
    """Own one selected voice and bridge whole-answer completion to runtime."""

    def __init__(
        self,
        controller: CanonicalSpeechController,
        session: OralSessionStore,
        gate: _MuteGate,
        runtime: _RuntimeOwner,
        *,
        on_answer_finished: Callable[[], None],
    ) -> None:
        self._controller = controller
        self._session = session
        self._gate = gate
        self._runtime = runtime
        self._on_answer_finished = on_answer_finished
        self._active_answer_id: str | None = None
        self._lock = threading.RLock()
        self._closed = False

    @classmethod
    def create(
        cls,
        selected_voice: str,
        session_path: str | Path,
        *,
        on_answer_finished: Callable[[], None],
        initially_muted: bool = False,
        worker_factory: Callable[[str], SynthesisWorker] = production_synthesis_worker,
        playback: SoundDevicePlayback | None = None,
    ) -> "CompanionSpeech":
        """Build the callback cycle without exposing partially initialized owners."""

        runtime = PersistentSpeechRuntime(selected_voice, worker_factory)
        session = OralSessionStore(session_path)
        holder: dict[str, CompanionSpeech] = {}
        pipeline = SpeechPipeline(
            runtime,
            playback or SoundDevicePlayback(),
            on_unit_completed=lambda unit_id: holder["owner"]._unit_completed(unit_id),
        )
        gate = _MuteGate(pipeline, muted=initially_muted)
        controller = CanonicalSpeechController(session, gate)
        owner = cls(
            controller,
            session,
            gate,
            runtime,
            on_answer_finished=on_answer_finished,
        )
        holder["owner"] = owner
        return owner

    @property
    def muted(self) -> bool:
        return self._gate.muted

    def accept(self, event: ReplyEvent) -> None:
        with self._lock:
            if self._closed:
                raise RuntimeError("speech presentation is closed")
            self._controller.accept(event)
            self._active_answer_id = event.event_id

    def set_muted(self, muted: bool) -> None:
        with self._lock:
            if self._closed:
                return
            if muted:
                self._gate.stop()
            self._gate.set_muted(muted)

    def continue_explicitly(self) -> None:
        """Resume only after a direct user action; unmute never auto-resumes."""

        with self._lock:
            if self._closed or self._gate.muted:
                return
            self._controller.continue_explicitly()

    def interrupt(self) -> None:
        with self._lock:
            if self._closed:
                return
            try:
                self._controller.interrupt()
            except RuntimeError:
                self._gate.stop()
            self._active_answer_id = None

    def stop(self) -> None:
        with self._lock:
            if not self._closed:
                self._gate.stop()

    def _unit_completed(self, unit_id: str) -> bool:
        with self._lock:
            if self._closed:
                return False
            answer_id = self._active_answer_id
            if answer_id is None or not self._controller.unit_completed(unit_id):
                return False
            state = self._session.restore(answer_id)
            finished = state is not None and state.status is OralStatus.COMPLETE
            if finished:
                self._active_answer_id = None
        if finished:
            self._on_answer_finished()
        return True

    def shutdown(self) -> bool:
        with self._lock:
            if self._closed:
                return True
            self._closed = True
            self._gate.stop()
        return self._runtime.shutdown()


__all__ = ["CompanionSpeech"]
