"""Integrated G7 lifecycle proof across the production controller seams."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from typing import Any

from talktomeclaude.capture import CaptureService, SnapshotCallableAdapter
from talktomeclaude.companion.app import DesktopCompanionApplication
from talktomeclaude.companion.capture_delivery import CaptureDeliveryCoordinator
from talktomeclaude.companion.contracts import (
    CompanionIntent,
    CompanionSnapshot,
    IntentKind,
)
from talktomeclaude.companion.runtime import CompanionController
from talktomeclaude.core import RuntimePhase
from talktomeclaude.diagnostics import DiagnosticStore
from talktomeclaude.platform.contracts import DeliveryCode, DeliveryResult
from talktomeclaude.reply import ReplyEvent


class _Resolution:
    code = "valid"
    evidence = object()


class _Injector:
    def __init__(self) -> None:
        self.texts: list[str] = []

    def snapshot_target(self) -> _Resolution:
        return _Resolution()

    def deliver(
        self,
        text: str,
        _evidence: object,
        *,
        mode: object,
        auto_submit: bool,
        cancelled: object = None,
    ) -> DeliveryResult:
        del mode, auto_submit, cancelled
        self.texts.append(text)
        return DeliveryResult(DeliveryCode.DELIVERED, pasted=True, submitted=True)


class _Microphone:
    def __init__(self, sink: CaptureDeliveryCoordinator) -> None:
        self._sink = sink
        self.closed = False

    def start(self) -> None:
        self._sink.add_audio(b"opaque-audio")

    def stop(self) -> None:
        return None

    def close(self) -> None:
        self.closed = True


class _Transcriber:
    def transcribe(self, _audio: object) -> tuple[str, float]:
        return "private operator turn", 0.99


class _Speech:
    def __init__(self) -> None:
        self.events: list[ReplyEvent] = []
        self.shutdowns = 0

    def accept(self, event: ReplyEvent) -> None:
        self.events.append(event)

    def set_muted(self, _muted: bool) -> None:
        return None

    def interrupt(self) -> None:
        return None

    def stop(self) -> None:
        return None

    def shutdown(self) -> bool:
        self.shutdowns += 1
        return True


class _Inbox:
    def __init__(self) -> None:
        self.reply: Any = None
        self.stops = 0

    def start(self, reply: Any, _status: Any) -> None:
        self.reply = reply

    def stop(self) -> bool:
        self.stops += 1
        return True


class _Shell:
    def __init__(self, run_turn: Any) -> None:
        self._run_turn = run_turn
        self.snapshots: list[CompanionSnapshot] = []
        self.closed = False

    def publish(self, snapshot: CompanionSnapshot) -> None:
        self.snapshots.append(snapshot)

    def run(self) -> None:
        self._run_turn()

    def close(self) -> None:
        self.closed = True


class _Hotkey:
    def __init__(self) -> None:
        self.started = False
        self.stopped = False

    def start(self) -> None:
        self.started = True

    def stop(self) -> bool:
        self.stopped = True
        return True


class IntegratedCompanionLifecycleTests(unittest.TestCase):
    def test_shell_capture_reply_speech_diagnostics_and_quit_share_one_runtime(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            diagnostics = DiagnosticStore(Path(temporary) / "diagnostics.json")
            injector = _Injector()
            capture = CaptureDeliveryCoordinator(
                CaptureService(
                    snapshot_resolver=SnapshotCallableAdapter(
                        injector.snapshot_target
                    )
                ),
                injector,
            )
            microphone = _Microphone(capture)
            speech = _Speech()
            inbox = _Inbox()
            controller = CompanionController(
                capture,
                microphone,
                lambda _cancelled: _Transcriber(),
                speech=speech,
                inbox=inbox,
                diagnostics=diagnostics,
                worker_starter=lambda _name, target: target(),
            )

            def run_turn() -> None:
                controller.dispatch(CompanionIntent(IntentKind.START_RECORDING))
                controller.dispatch(CompanionIntent(IntentKind.FINISH_RECORDING))
                self.assertEqual(
                    controller.snapshot.runtime.phase,
                    RuntimePhase.WAITING_FOR_CLAUDE,
                )
                assert inbox.reply is not None
                self.assertTrue(
                    inbox.reply(
                        ReplyEvent.create(
                            session="opaque-session",
                            event_id="opaque-event",
                            answer="private complete answer",
                        )
                    )
                )
                self.assertEqual(
                    controller.snapshot.runtime.phase,
                    RuntimePhase.SPEAKING,
                )
                controller.speech_finished()

            shell = _Shell(run_turn)
            hotkey = _Hotkey()
            application = DesktopCompanionApplication(controller, shell, hotkey)

            self.assertEqual(application.run(), 0)
            self.assertEqual(injector.texts, ["private operator turn"])
            self.assertEqual(len(speech.events), 1)
            self.assertEqual(speech.shutdowns, 1)
            self.assertTrue(microphone.closed)
            self.assertTrue(hotkey.started and hotkey.stopped)
            self.assertTrue(shell.closed)
            state, recovered = diagnostics.snapshot()
            self.assertFalse(recovered)
            serialized = repr(state)
            self.assertNotIn("private operator turn", serialized)
            self.assertNotIn("private complete answer", serialized)
            self.assertIn("capture_result", serialized)
            self.assertIn("reply_effect", serialized)


if __name__ == "__main__":
    unittest.main()
