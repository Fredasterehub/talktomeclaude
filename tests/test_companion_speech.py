from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from talktomeclaude.companion.speech import CompanionSpeech, _MuteGate
from talktomeclaude.reply import ReplyEvent
from talktomeclaude.speech import CanonicalSpeechController, OralSessionStore, OralStatus


class _Pipeline:
    def __init__(self) -> None:
        self.offers: list[tuple[str, str, str | None]] = []
        self.stops = 0

    def offer(
        self, unit_id: str, text: str, *, effect_id: str | None = None
    ) -> bool:
        self.offers.append((unit_id, text, effect_id))
        return True

    def stop(self) -> object:
        self.stops += 1
        return object()


class _Runtime:
    def __init__(self) -> None:
        self.shutdowns = 0

    def shutdown(self) -> bool:
        self.shutdowns += 1
        return True


class CompanionSpeechTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.path = Path(self.temporary.name) / "oral.json"
        self.pipeline = _Pipeline()
        self.gate = _MuteGate(self.pipeline, muted=False)
        self.session = OralSessionStore(self.path)
        self.controller = CanonicalSpeechController(self.session, self.gate)
        self.runtime = _Runtime()
        self.finished = 0

        def finish() -> None:
            self.finished += 1

        self.speech = CompanionSpeech(
            self.controller,
            self.session,
            self.gate,
            self.runtime,
            on_answer_finished=finish,
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    @staticmethod
    def event(identifier: str = "reply-1") -> ReplyEvent:
        return ReplyEvent.create(
            session="session", event_id=identifier, answer="A short complete reply."
        )

    def test_muted_answer_freezes_without_audio_admission(self) -> None:
        self.speech.set_muted(True)
        event = self.event()
        self.speech.accept(event)

        state = self.session.restore(event.event_id)
        self.assertIsNotNone(state)
        self.assertEqual(self.pipeline.offers, [])
        self.assertTrue(self.speech.muted)

    def test_unmute_does_not_auto_resume_muted_answer(self) -> None:
        self.speech.set_muted(True)
        self.speech.accept(self.event())
        self.speech.set_muted(False)
        self.assertEqual(self.pipeline.offers, [])

    def test_completion_persists_then_notifies_whole_answer(self) -> None:
        event = self.event()
        self.speech.accept(event)
        self.assertEqual(len(self.pipeline.offers), 1)
        unit_id = self.pipeline.offers[0][0]

        self.assertTrue(self.speech._unit_completed(unit_id))

        state = self.session.restore(event.event_id)
        assert state is not None
        self.assertEqual(state.status, OralStatus.COMPLETE)
        self.assertEqual(self.finished, 1)

    def test_interrupt_parks_before_new_turn_and_shutdown_is_idempotent(self) -> None:
        event = self.event()
        self.speech.accept(event)
        self.speech.interrupt()
        state = self.session.restore(event.event_id)
        assert state is not None
        self.assertEqual(state.status, OralStatus.PARKED)

        self.assertTrue(self.speech.shutdown())
        self.assertTrue(self.speech.shutdown())
        self.assertEqual(self.runtime.shutdowns, 1)


if __name__ == "__main__":
    unittest.main()
