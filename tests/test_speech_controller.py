from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from talktomeclaude.reply import ReplyEvent
from talktomeclaude.speech.controller import CanonicalSpeechController
from talktomeclaude.speech.session import (
    OralSessionError,
    OralSessionStore,
    OralStatus,
)


class _Pipeline:
    def __init__(self, state_path: Path) -> None:
        self.state_path = state_path
        self.offers: list[tuple[str, str]] = []
        self.stops = 0
        self.limit = 3
        self.events: list[str] = []
        self.effect_ids: set[str] = set()

    def offer(
        self,
        unit_id: str,
        text: str,
        *,
        effect_id: str | None = None,
    ) -> bool:
        if effect_id is not None and effect_id in self.effect_ids:
            return True
        if len(self.offers) >= self.limit:
            return False
        document = json.loads(self.state_path.read_text(encoding="utf-8"))
        self.assert_frozen(document)
        self.offers.append((unit_id, text))
        if effect_id is not None:
            self.effect_ids.add(effect_id)
        self.events.append("offer")
        return True

    @staticmethod
    def assert_frozen(document: dict[str, object]) -> None:
        answers = document.get("answers")
        if not isinstance(answers, dict) or not answers:
            raise AssertionError("speech offered before roadmap commit")
        record = next(iter(answers.values()))
        if not isinstance(record, dict) or "oral_roadmap_frozen" not in record:
            raise AssertionError("speech offered before roadmap commit")

    def stop(self) -> object:
        self.events.append("stop")
        self.stops += 1
        self.offers.clear()
        return object()


def _event(identity: str = "answer-1") -> ReplyEvent:
    return ReplyEvent.create(
        session="session-1",
        event_id=identity,
        answer=(
            "# Result\nThe result is 42.\n\n"
            "# Safety\nNever delete C:\\data. Risk: allow 3 minutes.\n\n"
            "# Next\nRun `python -m verify`.\n"
        ),
    )


class SpeechControllerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.path = Path(self.temporary.name) / "oral.json"
        self.store = OralSessionStore(self.path)
        self.pipeline = _Pipeline(self.path)
        self.controller = CanonicalSpeechController(self.store, self.pipeline)

    def test_canonical_reply_commits_before_offer_and_duplicate_does_not_reoffer(
        self,
    ) -> None:
        event = _event()
        first = self.controller.accept(event)
        initial_offers = list(self.pipeline.offers)
        duplicate = self.controller.accept(event)

        self.assertTrue(first.roadmap_created)
        self.assertFalse(duplicate.roadmap_created)
        self.assertGreater(first.units_scheduled, 0)
        self.assertEqual(duplicate.units_scheduled, 0)
        self.assertEqual(initial_offers, self.pipeline.offers)

    def test_preview_claim_releases_on_full_queue_and_reuses_stable_effect(self) -> None:
        self.pipeline.limit = 1
        event = _event("answer-preview")
        accepted = self.controller.accept(event)
        self.assertEqual(accepted.units_scheduled, 1)

        first_id = self.pipeline.offers.pop(0)[0]
        self.assertTrue(self.controller.unit_completed(first_id))
        state = self.store.restore(event.event_id)
        assert state is not None
        self.assertIsNotNone(state.current_unit)

        self.pipeline.limit = 3
        duplicate = self.controller.accept(event)
        self.assertGreaterEqual(duplicate.units_scheduled, 1)
        self.assertEqual(len(self.pipeline.effect_ids), 1)

    def test_completion_is_durable_and_refills_without_accepting_stale_callback(
        self,
    ) -> None:
        self.controller.accept(_event())
        first_id = self.pipeline.offers[0][0]
        self.pipeline.offers.pop(0)

        self.assertTrue(self.controller.unit_completed(first_id))
        self.assertFalse(self.controller.unit_completed(first_id))
        restored = OralSessionStore(self.path).restore("answer-1")
        self.assertIsNotNone(restored)
        assert restored is not None
        self.assertIn(first_id, restored.spoken_unit_ids)

    def test_completion_storage_failure_retains_retry_authority(self) -> None:
        self.controller.accept(_event())
        unit_id = self.pipeline.offers[0][0]
        original = self.store.complete_unit
        attempts = [0]

        def flaky(answer_id: str, completed_unit_id: str):
            attempts[0] += 1
            if attempts[0] == 1:
                raise OralSessionError("storage unavailable")
            return original(answer_id, completed_unit_id)

        with mock.patch.object(self.store, "complete_unit", side_effect=flaky):
            with self.assertRaises(OralSessionError):
                self.controller.unit_completed(unit_id)
            self.assertTrue(self.controller.unit_completed(unit_id))

        restored = self.store.restore("answer-1")
        assert restored is not None
        self.assertIn(unit_id, restored.spoken_unit_ids)

    def test_restart_resumes_exact_frozen_cursor_without_replanning_identity(self) -> None:
        self.controller.accept(_event())
        first_id = self.pipeline.offers[0][0]
        self.pipeline.offers.pop(0)
        self.controller.unit_completed(first_id)
        before = self.store.restore("answer-1")
        assert before is not None

        restarted_pipeline = _Pipeline(self.path)
        restarted = CanonicalSpeechController(
            OralSessionStore(self.path), restarted_pipeline
        )
        accepted = restarted.accept(_event())
        after = OralSessionStore(self.path).restore("answer-1")

        self.assertFalse(accepted.roadmap_created)
        assert after is not None
        self.assertEqual(before.roadmap.to_dict(), after.roadmap.to_dict())
        self.assertEqual(before.cursor, after.cursor)
        self.assertNotIn(first_id, {item[0] for item in restarted_pipeline.offers})

    def test_interruption_stops_before_parking_and_never_autoresumes(self) -> None:
        self.controller.accept(_event())
        result = self.controller.interrupt()
        parked = self.store.restore("answer-1")

        self.assertTrue(result.requires_new_turn)
        self.assertEqual(self.pipeline.events[-1], "stop")
        assert parked is not None
        self.assertEqual(parked.status, OralStatus.PARKED)
        self.assertEqual(self.pipeline.offers, [])

        returned = self.controller.go_back()
        self.assertEqual(returned.state.status, OralStatus.ACTIVE)  # type: ignore[union-attr]
        self.assertTrue(self.pipeline.offers)

    def test_completed_answer_is_not_spoken_again_on_reply_replay(self) -> None:
        event = _event("answer-complete")
        controller = self.controller
        controller.accept(event)
        while True:
            state = self.store.restore(event.event_id)
            assert state is not None
            if state.status is OralStatus.COMPLETE:
                break
            unit_id = state.current_unit.unit_id  # type: ignore[union-attr]
            self.pipeline.offers = [
                item for item in self.pipeline.offers if item[0] != unit_id
            ]
            controller.unit_completed(unit_id)
        self.pipeline.offers.clear()

        replay = controller.accept(event)

        self.assertEqual(replay.units_scheduled, 0)
        self.assertEqual(self.pipeline.offers, [])


if __name__ == "__main__":
    unittest.main()
