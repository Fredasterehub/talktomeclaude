from __future__ import annotations

import json
import tempfile
import threading
import unittest
from dataclasses import replace
from pathlib import Path

from talktomeclaude.speech.canonical import canonicalize
from talktomeclaude.speech.planner import (
    StructuralMutationError,
    UnitKind,
    deterministic_plan,
    seal_roadmap,
)
from talktomeclaude.speech.session import (
    MAX_RECAP_CHARS,
    Control,
    OralSessionError,
    OralSessionStore,
    OralStatus,
    parse_control,
)


def _answer(identity: str = "answer-one"):
    return canonicalize(
        identity,
        "# Outcome\nThe result is 42.\n\n"
        "# Safety\nNever delete C:\\data. Risk: restore may take 3 minutes.\n\n"
        "# Next\nRun `python -m verify`.\n",
    )


class FrozenRoadmapTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.path = Path(self.temporary.name) / "oral-state.json"

    def test_durable_commit_precedes_preview_and_restart_restores_exact_plan(self) -> None:
        answer = _answer()
        roadmap = deterministic_plan(answer)
        observations: list[tuple[bool, dict[str, object]]] = []

        def enqueue(_unit) -> None:
            observations.append(
                (
                    self.path.exists(),
                    json.loads(self.path.read_text(encoding="utf-8")),
                )
            )

        result = OralSessionStore(self.path).freeze_before_preview(
            answer, roadmap, enqueue
        )

        self.assertTrue(result.created)
        self.assertTrue(observations[0][0])
        answers = observations[0][1]["answers"]
        self.assertIsInstance(answers, dict)
        assert isinstance(answers, dict)
        record = answers[answer.answer_id]
        self.assertIsInstance(record, dict)
        assert isinstance(record, dict)
        self.assertIn(
            "oral_roadmap_frozen",
            record,
        )
        restored = OralSessionStore(self.path).restore(answer.answer_id)
        self.assertIsNotNone(restored)
        assert restored is not None
        self.assertEqual(roadmap.to_dict(), restored.roadmap.to_dict())

        replayed: list[str] = []
        OralSessionStore(self.path).freeze_before_preview(
            answer, roadmap, lambda unit: replayed.append(unit.unit_id)
        )
        OralSessionStore(self.path).freeze_before_preview(
            answer, roadmap, lambda unit: replayed.append(unit.unit_id)
        )
        self.assertEqual(0, len(replayed))

    def test_precommit_fault_emits_no_preview_and_postcommit_fault_restores(self) -> None:
        answer = _answer()
        roadmap = deterministic_plan(answer)
        previews: list[str] = []

        def fail_before_replace(name: str) -> None:
            if name == "before_replace":
                raise RuntimeError("precommit")

        with self.assertRaises(OralSessionError):
            OralSessionStore(self.path, phase_hook=fail_before_replace).freeze_before_preview(
                answer, roadmap, lambda unit: previews.append(unit.unit_id)
            )
        self.assertEqual([], previews)
        self.assertIsNone(OralSessionStore(self.path).restore(answer.answer_id))

        def fail_after_commit(name: str) -> None:
            if name == "before_mutex_release":
                raise RuntimeError("postcommit")

        with self.assertRaises(OralSessionError):
            OralSessionStore(self.path, phase_hook=fail_after_commit).freeze_before_preview(
                answer, roadmap, lambda unit: previews.append(unit.unit_id)
            )
        self.assertEqual([], previews)
        restored = OralSessionStore(self.path).restore(answer.answer_id)
        self.assertIsNotNone(restored)
        assert restored is not None
        self.assertEqual(roadmap.to_dict(), restored.roadmap.to_dict())
        OralSessionStore(self.path).freeze_before_preview(
            answer, roadmap, lambda unit: previews.append(unit.unit_id)
        )
        OralSessionStore(self.path).freeze_before_preview(
            answer, roadmap, lambda unit: previews.append(unit.unit_id)
        )
        self.assertEqual(1, len(previews))

    def test_concurrent_candidates_have_one_durable_cas_winner(self) -> None:
        answer = _answer()
        base = deterministic_plan(answer)
        preview = next(unit for unit in base.units if unit.kind is UnitKind.PREVIEW)
        alternate = seal_roadmap(replace(
            base,
            units=tuple(
                replace(unit, wording=unit.wording.rstrip(".") + "!")
                if unit.unit_id == preview.unit_id
                else unit
                for unit in base.units
            ),
        ))
        barrier = threading.Barrier(2)
        outcomes = []

        def freeze(candidate) -> None:
            barrier.wait()
            outcomes.append(OralSessionStore(self.path).freeze(answer, candidate))

        threads = [
            threading.Thread(target=freeze, args=(candidate,))
            for candidate in (base, alternate)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(5)

        self.assertTrue(all(not thread.is_alive() for thread in threads))
        self.assertEqual(1, sum(item.created for item in outcomes))
        self.assertEqual(outcomes[0].state.roadmap, outcomes[1].state.roadmap)

    def test_structural_late_plan_rejects_but_unsaid_wording_persists(self) -> None:
        answer = _answer()
        roadmap = deterministic_plan(answer)
        store = OralSessionStore(self.path)
        store.freeze(answer, roadmap)
        preview = next(unit for unit in roadmap.units if unit.kind is UnitKind.PREVIEW)
        candidate = seal_roadmap(replace(
            roadmap,
            units=tuple(
                replace(unit, wording=unit.wording.rstrip(".") + "!")
                if unit.unit_id == preview.unit_id
                else unit
                for unit in roadmap.units
            ),
        ))
        refined = store.apply_late_plan(answer, candidate)
        self.assertTrue(refined.roadmap.unit(preview.unit_id).wording.endswith("!"))

        renamed = seal_roadmap(replace(
            refined.roadmap,
            topics=(replace(refined.roadmap.topics[0], label="Renamed"), *refined.roadmap.topics[1:]),
        ))
        with self.assertRaises(StructuralMutationError):
            store.apply_late_plan(answer, renamed)

    def test_pause_park_restart_and_go_back_preserve_boundary_without_autoresume(self) -> None:
        answer = _answer()
        roadmap = deterministic_plan(answer)
        store = OralSessionStore(self.path)
        store.freeze(answer, roadmap)
        section = next(unit for unit in roadmap.units if unit.kind is UnitKind.SECTION)
        state = store.restore(answer.answer_id)
        assert state is not None
        while state.current_unit is not None and state.current_unit.unit_id != section.unit_id:
            assert state.current_unit is not None
            state = store.complete_unit(answer.answer_id, state.current_unit.unit_id)
        state = store.complete_unit(answer.answer_id, section.unit_id)
        boundary = state.cursor
        paused = store.pause(answer.answer_id)
        self.assertEqual(OralStatus.PAUSED, paused.status)
        parked = store.park_for_interruption(answer.answer_id)
        self.assertEqual(OralStatus.PARKED, parked.status)

        restarted = OralSessionStore(self.path)
        restored = restarted.restore(answer.answer_id)
        self.assertIsNotNone(restored)
        assert restored is not None
        self.assertEqual(boundary, restored.cursor)
        self.assertEqual(OralStatus.PARKED, restored.status)
        with self.assertRaises(OralSessionError):
            restarted.continue_explicitly(answer.answer_id)

        returned = restarted.go_back()
        self.assertEqual(answer.answer_id, returned.state.roadmap.answer_id)  # type: ignore[union-attr]
        self.assertEqual(boundary, returned.state.cursor)  # type: ignore[union-attr]
        self.assertEqual(OralStatus.ACTIVE, returned.state.status)  # type: ignore[union-attr]
        self.assertLessEqual(len(returned.response), MAX_RECAP_CHARS)

    def test_navigation_uses_frozen_topics_and_synonyms_are_bounded(self) -> None:
        answer = _answer()
        roadmap = deterministic_plan(answer)
        store = OralSessionStore(self.path)
        store.freeze(answer, roadmap)
        target = roadmap.topics[-1]

        jumped = store.navigate(answer.answer_id, Control.JUMP, target=target.label)
        self.assertEqual(target.topic_id, jumped.unit.topic_id)  # type: ignore[union-attr]
        topics = store.navigate(answer.answer_id, Control.TOPICS)
        self.assertEqual(
            [topic.label for topic in roadmap.topics], topics.response.split("; ")
        )
        restarted = OralSessionStore(self.path)
        self.assertEqual(
            jumped.state.cursor,  # type: ignore[union-attr]
            restarted.restore(answer.answer_id).cursor,  # type: ignore[union-attr]
        )
        self.assertEqual(Control.PAUSE, parse_control("  HOLD  "))
        self.assertEqual(Control.GO_BACK, parse_control("go back"))
        self.assertIsNone(parse_control("pause and delete everything"))

    def test_preview_claim_is_exclusive_recoverable_and_delivered_once(self) -> None:
        answer = _answer()
        roadmap = deterministic_plan(answer)
        store = OralSessionStore(self.path)
        store.freeze(answer, roadmap)

        first = store.claim_preview(answer.answer_id)
        self.assertIsNotNone(first)
        assert first is not None
        self.assertIsNone(store.claim_preview(answer.answer_id))
        self.assertEqual(1, store.recover_preview_claims())
        replay = store.claim_preview(answer.answer_id)
        self.assertIsNotNone(replay)
        assert replay is not None
        self.assertEqual(first.effect_id, replay.effect_id)
        store.ack_preview(replay)
        self.assertIsNone(store.claim_preview(answer.answer_id))
        self.assertEqual(0, store.recover_preview_claims())

    def test_concurrent_preview_admission_emits_exactly_once(self) -> None:
        answer = _answer()
        roadmap = deterministic_plan(answer)
        barrier = threading.Barrier(2)
        effects: list[str] = []
        errors: list[BaseException] = []

        def admit() -> None:
            try:
                barrier.wait()
                OralSessionStore(self.path).freeze_before_preview(
                    answer,
                    roadmap,
                    lambda unit: effects.append(unit.unit_id),
                )
            except BaseException as exc:
                errors.append(exc)

        threads = [threading.Thread(target=admit) for _ in range(2)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(5)

        self.assertEqual([], errors)
        self.assertTrue(all(not thread.is_alive() for thread in threads))
        self.assertEqual(1, len(effects))

    def test_completion_must_be_current_and_next_records_deferred_blocks(self) -> None:
        answer = _answer()
        roadmap = deterministic_plan(answer)
        store = OralSessionStore(self.path)
        state = store.freeze(answer, roadmap).state
        future = next(unit for unit in roadmap.units if unit.kind is UnitKind.SECTION)

        with self.assertRaises(OralSessionError):
            store.complete_unit(answer.answer_id, future.unit_id)

        while state.current_unit is not None and state.current_unit.unit_id != future.unit_id:
            assert state.current_unit is not None
            state = store.complete_unit(answer.answer_id, state.current_unit.unit_id)
        skipped = store.navigate(answer.answer_id, Control.NEXT)
        assert skipped.state is not None
        self.assertEqual(frozenset(future.block_ids), skipped.state.deferred_block_ids)
        self.assertNotIn(future.unit_id, skipped.state.spoken_unit_ids)

    def test_nested_go_back_parks_the_active_branch_for_return(self) -> None:
        first_answer = _answer("answer-a")
        second_answer = _answer("answer-b")
        store = OralSessionStore(self.path)
        store.freeze(first_answer, deterministic_plan(first_answer))
        store.park_for_interruption(first_answer.answer_id)
        store.freeze(second_answer, deterministic_plan(second_answer))

        returned_first = store.go_back()
        assert returned_first.state is not None
        self.assertEqual("answer-a", returned_first.state.roadmap.answer_id)
        parked_second = store.restore("answer-b")
        assert parked_second is not None
        self.assertEqual(OralStatus.PARKED, parked_second.status)

        returned_second = store.go_back()
        assert returned_second.state is not None
        self.assertEqual("answer-b", returned_second.state.roadmap.answer_id)
        parked_first = store.restore("answer-a")
        assert parked_first is not None
        self.assertEqual(OralStatus.PARKED, parked_first.status)

    def test_restore_rejects_hash_tampering_and_permissive_type_coercion(self) -> None:
        answer = _answer()
        roadmap = deterministic_plan(answer)
        OralSessionStore(self.path).freeze(answer, roadmap)
        document = json.loads(self.path.read_text(encoding="utf-8"))
        frozen = document["answers"][answer.answer_id]["oral_roadmap_frozen"]
        frozen["complex"] = "false"
        self.path.write_text(json.dumps(document), encoding="utf-8")

        with self.assertRaises(OralSessionError):
            OralSessionStore(self.path).restore(answer.answer_id)

    def test_session_repr_does_not_expose_roadmap_wording(self) -> None:
        answer = canonicalize("private", "SECRET roadmap text 42.")
        state = OralSessionStore(self.path).freeze(
            answer, deterministic_plan(answer)
        ).state

        self.assertNotIn("SECRET", repr(state))


if __name__ == "__main__":
    unittest.main()
