from __future__ import annotations

import unittest
from dataclasses import replace

from talktomeclaude.speech.canonical import canonicalize
from talktomeclaude.speech.planner import (
    OralTopic,
    OralUnit,
    RoadmapError,
    StructuralMutationError,
    UnitKind,
    deterministic_plan,
    refine_unsaid,
    seal_roadmap,
    validate_roadmap,
)
from talktomeclaude.speech.preservation import (
    BlockDispositionKind,
)


def _complex_answer():
    return canonicalize(
        "complex-answer",
        "# Result\nThe operation succeeded in 42 ms. It may need one retry.\n\n"
        "# Safety\nNever remove C:\\data\\cache. Risk: the backup could be stale.\n\n"
        "# Next step\nRun `python -m verify` and keep port 443 available.\n"
        "Trailing prose stays owned.",
    )


class DeterministicPlannerTests(unittest.TestCase):
    def test_simple_answer_has_no_artificial_preview_or_checkpoint(self) -> None:
        answer = canonicalize("simple", "A short direct answer with value 7.")
        roadmap = deterministic_plan(answer)

        self.assertFalse(roadmap.complex)
        self.assertEqual(1, len(roadmap.topics))
        self.assertEqual([UnitKind.SECTION], [unit.kind for unit in roadmap.units])
        self.assertEqual(answer.text, roadmap.units[0].wording)
        self.assertEqual((), roadmap.checkpoint_sequence)

    def test_complex_plan_is_stable_bounded_and_fully_accounts_for_blocks(self) -> None:
        answer = _complex_answer()
        first = deterministic_plan(answer)
        second = deterministic_plan(answer)

        self.assertEqual(first, second)
        self.assertTrue(first.complex)
        self.assertGreaterEqual(len(first.topics), 2)
        self.assertLessEqual(len(first.topics), 4)
        self.assertEqual(UnitKind.OUTCOME, first.units[0].kind)
        self.assertLessEqual(
            len([item for item in first.units[0].wording.replace("!", ".").split(".") if item.strip()]),
            2,
        )
        self.assertEqual(UnitKind.PREVIEW, first.units[1].kind)
        owned = [block_id for topic in first.topics for block_id in topic.block_ids]
        self.assertEqual(
            [block.block_id for block in answer.blocks],
            owned,
        )
        self.assertEqual(
            first.checkpoint_sequence,
            tuple(unit.unit_id for unit in first.units if unit.kind is UnitKind.CHECKPOINT),
        )
        validate_roadmap(answer, first)

    def test_structural_late_mutations_are_rejected_in_full(self) -> None:
        answer = _complex_answer()
        frozen = deterministic_plan(answer)
        first = frozen.topics[0]
        second = frozen.topics[1]
        merged = replace(
            first,
            block_ids=first.block_ids + second.block_ids,
        )
        split_at = max(1, len(first.block_ids) // 2)
        split = (
            replace(first, block_ids=first.block_ids[:split_at]),
            OralTopic(
                "topic-split",
                first.label + " continued",
                first.block_ids[split_at:],
            ),
        )
        mutations = {
            "rename": seal_roadmap(replace(frozen, topics=(replace(first, label="renamed"), *frozen.topics[1:]))),
            "reorder": seal_roadmap(replace(frozen, topics=(second, first, *frozen.topics[2:]))),
            "add": seal_roadmap(replace(
                frozen,
                topics=(*frozen.topics, OralTopic("topic-extra", "Extra", ())),
            )),
            "drop": seal_roadmap(replace(frozen, topics=frozen.topics[:-1])),
            "merge": seal_roadmap(replace(frozen, topics=(merged, *frozen.topics[2:]))),
            "split": seal_roadmap(replace(frozen, topics=(*split, *frozen.topics[1:]))),
            "reassign": seal_roadmap(replace(
                frozen,
                topics=(
                    replace(first, block_ids=first.block_ids + second.block_ids[:1]),
                    replace(second, block_ids=second.block_ids[1:]),
                    *frozen.topics[2:],
                ),
            )),
            "checkpoint": seal_roadmap(replace(frozen, checkpoint_sequence=frozen.checkpoint_sequence[:-1])),
        }
        for name, candidate in mutations.items():
            with self.subTest(name=name), self.assertRaises(RoadmapError):
                refine_unsaid(
                    answer,
                    frozen,
                    candidate,
                    spoken_unit_ids=frozenset(),
                )

    def test_only_unsaid_wording_inside_frozen_boundaries_can_change(self) -> None:
        answer = _complex_answer()
        frozen = deterministic_plan(answer)
        unsaid = next(unit for unit in frozen.units if unit.kind is UnitKind.SECTION)
        changed_wording = unsaid.wording.replace(".", "!")
        changed_dispositions = tuple(
            replace(item, wording=item.wording.replace(".", "!"))
            if item.unit_id == unsaid.unit_id
            else item
            for item in frozen.block_dispositions
        )
        candidate_units = tuple(
            replace(unit, wording=changed_wording)
            if unit.unit_id == unsaid.unit_id
            else unit
            for unit in frozen.units
        )
        candidate = seal_roadmap(
            replace(
                frozen,
                units=candidate_units,
                block_dispositions=changed_dispositions,
            )
        )

        refined = refine_unsaid(
            answer,
            frozen,
            candidate,
            spoken_unit_ids=frozenset(),
        )
        self.assertEqual(
            frozen.structural_signature(), refined.structural_signature()
        )
        self.assertNotEqual(unsaid.wording, refined.unit(unsaid.unit_id).wording)

        with self.assertRaises(StructuralMutationError):
            refine_unsaid(
                answer,
                frozen,
                candidate,
                spoken_unit_ids=frozenset({unsaid.unit_id}),
            )

    def test_protected_values_and_full_coverage_cannot_be_dropped(self) -> None:
        answer = _complex_answer()
        roadmap = deterministic_plan(answer)
        section = next(unit for unit in roadmap.units if unit.kind is UnitKind.SECTION)
        broken_units = tuple(
            OralUnit(
                unit.unit_id,
                unit.kind,
                unit.topic_id,
                unit.block_ids,
                "details omitted",
            )
            if unit.unit_id == section.unit_id
            else unit
            for unit in roadmap.units
        )
        with self.assertRaises(RoadmapError):
            validate_roadmap(
                answer,
                seal_roadmap(replace(roadmap, units=broken_units)),
            )

    def test_dispositions_account_for_each_block_exactly_once(self) -> None:
        answer = _complex_answer()
        roadmap = deterministic_plan(answer)

        self.assertEqual(
            [block.block_id for block in answer.blocks],
            [item.block_id for item in roadmap.block_dispositions],
        )
        self.assertTrue(
            all(
                item.kind is BlockDispositionKind.SPOKEN and item.unit_id
                for item in roadmap.block_dispositions
            )
        )

        missing = seal_roadmap(
            replace(roadmap, block_dispositions=roadmap.block_dispositions[:-1])
        )
        duplicate = seal_roadmap(
            replace(
                roadmap,
                block_dispositions=(
                    *roadmap.block_dispositions,
                    roadmap.block_dispositions[-1],
                ),
            )
        )
        for candidate in (missing, duplicate):
            with self.assertRaises(RoadmapError):
                validate_roadmap(answer, candidate)

    def test_repeated_protected_values_are_multiplicity_safe(self) -> None:
        answer = canonicalize("repeated", "Value 42 must remain 42 exactly.")
        roadmap = deterministic_plan(answer)
        disposition = roadmap.block_dispositions[0]
        section = roadmap.units[0]
        shortened = replace(disposition, wording="Value 42 must remain exactly.")
        candidate = seal_roadmap(
            replace(
                roadmap,
                block_dispositions=(shortened,),
                units=(replace(section, wording=shortened.wording),),
            )
        )

        with self.assertRaises(RoadmapError):
            validate_roadmap(answer, candidate)

    def test_plain_facts_and_framing_cannot_be_rewritten(self) -> None:
        answer = canonicalize(
            "plain-facts",
            "# Result\nThe blue widget improves battery life and reduces heat.\n\n"
            "# Detail\nThe product is available now.\n",
        )
        roadmap = deterministic_plan(answer)
        section = next(unit for unit in roadmap.units if unit.kind is UnitKind.SECTION)
        changed_dispositions = tuple(
            replace(item, wording="The widget is available.")
            if item.unit_id == section.unit_id
            else item
            for item in roadmap.block_dispositions
        )
        changed_section = replace(section, wording="The widget is available.")
        omitted = seal_roadmap(
            replace(
                roadmap,
                units=tuple(
                    changed_section if item.unit_id == section.unit_id else item
                    for item in roadmap.units
                ),
                block_dispositions=changed_dispositions,
            )
        )
        invented = seal_roadmap(
            replace(
                roadmap,
                units=(
                    replace(roadmap.units[0], wording="This product is known to explode."),
                    *roadmap.units[1:],
                ),
            )
        )

        for candidate in (omitted, invented):
            with self.assertRaises(RoadmapError):
                validate_roadmap(answer, candidate)

    def test_semantic_operators_and_unicode_decisions_are_authoritative(self) -> None:
        answer = canonicalize(
            "semantic-symbols",
            "The invariant is x < y. Deployment status: ✅.",
        )
        roadmap = deterministic_plan(answer)
        disposition = roadmap.block_dispositions[0]
        section = roadmap.units[0]
        for wording in (
            "The invariant is x > y. Deployment status: ✅.",
            "The invariant is x < y. Deployment status: ❌.",
        ):
            candidate = seal_roadmap(
                replace(
                    roadmap,
                    units=(replace(section, wording=wording),),
                    block_dispositions=(replace(disposition, wording=wording),),
                )
            )
            with self.subTest(wording=wording), self.assertRaises(RoadmapError):
                validate_roadmap(answer, candidate)

    def test_strict_structure_rejects_bad_order_long_outcome_and_empty_topic(self) -> None:
        answer = _complex_answer()
        roadmap = deterministic_plan(answer)
        outcome = roadmap.units[0]
        preview = roadmap.units[1]
        mutations = (
            replace(roadmap, units=(preview, outcome, *roadmap.units[2:])),
            replace(
                roadmap,
                units=(
                    replace(outcome, wording="One. Two. Three."),
                    *roadmap.units[1:],
                ),
            ),
            replace(
                roadmap,
                topics=(replace(roadmap.topics[0], block_ids=()), *roadmap.topics[1:]),
            ),
        )
        for mutation in mutations:
            with self.assertRaises(RoadmapError):
                validate_roadmap(answer, seal_roadmap(mutation))

    def test_content_bearing_roadmap_fields_do_not_leak_through_repr(self) -> None:
        answer = canonicalize("private", "SECRET canonical answer value 42.")
        roadmap = deterministic_plan(answer)

        rendered = repr(roadmap)
        self.assertNotIn("SECRET", rendered)
        self.assertNotIn("canonical answer", rendered)


if __name__ == "__main__":
    unittest.main()
