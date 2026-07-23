"""Tests for the voice-fireable command catalog: parsing a session's
system/init event and merging it with the user-owned flags persisted across
launches."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from talktomeclaude import command_catalog as cc

FIXTURE = Path(__file__).parent / "fixtures" / "init-event.json"


class ParseInitEventTests(unittest.TestCase):
    def setUp(self) -> None:
        self.event = json.loads(FIXTURE.read_text(encoding="utf-8"))

    def test_excludes_builtin_interactive_commands(self) -> None:
        ids = {record["id"] for record in cc.parse_init_event(self.event)}
        self.assertNotIn("model", ids)
        self.assertNotIn("help", ids)

    def test_includes_fireable_custom_command(self) -> None:
        ids = {record["id"] for record in cc.parse_init_event(self.event)}
        self.assertIn("kiln-fire", ids)

    def test_records_carry_the_full_key_set(self) -> None:
        records = cc.parse_init_event(self.event)
        self.assertTrue(records)
        needed = {"id", "namespace", "description", "mutating", "enabled", "favorite", "fire_count"}
        self.assertTrue(needed <= set(records[0].keys()))

    def test_empty_event_yields_no_records(self) -> None:
        self.assertEqual(cc.parse_init_event({}), [])


class MergeWithSavedTests(unittest.TestCase):
    def test_refreshes_description_and_preserves_user_flags(self) -> None:
        init_records = [
            {
                "id": "kiln-fire",
                "namespace": "kiln",
                "description": "fresh",
                "mutating": True,
                "enabled": True,
                "favorite": False,
                "fire_count": 0,
            }
        ]
        saved = {"kiln-fire": {"enabled": False, "favorite": True, "fire_count": 5}}

        merged = {record["id"]: record for record in cc.merge_with_saved(init_records, saved)}["kiln-fire"]

        self.assertEqual(merged["description"], "fresh")
        self.assertIs(merged["enabled"], False)
        self.assertIs(merged["favorite"], True)
        self.assertEqual(merged["fire_count"], 5)

    def test_unknown_command_gets_default_flags(self) -> None:
        init_records = [
            {
                "id": "new-command",
                "namespace": "ns",
                "description": "d",
                "mutating": False,
                "enabled": True,
                "favorite": False,
                "fire_count": 0,
            }
        ]

        merged = cc.merge_with_saved(init_records, {})[0]

        self.assertEqual(merged["enabled"], True)
        self.assertEqual(merged["favorite"], False)
        self.assertEqual(merged["fire_count"], 0)


if __name__ == "__main__":
    unittest.main()
