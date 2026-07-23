"""Tests for the voice-fireable command catalog: parsing a session's
system/init event and merging it with the user-owned flags persisted across
launches."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from talktomeclaude import command_catalog as cc

FIXTURE = Path(__file__).parent / "fixtures" / "init-event.json"


def _record(command_id: str, namespace: str, **flags) -> dict:
    return {
        "id": command_id,
        "namespace": namespace,
        "description": "d",
        "mutating": True,
        "enabled": flags.get("enabled", True),
        "favorite": flags.get("favorite", False),
        "fire_count": flags.get("fire_count", 0),
    }


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


class QualifiedIdentityTests(unittest.TestCase):
    def test_namespaced_record_qualifies_as_namespace_id(self) -> None:
        self.assertEqual(cc.qualified_id(_record("kiln-fire", "kiln")), "kiln:kiln-fire")

    def test_top_level_record_keeps_its_bare_id(self) -> None:
        self.assertEqual(cc.qualified_id(_record("mytool", "")), "mytool")

    def test_qualified_saved_key_wins_over_legacy_bare_key(self) -> None:
        saved = {
            "kiln:kiln-fire": {"enabled": False, "favorite": True, "fire_count": 9},
            "kiln-fire": {"enabled": True, "favorite": False, "fire_count": 1},
        }
        merged = cc.merge_with_saved([_record("kiln-fire", "kiln")], saved)[0]
        self.assertIs(merged["enabled"], False)
        self.assertIs(merged["favorite"], True)
        self.assertEqual(merged["fire_count"], 9)

    def test_legacy_bare_key_is_honored_when_unambiguous(self) -> None:
        saved = {"kiln-fire": {"enabled": False, "favorite": True, "fire_count": 5}}
        merged = cc.merge_with_saved([_record("kiln-fire", "kiln")], saved)[0]
        self.assertIs(merged["enabled"], False)
        self.assertEqual(merged["fire_count"], 5)

    def test_legacy_bare_key_is_ignored_on_a_namespace_collision(self) -> None:
        saved = {"deploy": {"enabled": False, "fire_count": 4}}
        merged = cc.merge_with_saved(
            [_record("deploy", "web"), _record("deploy", "api")], saved
        )
        for record in merged:
            self.assertIs(record["enabled"], True)
            self.assertEqual(record["fire_count"], 0)

    def test_qualified_keys_still_apply_on_a_namespace_collision(self) -> None:
        saved = {"web:deploy": {"fire_count": 7}}
        merged = {
            cc.qualified_id(record): record
            for record in cc.merge_with_saved(
                [_record("deploy", "web"), _record("deploy", "api")], saved
            )
        }
        self.assertEqual(merged["web:deploy"]["fire_count"], 7)
        self.assertEqual(merged["api:deploy"]["fire_count"], 0)


class SaveFlagsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        env = mock.patch.dict(
            os.environ, {"TALKTOMECLAUDE_CONFIG_DIR": self.tmp.name}, clear=False
        )
        env.start()
        self.addCleanup(env.stop)

    def test_flags_persist_under_qualified_keys(self) -> None:
        cc.save_flags([_record("kiln-fire", "kiln", fire_count=2), _record("mytool", "")])
        saved = cc.load_saved_flags()
        self.assertEqual(set(saved), {"kiln:kiln-fire", "mytool"})
        self.assertEqual(saved["kiln:kiln-fire"]["fire_count"], 2)

    def test_legacy_bare_file_migrates_to_qualified_on_next_save(self) -> None:
        cc.catalog_path().parent.mkdir(parents=True, exist_ok=True)
        cc.catalog_path().write_text(
            json.dumps({"kiln-fire": {"enabled": False, "favorite": True, "fire_count": 3}}),
            encoding="utf-8",
        )
        merged = cc.merge_with_saved([_record("kiln-fire", "kiln")], cc.load_saved_flags())
        self.assertEqual(merged[0]["fire_count"], 3)
        cc.save_flags(merged)
        self.assertEqual(set(cc.load_saved_flags()), {"kiln:kiln-fire"})


if __name__ == "__main__":
    unittest.main()
