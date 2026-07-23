"""Tests for the voice-command intent router: a deterministic keyword
prefilter ahead of any model round-trip, the isolated intent sub-call command
builder, the constrained intent-JSON parser, and the sanitizers that reduce
the parser's untrusted output to safe, catalog-validated values."""

from __future__ import annotations

import json
import unittest

from talktomeclaude import intent

CATALOG = [
    {"id": "kiln-fire", "namespace": "kiln"},
    {"id": "commit", "namespace": "git"},
]


class KeywordPrefilterTests(unittest.TestCase):
    def test_resolves_exact_command_name(self) -> None:
        self.assertEqual(intent.keyword_prefilter("kiln-fire", CATALOG), "kiln-fire")

    def test_resolves_namespaced_utterance(self) -> None:
        self.assertEqual(intent.keyword_prefilter("kiln:kiln-fire", CATALOG), "kiln-fire")

    def test_is_case_insensitive(self) -> None:
        self.assertEqual(intent.keyword_prefilter("KILN-FIRE", CATALOG), "kiln-fire")

    def test_returns_none_for_unrelated_utterance(self) -> None:
        self.assertIsNone(intent.keyword_prefilter("what is the capital of france", CATALOG))

    def test_returns_none_for_empty_catalog(self) -> None:
        self.assertIsNone(intent.keyword_prefilter("commit", []))


class IntentSubcallCommandTests(unittest.TestCase):
    def test_builds_isolated_json_call(self) -> None:
        cmd = intent.intent_subcall_command("classify this", "claude-haiku")
        self.assertIn("-p", cmd)
        self.assertIn("--model", cmd)
        self.assertIn("claude-haiku", cmd)
        self.assertIn("--output-format", cmd)
        self.assertIn("json", cmd)
        self.assertNotIn("--resume", cmd)

    def test_carries_the_prompt_text(self) -> None:
        cmd = intent.intent_subcall_command("classify this", "claude-haiku")
        self.assertIn("classify this", cmd)


class ParseIntentResponseTests(unittest.TestCase):
    def test_exposes_the_locked_intent_contract(self) -> None:
        payload = json.dumps(
            {
                "command_id": "commit",
                "args": "-m x",
                "missing_slots": [],
                "confidence": 0.9,
                "alternatives": [],
            }
        )
        result = intent.parse_intent_response(payload)
        self.assertEqual(result.command_id, "commit")
        self.assertEqual(result.args, "-m x")
        self.assertEqual(result.missing_slots, [])
        self.assertEqual(result.confidence, 0.9)
        self.assertEqual(result.alternatives, [])

    def test_parses_missing_slots_and_alternatives(self) -> None:
        payload = json.dumps(
            {
                "command_id": None,
                "args": "",
                "missing_slots": ["message"],
                "confidence": 0.4,
                "alternatives": ["commit", "commit-amend"],
            }
        )
        result = intent.parse_intent_response(payload)
        self.assertIsNone(result.command_id)
        self.assertEqual(result.missing_slots, ["message"])
        self.assertEqual(result.alternatives, ["commit", "commit-amend"])


QUALIFIED_CATALOG = [
    {"id": "kiln-fire", "namespace": "kiln"},
    {"id": "commit", "namespace": "git"},
    {"id": "deploy", "namespace": "web"},
    {"id": "deploy", "namespace": "api"},
    {"id": "mytool", "namespace": ""},
]


class SanitizeMissingSlotsTests(unittest.TestCase):
    def test_non_list_input_yields_no_slots(self) -> None:
        self.assertEqual(intent.sanitize_missing_slots("message"), [])
        self.assertEqual(intent.sanitize_missing_slots(None), [])

    def test_keeps_only_unique_nonempty_strings(self) -> None:
        raw = [None, "", "   ", 3, "message", "message ", "scope"]
        self.assertEqual(intent.sanitize_missing_slots(raw), ["message", "scope"])


class SanitizeAlternativesTests(unittest.TestCase):
    def test_non_list_input_yields_no_alternatives(self) -> None:
        self.assertEqual(intent.sanitize_alternatives("commit", QUALIFIED_CATALOG), [])

    def test_bare_name_canonicalizes_to_qualified_identity(self) -> None:
        self.assertEqual(
            intent.sanitize_alternatives(["commit"], QUALIFIED_CATALOG), ["git:commit"]
        )

    def test_top_level_command_keeps_its_bare_identity(self) -> None:
        self.assertEqual(
            intent.sanitize_alternatives(["mytool"], QUALIFIED_CATALOG), ["mytool"]
        )

    def test_ambiguous_bare_name_is_dropped(self) -> None:
        self.assertEqual(intent.sanitize_alternatives(["deploy"], QUALIFIED_CATALOG), [])

    def test_unknown_and_non_string_entries_are_dropped(self) -> None:
        self.assertEqual(
            intent.sanitize_alternatives(["bogus", 7, None], QUALIFIED_CATALOG), []
        )

    def test_dedupes_bare_and_qualified_forms(self) -> None:
        self.assertEqual(
            intent.sanitize_alternatives(["commit", "git:commit"], QUALIFIED_CATALOG),
            ["git:commit"],
        )

    def test_caps_spoken_alternatives_at_three(self) -> None:
        raw = ["kiln:kiln-fire", "git:commit", "web:deploy", "api:deploy", "mytool"]
        self.assertEqual(
            intent.sanitize_alternatives(raw, QUALIFIED_CATALOG),
            ["kiln:kiln-fire", "git:commit", "web:deploy"],
        )


if __name__ == "__main__":
    unittest.main()
