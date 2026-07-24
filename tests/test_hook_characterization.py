"""Characterization locks for the existing Claude Code Stop hook."""

from __future__ import annotations

import io
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from click.testing import CliRunner

from talktomeclaude import config
from talktomeclaude.cli import main
from talktomeclaude.hook import read_stop_event, stop_dialogue
from talktomeclaude.tts import TTSError


class HookCharacterizationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.env = mock.patch.dict(
            os.environ, {"TALKTOMECLAUDE_CONFIG_DIR": self.tmp.name}, clear=False
        )
        self.env.start()
        self.addCleanup(self.env.stop)
        self.runner = CliRunner()

    def test_stop_event_parser_accepts_only_json_objects(self) -> None:
        event = read_stop_event(io.StringIO('{"hook_event_name":"Stop","session_id":"s"}'))
        self.assertEqual(event, {"hook_event_name": "Stop", "session_id": "s"})
        for payload in ("", "not json", "[]", '"text"'):
            with self.subTest(payload=payload):
                self.assertIsNone(read_stop_event(io.StringIO(payload)))

    def test_stop_event_parser_bounds_original_stdin_before_json_decode(self) -> None:
        payload = '{"last_assistant_message":"' + ("x" * 64) + '"}'

        self.assertIsNone(read_stop_event(io.StringIO(payload), max_bytes=32))

    def test_dialogue_uses_last_assistant_message_not_transcript_path(self) -> None:
        event = {
            "last_assistant_message": "Café ☕ is ready. Run `unsafe()` only onscreen.",
            "transcript_path": "C:/must/not/be/read.jsonl",
        }

        dialogue = stop_dialogue(event)

        self.assertEqual(dialogue, "Café ☕ is ready. Run only onscreen.")
        self.assertNotIn("transcript", dialogue)

    def test_missing_or_non_string_message_is_silent(self) -> None:
        for event in ({}, {"last_assistant_message": None}, {"last_assistant_message": []}):
            with self.subTest(event=event):
                self.assertEqual(stop_dialogue(event), "")

    def test_dry_run_prints_speakable_unicode_and_exits_zero(self) -> None:
        payload = '{"last_assistant_message":"Hello café ☕.\\nSecond line."}'

        result = self.runner.invoke(main, ["hook", "stop", "--dry-run"], input=payload)

        self.assertEqual(result.exit_code, 0, result.output)
        self.assertEqual(result.output, "SPEAK: Hello café ☕. Second line.\n")

    def test_muted_and_malformed_events_exit_zero_without_synthesis(self) -> None:
        config.set_voice_assist(False)
        with mock.patch("talktomeclaude.cli.synthesize") as synth:
            muted = self.runner.invoke(
                main,
                ["hook", "stop"],
                input='{"last_assistant_message":"should stay silent"}',
            )
            malformed = self.runner.invoke(main, ["hook", "stop"], input="not json")

        self.assertEqual(muted.exit_code, 0, muted.output)
        self.assertEqual(malformed.exit_code, 0, malformed.output)
        synth.assert_not_called()

    def test_synthesis_failure_still_exits_zero_and_cleans_temp_file(self) -> None:
        wav_path = Path(self.tmp.name) / "hook.wav"
        wav_path.write_bytes(b"")
        with mock.patch(
            "talktomeclaude.cli._temporary_wav_path", return_value=wav_path
        ), mock.patch(
            "talktomeclaude.cli.synthesize", side_effect=TTSError("engine unavailable")
        ):
            result = self.runner.invoke(
                main,
                ["hook", "stop"],
                input='{"last_assistant_message":"speak me"}',
            )

        self.assertEqual(result.exit_code, 0, result.output)
        self.assertFalse(wav_path.exists())


if __name__ == "__main__":
    unittest.main()
