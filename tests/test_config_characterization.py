"""Behavior locks for the legacy single-file configuration helpers."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from talktomeclaude import config


class ConfigCharacterizationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.env = mock.patch.dict(
            os.environ, {"TALKTOMECLAUDE_CONFIG_DIR": self.tmp.name}, clear=False
        )
        self.env.start()
        self.addCleanup(self.env.stop)
        self.path = Path(self.tmp.name) / "config.json"

    def test_fresh_install_defaults_are_stable(self) -> None:
        self.assertEqual(config.recording_mode(), "push-to-talk")
        self.assertTrue(config.voice_assist_enabled())
        self.assertIsNone(config.remote())
        self.assertIsNone(config.remote_cwd())
        self.assertIsNone(config.default_voice_name())
        self.assertEqual(config.onboarding_version(), 0)

    def test_malformed_file_fails_to_safe_accessor_defaults(self) -> None:
        self.path.write_text('{"recording-mode":', encoding="utf-8")

        self.assertEqual(config.load(), {})
        self.assertEqual(config.recording_mode(), "push-to-talk")
        self.assertTrue(config.voice_assist_enabled())
        self.assertIsNone(config.remote())
        self.assertIsNone(config.default_voice_name())
        self.assertEqual(config.onboarding_version(), 0)

    def test_known_updates_preserve_unknown_and_independent_values(self) -> None:
        seeded = {
            "future-setting": {"nested": [1, 2, 3]},
            "recording-mode": "push-toggle",
            "remote": "dev@example",
            "remote-cwd": "/srv/Claude Projects/main",
            "default-voice": "rick",
            "onboarding-version": 7,
        }
        self.path.write_text(json.dumps(seeded), encoding="utf-8")

        config.set_voice_assist(False)

        saved = json.loads(self.path.read_text(encoding="utf-8"))
        self.assertEqual(saved["future-setting"], {"nested": [1, 2, 3]})
        self.assertEqual(config.recording_mode(), "push-toggle")
        self.assertEqual(config.remote(), "dev@example")
        self.assertEqual(config.remote_cwd(), "/srv/Claude Projects/main")
        self.assertEqual(config.default_voice_name(), "rick")
        self.assertEqual(config.onboarding_version(), 7)
        self.assertFalse(config.voice_assist_enabled())


if __name__ == "__main__":
    unittest.main()
