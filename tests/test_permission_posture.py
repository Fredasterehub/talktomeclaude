"""Tests for the operator's claude-permissions posture: config persistence
and its threading into the extracted Claude command builder."""

from __future__ import annotations

import os
import tempfile
import unittest
from unittest import mock

from talktomeclaude import config
from talktomeclaude.listen import build_claude_command


class PermissionConfigTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.env = mock.patch.dict(
            os.environ, {"TALKTOMECLAUDE_CONFIG_DIR": self.tmp.name}, clear=False
        )
        self.env.start()
        self.addCleanup(self.env.stop)

    def test_defaults_off_and_round_trips_each_value(self) -> None:
        self.assertEqual(config.claude_permissions(), "off")
        for value in ("skip", "acceptEdits", "bypassPermissions", "off"):
            config.set_claude_permissions(value)
            self.assertEqual(config.claude_permissions(), value)

    def test_unknown_value_rejected(self) -> None:
        with self.assertRaises(ValueError):
            config.set_claude_permissions("banana")


class BuildClaudeCommandPermissionTests(unittest.TestCase):
    def test_off_adds_no_flags(self) -> None:
        argv = build_claude_command("hi", None, permission="off")
        self.assertNotIn("--dangerously-skip-permissions", argv)
        self.assertNotIn("--permission-mode", argv)

    def test_skip_adds_dangerous_flag_locally(self) -> None:
        argv = build_claude_command("hi", None, permission="skip")
        self.assertIn("--dangerously-skip-permissions", argv)

    def test_accept_edits_adds_permission_mode_locally(self) -> None:
        argv = build_claude_command("hi", None, permission="acceptEdits")
        self.assertIn("--permission-mode", argv)
        self.assertIn("acceptEdits", argv)

    def test_bypass_permissions_threads_into_remote_shell_string(self) -> None:
        joined = " ".join(
            build_claude_command(
                "hi", None, remote="u@h", permission="bypassPermissions"
            )
        )
        self.assertIn("--permission-mode", joined)
        self.assertIn("bypassPermissions", joined)

    def test_skip_threads_into_remote_shell_string(self) -> None:
        joined = " ".join(
            build_claude_command("hi", None, remote="u@h", permission="skip")
        )
        self.assertIn("--dangerously-skip-permissions", joined)
