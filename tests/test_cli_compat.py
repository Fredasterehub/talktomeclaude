"""Characterization locks for the pre-companion command-line surface."""

from __future__ import annotations

import unittest

from click.testing import CliRunner

from talktomeclaude.cli import main


class CliCompatibilityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.runner = CliRunner()

    def test_top_level_help_keeps_existing_entry_points(self) -> None:
        result = self.runner.invoke(main, ["--help"])

        self.assertEqual(result.exit_code, 0, result.output)
        for command in ("setup", "config", "voice", "voices", "listen", "hook", "ui"):
            self.assertIn(command, result.output)

    def test_nested_entry_point_help_exits_successfully(self) -> None:
        for arguments in (
            ["setup", "--help"],
            ["config", "--help"],
            ["voice", "--help"],
            ["voices", "--help"],
            ["listen", "--help"],
            ["hook", "--help"],
        ):
            with self.subTest(arguments=arguments):
                result = self.runner.invoke(main, arguments)
                self.assertEqual(result.exit_code, 0, result.output)
                self.assertIn("Usage:", result.output)


if __name__ == "__main__":
    unittest.main()
