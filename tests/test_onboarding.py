"""Tests for the onboarding version bookkeeping and the first-run wizard screen."""

from __future__ import annotations

import os
import tempfile
import unittest
from unittest import mock

from click.testing import CliRunner

from talktomeclaude import cli, config


class OnboardingConfigTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.env = mock.patch.dict(
            os.environ, {"TALKTOMECLAUDE_CONFIG_DIR": self.tmp.name}, clear=False
        )
        self.env.start()
        self.addCleanup(self.env.stop)

    def test_version_defaults_to_zero_and_round_trips(self) -> None:
        self.assertEqual(config.onboarding_version(), 0)
        config.set_onboarding_version(3)
        self.assertEqual(config.onboarding_version(), 3)

    def test_needed_compares_stored_against_current(self) -> None:
        config.set_onboarding_version(0)
        self.assertTrue(config.onboarding_needed(1))
        config.set_onboarding_version(5)
        self.assertFalse(config.onboarding_needed(1))
        self.assertTrue(config.onboarding_needed(6))


class SetupCommandTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.env = mock.patch.dict(
            os.environ, {"TALKTOMECLAUDE_CONFIG_DIR": self.tmp.name}, clear=False
        )
        self.env.start()
        self.addCleanup(self.env.stop)
        self.runner = CliRunner()

    def test_setup_help_documents_reset_and_force(self) -> None:
        result = self.runner.invoke(cli.main, ["setup", "--help"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("reset", result.output.lower())
        self.assertIn("force", result.output.lower())

    def test_subcommands_never_gated_by_onboarding(self) -> None:
        # A fresh install (onboarding version 0) must not block an unrelated
        # subcommand with the wizard — only the bare dashboard launch is gated.
        self.assertEqual(config.onboarding_version(), 0)
        result = self.runner.invoke(cli.main, ["config", "get", "recording-mode"])
        self.assertEqual(result.exit_code, 0)
        self.assertEqual(config.onboarding_version(), 0)  # untouched


class OnboardingScreenTests(unittest.IsolatedAsyncioTestCase):
    async def test_is_a_textual_screen(self) -> None:
        from textual.screen import Screen

        from talktomeclaude.onboarding import OnboardingScreen

        self.assertTrue(issubclass(OnboardingScreen, Screen))

    async def test_defaults_fast_path_completes_immediately(self) -> None:
        from textual.app import App, ComposeResult

        from talktomeclaude.onboarding import OnboardingScreen

        result: dict = {}

        class _Host(App[None]):
            def compose(self) -> ComposeResult:
                return
                yield

            def on_mount(self) -> None:
                self.push_screen(OnboardingScreen(), lambda value: result.setdefault("done", value))

        app = _Host()
        async with app.run_test() as pilot:
            await pilot.pause()
            await pilot.press("enter")
            await pilot.pause()
            self.assertIn("done", result)

    async def test_customize_walks_the_guided_sequence_and_persists(self) -> None:
        import tempfile

        from textual.app import App, ComposeResult

        from talktomeclaude.onboarding import OnboardingScreen

        with tempfile.TemporaryDirectory() as directory:
            with mock.patch.dict(
                os.environ, {"TALKTOMECLAUDE_CONFIG_DIR": directory}, clear=False
            ):
                result: dict = {}
                screen = OnboardingScreen()

                class _Host(App[None]):
                    def compose(self) -> ComposeResult:
                        return
                        yield

                    def on_mount(self) -> None:
                        self.push_screen(
                            screen, lambda value: result.setdefault("done", value)
                        )

                async with _Host().run_test() as pilot:
                    await pilot.pause()
                    await pilot.press("down", "enter")  # Customize step by step
                    await pilot.pause()
                    self.assertEqual(screen._step, "hardware")
                    await pilot.press("enter")  # Continue
                    await pilot.pause()
                    self.assertEqual(screen._step, "claude")
                    await pilot.press("enter")  # Local Claude Code
                    await pilot.pause()
                    self.assertEqual(screen._step, "voice")
                    await pilot.press("enter")  # Auto (recommended)
                    await pilot.pause()
                    self.assertEqual(screen._step, "spoken")
                    await pilot.press("s")  # skippable pane
                    await pilot.pause()
                    self.assertEqual(screen._step, "recording")
                    await pilot.press("enter")  # push-to-talk default
                    await pilot.pause()
                    self.assertEqual(screen._step, "wake")
                    await pilot.press("down", "enter")  # enable the wake word
                    await pilot.pause()
                    self.assertEqual(screen._step, "wake-phrase")
                    await pilot.press("enter")  # keep the default phrase
                    await pilot.pause()
                    self.assertEqual(screen._step, "permissions")
                    await pilot.press("enter")  # off
                    await pilot.pause()
                    self.assertEqual(screen._step, "finish")
                    self.assertEqual(config.onboarding_version(), 0)  # not yet
                    await pilot.press("enter")  # Finish
                    await pilot.pause()

                self.assertIs(result.get("done"), True)
                self.assertEqual(config.recording_mode(), "push-to-talk")
                self.assertTrue(config.wake_word_enabled())
                self.assertEqual(config.claude_permissions(), "off")
                self.assertGreaterEqual(config.onboarding_version(), 1)
                self.assertIsNotNone(config.onboarding_completed_at())


if __name__ == "__main__":
    unittest.main()
