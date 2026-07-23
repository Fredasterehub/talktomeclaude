"""Tests for the staged companion headless recovery surface."""

from __future__ import annotations

import unittest
from unittest.mock import patch

from click.testing import CliRunner

from talktomeclaude.cli import main
from talktomeclaude.companion.contracts import (
    CompanionIntent,
    CompanionSnapshot,
    IntentKind,
)
from talktomeclaude.companion.headless import HeadlessController, IntentUnavailableError
from talktomeclaude.companion import viewmodel
from talktomeclaude.core import RuntimePhase, RuntimeState


def _state(phase: RuntimePhase) -> RuntimeState:
    if phase is RuntimePhase.DISCONNECTED:
        return RuntimeState(phase, resume_phase=RuntimePhase.WAITING_FOR_CLAUDE)
    if phase is RuntimePhase.RECOVERABLE_ERROR:
        return RuntimeState(
            phase,
            resume_phase=RuntimePhase.IDLE,
            error_code="synthetic_failure",
        )
    return RuntimeState(phase)


class CompanionContractTests(unittest.TestCase):
    def test_presentation_mapping_is_exhaustive_over_runtime_phases(self) -> None:
        self.assertEqual(set(RuntimePhase), set(viewmodel._PRESENTATION))

    def test_every_state_has_a_non_color_focus_free_view_model(self) -> None:
        for phase in RuntimePhase:
            with self.subTest(phase=phase):
                view = viewmodel.to_view_model(
                    CompanionSnapshot(_state(phase), "safe detail")
                )
                self.assertTrue(view.cue)
                self.assertTrue(view.status)
                self.assertEqual(view.detail, "safe detail")
                self.assertFalse(view.focus_requested)

    def test_headless_controller_uses_core_events_for_workflow_intents(self) -> None:
        controller = HeadlessController()

        current = controller.dispatch(CompanionIntent(IntentKind.STATUS))
        recording = controller.dispatch(CompanionIntent(IntentKind.START_RECORDING))
        transcribing = controller.dispatch(CompanionIntent(IntentKind.FINISH_RECORDING))
        idle = controller.dispatch(CompanionIntent(IntentKind.CANCEL))

        self.assertEqual(current.runtime.phase, RuntimePhase.IDLE)
        self.assertEqual(recording.runtime.phase, RuntimePhase.RECORDING)
        self.assertEqual(transcribing.runtime.phase, RuntimePhase.TRANSCRIBING)
        self.assertEqual(idle.runtime.phase, RuntimePhase.IDLE)

    def test_quit_requests_core_shutdown_and_exposes_stopping(self) -> None:
        controller = HeadlessController()

        stopping = controller.dispatch(CompanionIntent(IntentKind.QUIT))

        self.assertEqual(stopping.runtime.phase, RuntimePhase.STOPPING)

    def test_unwired_workflow_intent_fails_without_changing_state(self) -> None:
        controller = HeadlessController()

        with self.assertRaises(IntentUnavailableError):
            controller.dispatch(CompanionIntent(IntentKind.FINISH_RECORDING))

        self.assertEqual(controller.snapshot.runtime, RuntimeState())

    def test_presentation_only_settings_intent_does_not_mutate_runtime(self) -> None:
        controller = HeadlessController()

        with self.assertRaises(IntentUnavailableError):
            controller.dispatch(CompanionIntent(IntentKind.OPEN_SETTINGS))

        self.assertEqual(controller.snapshot.runtime, RuntimeState())


class CompanionCliTests(unittest.TestCase):
    def setUp(self) -> None:
        self.runner = CliRunner()

    def test_explicit_headless_command_reports_ready_and_exits(self) -> None:
        result = self.runner.invoke(main, ["companion", "--headless"])

        self.assertEqual(result.exit_code, 0, result.output)
        self.assertEqual(result.output, "IDLE: Companion ready\n")

    def test_desktop_command_is_explicitly_unavailable_during_staging(self) -> None:
        result = self.runner.invoke(main, ["companion"])

        self.assertNotEqual(result.exit_code, 0)
        self.assertIn("use --headless for recovery", result.output)

    def test_ui_and_tui_keep_launching_the_legacy_dashboard(self) -> None:
        for command in ("ui", "tui"):
            with self.subTest(command=command), patch(
                "talktomeclaude.cli._launch_dashboard"
            ) as launch:
                result = self.runner.invoke(main, [command])
                self.assertEqual(result.exit_code, 0, result.output)
                launch.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
