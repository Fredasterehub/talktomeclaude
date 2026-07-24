"""Tests for the production-backed companion headless recovery surface."""

from __future__ import annotations

import unittest
from collections.abc import Callable
from unittest.mock import patch

from click.testing import CliRunner

from talktomeclaude.cli import main
from talktomeclaude.companion import viewmodel
from talktomeclaude.companion.contracts import (
    CompanionIntent,
    CompanionSnapshot,
    IntentKind,
)
from talktomeclaude.companion.headless import (
    HeadlessCompanionApplication,
    HeadlessController,
    IntentUnavailableError,
    run_headless,
)
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


class _ProductionController:
    def __init__(self) -> None:
        self._snapshot = CompanionSnapshot(RuntimeState(), "production ready")
        self.listeners: list[Callable[[CompanionSnapshot], None]] = []
        self.intents: list[CompanionIntent] = []
        self.background_starts = 0

    @property
    def snapshot(self) -> CompanionSnapshot:
        return self._snapshot

    def subscribe(
        self, listener: Callable[[CompanionSnapshot], None]
    ) -> Callable[[], None]:
        self.listeners.append(listener)

        def unsubscribe() -> None:
            self.listeners.remove(listener)

        return unsubscribe

    def start_background(self) -> None:
        self.background_starts += 1
        self._publish()

    def dispatch(self, intent: CompanionIntent) -> CompanionSnapshot:
        self.intents.append(intent)
        phase = self._snapshot.runtime.phase
        muted = self._snapshot.output_muted
        detail = self._snapshot.detail
        if intent.kind is IntentKind.START_RECORDING:
            phase, detail = RuntimePhase.RECORDING, "microphone active"
        elif intent.kind is IntentKind.FINISH_RECORDING:
            phase, detail = RuntimePhase.TRANSCRIBING, "transcribing locally"
        elif intent.kind is IntentKind.CANCEL:
            phase, detail = RuntimePhase.IDLE, "cancelled"
        elif intent.kind is IntentKind.TOGGLE_OUTPUT_MUTE:
            muted, detail = not muted, "spoken output changed"
        elif intent.kind is IntentKind.QUIT:
            phase, detail = RuntimePhase.STOPPING, "stopping companion"
        self._snapshot = CompanionSnapshot(RuntimeState(phase), detail, muted)
        self._publish()
        return self._snapshot

    def _publish(self) -> None:
        for listener in tuple(self.listeners):
            listener(self._snapshot)


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

    def test_headless_controller_delegates_to_production_controller(self) -> None:
        production = _ProductionController()
        controller = HeadlessController(production)

        current = controller.dispatch(CompanionIntent(IntentKind.STATUS))
        recording = controller.dispatch(CompanionIntent(IntentKind.START_RECORDING))
        transcribing = controller.dispatch(CompanionIntent(IntentKind.FINISH_RECORDING))
        idle = controller.dispatch(CompanionIntent(IntentKind.CANCEL))

        self.assertEqual(current.runtime.phase, RuntimePhase.IDLE)
        self.assertEqual(recording.runtime.phase, RuntimePhase.RECORDING)
        self.assertEqual(transcribing.runtime.phase, RuntimePhase.TRANSCRIBING)
        self.assertEqual(idle.runtime.phase, RuntimePhase.IDLE)
        self.assertEqual(
            [intent.kind for intent in production.intents],
            [
                IntentKind.STATUS,
                IntentKind.START_RECORDING,
                IntentKind.FINISH_RECORDING,
                IntentKind.CANCEL,
            ],
        )

    def test_desktop_only_surface_is_rejected_without_dispatch(self) -> None:
        production = _ProductionController()
        controller = HeadlessController(production)

        with self.assertRaises(IntentUnavailableError):
            controller.dispatch(
                CompanionIntent(IntentKind.OPEN_SETTINGS, allow_focus=True)
            )

        self.assertEqual(production.intents, [])
        self.assertEqual(controller.snapshot.runtime, RuntimeState())


class HeadlessApplicationTests(unittest.TestCase):
    def test_hosts_background_and_exercises_production_workflow_without_gui(
        self,
    ) -> None:
        production = _ProductionController()
        commands = iter(["status", "start", "finish", "cancel", "mute", "quit"])
        output: list[str] = []
        application = HeadlessCompanionApplication(
            HeadlessController(production),
            read=lambda: next(commands, None),
            write=output.append,
        )

        self.assertEqual(application.run(), 0)

        self.assertEqual(production.background_starts, 1)
        self.assertEqual(production.listeners, [])
        self.assertIn("IDLE: Companion ready — production ready", output)
        self.assertTrue(any(line.startswith("RECORDING: Recording") for line in output))
        self.assertTrue(
            any(line.startswith("TRANSCRIBING: Transcribing speech") for line in output)
        )
        self.assertTrue(any("[MUTED]" in line for line in output))
        self.assertEqual(production.intents[-1].kind, IntentKind.QUIT)

    def test_eof_still_quits_production_controller_and_prints_no_content(self) -> None:
        production = _ProductionController()
        production._snapshot = CompanionSnapshot(
            RuntimeState(),
            "safe status",
        )
        output: list[str] = []

        result = run_headless(output.append, production, read=lambda: None)

        self.assertEqual(result, 0)
        self.assertEqual(production.intents, [CompanionIntent(IntentKind.QUIT)])
        self.assertEqual(output, ["IDLE: Companion ready — safe status"])

    def test_action_failure_and_unknown_command_do_not_expose_exception_text(
        self,
    ) -> None:
        class FailingController(_ProductionController):
            def dispatch(self, intent: CompanionIntent) -> CompanionSnapshot:
                if intent.kind is IntentKind.START_RECORDING:
                    raise RuntimeError("SECRET transcript and path")
                return super().dispatch(intent)

        production = FailingController()
        commands = iter(["start", "not-a-command", "quit"])
        output: list[str] = []

        HeadlessCompanionApplication(
            HeadlessController(production),
            read=lambda: next(commands, None),
            write=output.append,
        ).run()

        self.assertIn("ERROR: Action unavailable", output)
        self.assertIn("ERROR: Unknown command", output)
        self.assertNotIn("SECRET", repr(output))


class CompanionCliTests(unittest.TestCase):
    def setUp(self) -> None:
        self.runner = CliRunner()

    def test_explicit_headless_command_uses_production_headless_entrypoint(
        self,
    ) -> None:
        with patch(
            "talktomeclaude.companion.headless.run_headless", return_value=0
        ) as launch:
            result = self.runner.invoke(main, ["companion", "--headless"])

        self.assertEqual(result.exit_code, 0, result.output)
        launch.assert_called_once()

    def test_desktop_command_uses_production_companion_entrypoint(self) -> None:
        with patch(
            "talktomeclaude.companion.app.run_desktop_companion", return_value=0
        ) as launch:
            result = self.runner.invoke(main, ["companion"])

        self.assertEqual(result.exit_code, 0, result.output)
        launch.assert_called_once_with()

    def test_ui_and_tui_keep_launching_the_legacy_dashboard(self) -> None:
        for command in ("ui", "tui"):
            with (
                self.subTest(command=command),
                patch("talktomeclaude.cli._launch_dashboard") as launch,
            ):
                result = self.runner.invoke(main, [command])
                self.assertEqual(result.exit_code, 0, result.output)
                launch.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
