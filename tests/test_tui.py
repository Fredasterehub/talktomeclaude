"""Tests for the dependency-free terminal dashboard."""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest import mock

from click.testing import CliRunner

from talktomeclaude import cli, listen, tui


class DashboardRenderTests(unittest.TestCase):
    def state(self, **overrides) -> tui.DashboardState:
        values = {
            "remote": "dev@example",
            "remote_cwd": "/DEV/ghostundo",
            "mode": "push-toggle",
            "voice_enabled": True,
        }
        values.update(overrides)
        return tui.DashboardState(**values)

    def test_dashboard_has_stable_dimensions_and_required_context(self) -> None:
        state = self.state()
        state.add_dialogue("You", "Please inspect the project.")
        state.add_dialogue("Claude", "I am checking it now.")

        ready = tui.render_dashboard(state, width=72, height=22)
        state.phase = "recording"
        state.add_level(0.04)
        recording = tui.render_dashboard(state, width=72, height=22)

        self.assertEqual(len(ready.splitlines()), 22)
        self.assertEqual(len(recording.splitlines()), 22)
        self.assertTrue(all(len(line) <= 72 for line in recording.splitlines()))
        self.assertIn("GOAL", recording)
        self.assertIn("MODE", recording)
        self.assertIn("SOURCE", recording)
        self.assertIn("KEYS", recording)
        self.assertIn("/DEV/ghostundo", recording)
        self.assertIn("RECORDING", recording)

    def test_compact_dashboard_keeps_controls_visible(self) -> None:
        state = self.state()
        state.add_dialogue("You", "A wrapped message that needs several compact lines.")
        state.add_dialogue("Claude", "The latest reply remains labelled.")

        canvas = tui.render_dashboard(state, width=40, height=16)

        self.assertEqual(len(canvas.splitlines()), 16)
        self.assertIn("KEYS", canvas)
        self.assertIn("Q Quit", canvas)
        self.assertIn("CLAUDE", canvas)
        self.assertTrue(all(len(line) <= 40 for line in canvas.splitlines()))

    def test_reduced_motion_uses_static_signal(self) -> None:
        state = self.state(reduced_motion=True)

        canvas = tui.render_dashboard(state, width=60, height=18)

        self.assertIn("-" * 50, canvas)

    def test_project_picker_marks_current_directory(self) -> None:
        state = self.state()
        projects = ["/DEV/another", "/DEV/ghostundo", "/DEV/kiln"]

        canvas = tui.render_project_picker(state, projects, selected=1, width=70, height=20)

        self.assertIn("> /DEV/ghostundo  current", canvas)
        self.assertIn("Enter Select", canvas)


class RemoteProjectTests(unittest.TestCase):
    def test_project_discovery_returns_sorted_unique_paths(self) -> None:
        completed = SimpleNamespace(
            returncode=0,
            stdout="/DEV/zeta\n/DEV/Alpha\n/DEV/zeta\n",
            stderr="",
        )

        with mock.patch.object(tui, "_ssh_base", return_value=["ssh", "dev@example"]), mock.patch.object(
            tui, "_remote_shell_command", side_effect=lambda command: command
        ), mock.patch.object(tui.subprocess, "run", return_value=completed) as run:
            projects = tui.discover_remote_projects("dev@example", "/DEV")

        self.assertEqual(projects, ["/DEV/Alpha", "/DEV/zeta"])
        command = run.call_args.args[0]
        self.assertEqual(command[:2], ["ssh", "dev@example"])
        self.assertIn("find -- /DEV", command[-1])
        self.assertEqual(run.call_args.kwargs["encoding"], "utf-8")
        self.assertEqual(run.call_args.kwargs["errors"], "replace")

    def test_project_discovery_reports_ssh_failure(self) -> None:
        completed = SimpleNamespace(returncode=255, stdout="", stderr="connection failed")

        with mock.patch.object(tui.subprocess, "run", return_value=completed):
            with self.assertRaisesRegex(tui.TUIError, "connection failed"):
                tui.discover_remote_projects("dev@example")


class LiveSignalTests(unittest.TestCase):
    def test_space_toggle_reports_recording_and_audio_level(self) -> None:
        keys = mock.Mock()
        keys.read_key.side_effect = [" ", " "]
        block = mock.Mock()
        block.copy.return_value = block
        stream = mock.MagicMock()
        stream.__enter__.return_value.read.return_value = (block, False)
        sounddevice = mock.Mock()
        sounddevice.InputStream.return_value = stream
        levels: list[float] = []
        recording = mock.Mock()

        with mock.patch.object(listen, "_sounddevice", return_value=sounddevice), mock.patch.object(
            listen, "_rms", return_value=0.125
        ), mock.patch.object(listen, "_finish", return_value="audio"):
            result = listen._record_push_toggle(
                keys,
                trigger_key=" ",
                on_level=levels.append,
                on_recording=recording,
            )

        self.assertEqual(result, "audio")
        self.assertEqual(levels, [0.125])
        recording.assert_called_once_with()

    def test_trigger_wait_ignores_non_matching_keys(self) -> None:
        keys = mock.Mock()
        keys.read_key.side_effect = ["p", " "]

        self.assertEqual(listen._wait_for_trigger(keys, " "), " ")

    def test_immediate_toggle_uses_dashboard_space_as_start(self) -> None:
        keys = mock.Mock()
        keys.read_key.return_value = " "
        block = mock.Mock()
        block.copy.return_value = block
        stream = mock.MagicMock()
        stream.__enter__.return_value.read.return_value = (block, False)
        sounddevice = mock.Mock()
        sounddevice.InputStream.return_value = stream

        with mock.patch.object(listen, "_sounddevice", return_value=sounddevice), mock.patch.object(
            listen, "_rms", return_value=0.1
        ), mock.patch.object(listen, "_finish", return_value="audio"):
            result = listen._record_push_toggle(
                keys,
                trigger_key=" ",
                start_immediately=True,
            )

        self.assertEqual(result, "audio")
        keys.read_key.assert_called_once_with(0)


class DashboardCLITests(unittest.TestCase):
    def test_no_arguments_launches_dashboard(self) -> None:
        runner = CliRunner()

        with mock.patch.object(cli, "_launch_dashboard") as launch:
            result = runner.invoke(cli.main, [])

        self.assertEqual(result.exit_code, 0, result.output)
        launch.assert_called_once_with()

    def test_ui_command_launches_dashboard(self) -> None:
        runner = CliRunner()

        with mock.patch.object(cli, "_launch_dashboard") as launch:
            result = runner.invoke(cli.main, ["ui"])

        self.assertEqual(result.exit_code, 0, result.output)
        launch.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
