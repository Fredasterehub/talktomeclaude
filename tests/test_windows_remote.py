"""Focused cross-platform tests for terminal input and remote execution."""

from __future__ import annotations

import importlib
import json
import os
import shlex
import sys
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from click.testing import CliRunner

from talktomeclaude import config
from talktomeclaude.cli import main
from talktomeclaude import listen
from talktomeclaude.stt import CPU_TIER, GPU_TIER


class WindowsTerminalTests(unittest.TestCase):
    def test_windows_import_does_not_require_posix_terminal_modules(self) -> None:
        real_import = __import__

        def guarded_import(name, *args, **kwargs):
            if name in {"select", "termios", "tty"}:
                raise AssertionError(f"POSIX-only import attempted on Windows: {name}")
            return real_import(name, *args, **kwargs)

        with mock.patch.object(os, "name", "nt"), mock.patch(
            "builtins.__import__", side_effect=guarded_import
        ):
            reloaded = importlib.reload(listen)
            self.assertTrue(reloaded._is_windows())

        # Restore module-level platform-dependent state for the rest of this process.
        importlib.reload(listen)

    def test_windows_raw_keys_reads_without_posix_file_descriptor(self) -> None:
        console = mock.Mock()
        console.kbhit.return_value = True
        console.getwch.return_value = "k"
        stdin = mock.Mock()
        stdin.isatty.return_value = True

        with mock.patch.object(os, "name", "nt"), mock.patch.object(
            sys, "stdin", stdin
        ), mock.patch.dict(sys.modules, {"msvcrt": console}):
            with listen._RawKeys() as keys:
                self.assertEqual(keys.read_key(0), "k")

        stdin.fileno.assert_not_called()
        console.kbhit.assert_called_once_with()
        console.getwch.assert_called_once_with()

    def test_windows_raw_keys_raises_keyboard_interrupt_for_ctrl_c(self) -> None:
        console = mock.Mock()
        console.kbhit.return_value = True
        console.getwch.return_value = "\x03"
        stdin = mock.Mock()
        stdin.isatty.return_value = True

        with mock.patch.object(os, "name", "nt"), mock.patch.object(
            sys, "stdin", stdin
        ), mock.patch.dict(sys.modules, {"msvcrt": console}):
            keys = listen._RawKeys()
            with self.assertRaises(KeyboardInterrupt):
                keys.read_key(0)

    def test_windows_raw_keys_uses_physical_key_state(self) -> None:
        import ctypes

        console = mock.Mock()
        console.kbhit.return_value = True
        console.getwch.return_value = "k"
        stdin = mock.Mock()
        stdin.isatty.return_value = True
        user32 = mock.Mock()
        user32.VkKeyScanW.return_value = ord("K")
        user32.GetAsyncKeyState.return_value = 0x8000

        with mock.patch.object(os, "name", "nt"), mock.patch.object(
            sys, "stdin", stdin
        ), mock.patch.dict(sys.modules, {"msvcrt": console}), mock.patch.object(
            ctypes, "windll", SimpleNamespace(user32=user32), create=True
        ):
            keys = listen._RawKeys()
            self.assertTrue(keys.is_pressed("k"))

        user32.VkKeyScanW.assert_called_once_with("k")
        user32.GetAsyncKeyState.assert_called_once_with(ord("K"))
        self.assertEqual(user32.VkKeyScanW.argtypes, [ctypes.c_wchar])
        self.assertIs(user32.VkKeyScanW.restype, ctypes.c_short)
        self.assertEqual(user32.GetAsyncKeyState.argtypes, [ctypes.c_int])
        self.assertIs(user32.GetAsyncKeyState.restype, ctypes.c_short)

    def test_windows_extended_key_preserves_e0_scan_prefix(self) -> None:
        import ctypes

        stdin = mock.Mock()
        stdin.isatty.return_value = True
        user32 = mock.Mock()
        user32.MapVirtualKeyW.return_value = 0x26  # VK_UP
        user32.GetAsyncKeyState.return_value = 0x8000

        with mock.patch.object(os, "name", "nt"), mock.patch.object(
            sys, "stdin", stdin
        ), mock.patch.object(
            ctypes, "windll", SimpleNamespace(user32=user32), create=True
        ):
            keys = listen._RawKeys()
            self.assertTrue(keys.is_pressed("\xe0H"))

        user32.MapVirtualKeyW.assert_called_once_with(0xE048, 3)
        user32.GetAsyncKeyState.assert_called_once_with(0x26)

    def test_windows_push_to_talk_stops_on_physical_key_release(self) -> None:
        keys = mock.Mock()
        keys.read_key.return_value = "k"
        keys.is_pressed.side_effect = [True, True, False]
        block = mock.Mock()
        block.copy.return_value = block
        stream = mock.MagicMock()
        stream.__enter__.return_value.read.return_value = (block, False)
        sounddevice = mock.Mock()
        sounddevice.InputStream.return_value = stream

        with mock.patch.object(listen, "_sounddevice", return_value=sounddevice), mock.patch.object(
            listen, "_finish", return_value="audio"
        ), mock.patch.object(
            listen.time, "monotonic", side_effect=[0.0, 1.0, 2.0]
        ):
            result = listen._record_push_to_talk(keys)

        self.assertEqual(result, "audio")
        keys.read_key.assert_called_once_with(None)
        self.assertEqual(keys.is_pressed.call_count, 3)
        self.assertEqual(stream.__enter__.return_value.read.call_count, 2)

    def test_always_on_does_not_open_raw_keys_on_windows(self) -> None:
        transcriber = mock.Mock()
        transcriber.transcribe.return_value = "hello"
        statuses: list[str] = []

        with mock.patch.object(os, "name", "nt"), mock.patch.object(
            listen, "UtteranceTranscriber", return_value=transcriber
        ), mock.patch.object(listen, "_RawKeys") as raw_keys, mock.patch.object(
            listen, "_record_always_on", return_value=object()
        ), mock.patch.object(
            listen, "_prompt_claude", return_value=("reply", "session-1")
        ):
            listen.run_listen(
                mode="always-on",
                session_id=None,
                tmux_pane=None,
                device="cpu",
                model=None,
                once=True,
                echo=lambda _message: None,
                speak=lambda _message: None,
                status=statuses.append,
            )

        raw_keys.assert_not_called()
        self.assertIn("listening (hands-free); Ctrl-C to stop", statuses)

    def test_posix_raw_keys_preserves_cbreak_select_behavior(self) -> None:
        stdin = mock.Mock()
        stdin.isatty.return_value = True
        stdin.fileno.return_value = 7
        termios = mock.Mock(TCSADRAIN=123)
        termios.tcgetattr.return_value = ["saved"]
        tty = mock.Mock()
        select = mock.Mock()
        select.select.return_value = ([7], [], [])

        with mock.patch.object(os, "name", "posix"), mock.patch.object(
            sys, "stdin", stdin
        ), mock.patch.dict(
            sys.modules, {"termios": termios, "tty": tty, "select": select}
        ), mock.patch.object(listen.os, "read", return_value=b"p"):
            with listen._RawKeys() as keys:
                self.assertEqual(keys.read_key(0.25), "p")

        termios.tcgetattr.assert_called_once_with(7)
        tty.setcbreak.assert_called_once_with(7)
        select.select.assert_called_once_with([7], [], [], 0.25)
        termios.tcsetattr.assert_called_once_with(7, 123, ["saved"])


class SSHCommandTests(unittest.TestCase):
    def test_posix_ssh_keeps_connection_multiplexing(self) -> None:
        with mock.patch.object(os, "name", "posix"):
            command = listen._ssh_base("dev@example")

        self.assertEqual(command[-1], "dev@example")
        self.assertIn("ControlMaster=auto", command)
        self.assertIn("ControlPersist=600", command)
        self.assertTrue(any(arg.startswith("ControlPath=") for arg in command))

    def test_windows_ssh_omits_unix_control_socket_options(self) -> None:
        with mock.patch.object(os, "name", "nt"):
            command = listen._ssh_base("dev@example")

        self.assertEqual(command, ["ssh", "-o", "ConnectTimeout=10", "dev@example"])
        self.assertFalse(any("Control" in arg for arg in command))

    def test_remote_cwd_is_shell_quoted_in_claude_command(self) -> None:
        cwd = "/srv/projects/space and 'quote'; echo unsafe"
        text = "please inspect $(uname) and 'quotes'"
        completed = SimpleNamespace(
            returncode=0,
            stdout=json.dumps({"result": "done", "session_id": "new-session"}),
            stderr="",
        )

        with mock.patch.object(listen.subprocess, "run", return_value=completed) as run:
            result = listen._prompt_claude(
                text,
                "old-session",
                remote="dev@example",
                remote_cwd=cwd,
            )

        inner = (
            f"cd -- {shlex.quote(cwd)} && "
            f"claude -p {shlex.quote(text)} --output-format json "
            f"--resume {shlex.quote('old-session')}"
        )
        expected_remote_command = f"bash -lc {shlex.quote(inner)}"
        command = run.call_args.args[0]
        self.assertEqual(command[-1], expected_remote_command)
        self.assertEqual(result, ("done", "new-session"))

    def test_prompt_claude_reports_missing_captured_output(self) -> None:
        completed = SimpleNamespace(returncode=0, stdout=None, stderr=None)

        with mock.patch.object(listen.subprocess, "run", return_value=completed):
            with self.assertRaisesRegex(listen.ListenError, "returned no JSON output"):
                listen._prompt_claude("hello", None, remote="dev@example")

    def test_prompt_claude_reports_progress_while_process_runs(self) -> None:
        process = mock.Mock(returncode=0)

        def communicate():
            time.sleep(0.03)
            return json.dumps({"result": "done", "session_id": "session-1"}), ""

        process.communicate.side_effect = communicate
        progress = mock.Mock()

        with mock.patch.object(listen.subprocess, "Popen", return_value=process) as popen:
            result = listen._prompt_claude(
                "hello",
                None,
                remote="dev@example",
                on_wait=progress,
            )

        self.assertEqual(result, ("done", "session-1"))
        progress.assert_called()
        self.assertIs(popen.call_args.kwargs["stdin"], listen.subprocess.DEVNULL)
        self.assertIs(popen.call_args.kwargs["stdout"], listen.subprocess.PIPE)
        self.assertEqual(popen.call_args.kwargs["encoding"], "utf-8")
        self.assertEqual(popen.call_args.kwargs["errors"], "replace")


class TranscriberFallbackTests(unittest.TestCase):
    def test_live_auto_cuda_decode_failure_falls_back_to_cpu(self) -> None:
        gpu = mock.Mock()
        gpu.transcribe.side_effect = RuntimeError("cublas64_12.dll is not found")
        cpu = mock.Mock()
        cpu.transcribe.return_value = ([SimpleNamespace(text=" recovered ")], None)
        statuses: list[str] = []

        with mock.patch.object(
            listen, "detect_tier", side_effect=[GPU_TIER, CPU_TIER]
        ), mock.patch.object(
            listen.UtteranceTranscriber, "_load", side_effect=[gpu, cpu]
        ):
            transcriber = listen.UtteranceTranscriber("auto", on_status=statuses.append)
            result = transcriber.transcribe(object())

        self.assertEqual(result, "recovered")
        self.assertEqual(transcriber.tier.device, "cpu")
        self.assertTrue(any("degraded" in message for message in statuses))


class RemoteCwdConfigAndCLITests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tempdir.cleanup)
        self.env = mock.patch.dict(
            os.environ,
            {"CLAUDE_PLUGIN_DATA": self.tempdir.name},
            clear=False,
        )
        self.env.start()
        self.addCleanup(self.env.stop)
        self.runner = CliRunner()

    def test_remote_cwd_config_set_and_get_persists_exact_path(self) -> None:
        remote_cwd = "/srv/Claude Projects/main"

        set_result = self.runner.invoke(main, ["config", "set", "remote-cwd", remote_cwd])
        get_result = self.runner.invoke(main, ["config", "get", "remote-cwd"])

        self.assertEqual(set_result.exit_code, 0, set_result.output)
        self.assertEqual(get_result.exit_code, 0, get_result.output)
        self.assertEqual(get_result.output.strip(), remote_cwd)
        self.assertEqual(config.remote_cwd(), remote_cwd)
        saved = json.loads((Path(self.tempdir.name) / "config.json").read_text())
        self.assertEqual(saved["remote-cwd"], remote_cwd)

    def test_remote_cwd_config_can_be_cleared_to_remote_home(self) -> None:
        config.set_remote_cwd("/srv/configured")

        clear_result = self.runner.invoke(
            main, ["config", "set", "remote-cwd", "home"]
        )
        get_result = self.runner.invoke(main, ["config", "get", "remote-cwd"])

        self.assertEqual(clear_result.exit_code, 0, clear_result.output)
        self.assertEqual(get_result.exit_code, 0, get_result.output)
        self.assertEqual(get_result.output.strip(), "home")
        self.assertIsNone(config.remote_cwd())

    def test_listen_remote_cwd_per_run_overrides_persisted_value(self) -> None:
        config.set_remote("configured@example")
        config.set_remote_cwd("/srv/configured")

        with mock.patch("talktomeclaude.listen.run_listen") as run:
            result = self.runner.invoke(
                main,
                [
                    "listen",
                    "--remote",
                    "override@example",
                    "--remote-cwd",
                    "/srv/per run",
                    "--once",
                ],
            )

        self.assertEqual(result.exit_code, 0, result.output)
        self.assertEqual(run.call_args.kwargs["remote"], "override@example")
        self.assertEqual(run.call_args.kwargs["remote_cwd"], "/srv/per run")

    def test_listen_uses_persisted_remote_cwd(self) -> None:
        config.set_remote("configured@example")
        config.set_remote_cwd("/srv/configured")

        with mock.patch("talktomeclaude.listen.run_listen") as run:
            result = self.runner.invoke(main, ["listen", "--once"])

        self.assertEqual(result.exit_code, 0, result.output)
        self.assertEqual(run.call_args.kwargs["remote"], "configured@example")
        self.assertEqual(run.call_args.kwargs["remote_cwd"], "/srv/configured")


if __name__ == "__main__":
    unittest.main()
