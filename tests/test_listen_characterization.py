"""Focused behavior locks for capture and Claude stream process boundaries."""

from __future__ import annotations

import io
import json
import subprocess
import unittest
from types import SimpleNamespace
from unittest import mock

from talktomeclaude import listen
from talktomeclaude.stt import CPU_TIER


class _StreamProcess:
    def __init__(self, *events: dict) -> None:
        self.stdout = iter(json.dumps(event, ensure_ascii=False) + "\n" for event in events)
        self.stderr = io.StringIO("")
        self.returncode = None
        self.terminated = False

    def terminate(self) -> None:
        self.terminated = True
        self.returncode = 0

    def wait(self, timeout=None) -> int:
        self.returncode = 0
        return 0

    def kill(self) -> None:
        self.returncode = 0


class CaptureCharacterizationTests(unittest.TestCase):
    def test_push_toggle_ignores_long_silence_until_the_second_toggle(self) -> None:
        keys = mock.Mock()
        keys.read_key.side_effect = [" ", None, None, " "]
        block = mock.Mock()
        block.copy.return_value = block
        stream = mock.MagicMock()
        stream.__enter__.return_value.read.return_value = (block, False)
        sounddevice = mock.Mock()
        sounddevice.InputStream.return_value = stream

        with mock.patch.object(listen, "_sounddevice", return_value=sounddevice), mock.patch.object(
            listen, "_finish", return_value="audio"
        ), mock.patch.object(listen.time, "monotonic") as monotonic:
            result = listen._record_push_toggle(keys, trigger_key=" ")

        self.assertEqual(result, "audio")
        self.assertEqual(stream.__enter__.return_value.read.call_count, 3)
        self.assertEqual(keys.read_key.call_args_list[-1], mock.call(0))
        monotonic.assert_not_called()

    def test_live_transcriber_preserves_exact_unicode_segments(self) -> None:
        model = mock.Mock()
        model.transcribe.return_value = (
            [SimpleNamespace(text=" Café "), SimpleNamespace(text="世界 👋 ")],
            None,
        )
        with mock.patch.object(listen, "detect_tier", return_value=CPU_TIER), mock.patch.object(
            listen.UtteranceTranscriber, "_load", return_value=model
        ):
            transcriber = listen.UtteranceTranscriber("cpu")
            text = transcriber.transcribe(object())

        self.assertEqual(text, "Café 世界 👋")


class StreamCharacterizationTests(unittest.TestCase):
    def test_stream_returns_exact_unicode_and_reaps_after_result(self) -> None:
        process = _StreamProcess(
            {"type": "system", "session_id": "session-é"},
            {"type": "result", "result": "Café 世界 👋", "session_id": "session-é"},
        )
        seen: list[dict] = []
        with mock.patch.object(listen.subprocess, "Popen", return_value=process):
            result, session_id = listen._consume_stream(
                ["claude"], seen.append, None, "dev@example"
            )

        self.assertEqual(result, "Café 世界 👋")
        self.assertEqual(session_id, "session-é")
        self.assertEqual(len(seen), 2)
        self.assertTrue(process.terminated)

    def test_reap_escalates_from_terminate_to_kill_after_timeout(self) -> None:
        process = mock.Mock()
        process.wait.side_effect = [subprocess.TimeoutExpired("claude", 0.01), 0]

        listen._reap(process, timeout=0.01)

        process.terminate.assert_called_once_with()
        process.kill.assert_called_once_with()
        self.assertEqual(process.wait.call_args_list, [mock.call(0.01), mock.call()])


if __name__ == "__main__":
    unittest.main()
