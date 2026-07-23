"""The production voice loop wiring: catalog-driven command firing, wake-word
gating, the conveyance checkpoint loop, and barge-in playback selection."""

from __future__ import annotations

import os
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

from talktomeclaude import command_catalog, config, listen


class _Isolated(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        env = mock.patch.dict(
            os.environ, {"CLAUDE_PLUGIN_DATA": self.tmp.name}, clear=False
        )
        env.start()
        self.addCleanup(env.stop)


class VoiceCommandDispatchTests(_Isolated):
    def _run(self, utterances: list[str], prompts: list[str], spoken: list[str]) -> None:
        stop_event = threading.Event()
        takes = iter(utterances)

        def next_take():
            try:
                next_utterance = next(takes)
            except StopIteration:
                stop_event.set()
                return None
            return next_utterance

        captured: list[str | None] = []

        def record(**_kwargs):
            captured.append(next_take())
            return captured[-1]

        def fake_prompt(text, session_id, **kwargs):
            prompts.append(text)
            handler = kwargs.get("on_event")
            if handler is not None:
                handler(
                    {
                        "type": "system",
                        "subtype": "init",
                        "slash_commands": ["kiln-fire", "model", "help"],
                    }
                )
            return ("ok", "sess-1")

        transcriber = mock.Mock()
        transcriber.transcribe.side_effect = lambda audio: audio or ""

        with mock.patch.object(os, "name", "posix"), mock.patch.object(
            listen, "_record_always_on", side_effect=lambda **kwargs: next_take()
        ), mock.patch.object(
            listen, "UtteranceTranscriber", return_value=transcriber
        ), mock.patch.object(listen, "_prompt_claude", fake_prompt):
            listen.run_listen(
                mode="always-on",
                session_id=None,
                tmux_pane=None,
                device="cpu",
                model=None,
                once=False,
                echo=lambda _line: None,
                speak=spoken.append,
                status=lambda _line: None,
                stop_event=stop_event,
                on_event=lambda _event: None,
            )

    def test_exact_command_name_confirms_then_fires_into_the_same_session(self) -> None:
        prompts: list[str] = []
        spoken: list[str] = []
        self._run(["hello", "kiln-fire", "go"], prompts, spoken)
        self.assertEqual(prompts, ["hello", "/kiln-fire"])
        self.assertTrue(any("Firing /kiln-fire" in line for line in spoken))
        saved = command_catalog.load_saved_flags()
        self.assertEqual(saved["kiln-fire"]["fire_count"], 1)

    def test_cancel_drops_the_pending_command(self) -> None:
        # The catalog is discovered from the session's init event, so the
        # first (ordinary) turn seeds it before a command can resolve.
        prompts: list[str] = []
        spoken: list[str] = []
        self._run(["hello", "kiln-fire", "cancel"], prompts, spoken)
        self.assertEqual(prompts, ["hello"])
        saved = command_catalog.load_saved_flags()
        self.assertEqual(saved["kiln-fire"]["fire_count"], 0)

    def test_ordinary_content_never_resolves_without_a_catalog(self) -> None:
        prompts: list[str] = []
        spoken: list[str] = []
        with mock.patch.object(
            listen, "_classify_intent", side_effect=AssertionError("no sub-call")
        ):
            self._run(["what is the capital of france"], prompts, spoken)
        self.assertEqual(prompts, ["what is the capital of france"])


class WakeGateTests(_Isolated):
    def _run_once(self, spoken: list[str]) -> None:
        transcriber = mock.Mock()
        transcriber.transcribe.return_value = "hello"
        with mock.patch.object(os, "name", "posix"), mock.patch.object(
            listen, "_record_always_on", return_value=object()
        ), mock.patch.object(
            listen, "UtteranceTranscriber", return_value=transcriber
        ), mock.patch.object(listen, "_prompt_claude", return_value=("ok", "s1")):
            listen.run_listen(
                mode="always-on",
                session_id=None,
                tmux_pane=None,
                device="cpu",
                model=None,
                once=True,
                echo=lambda _line: None,
                speak=spoken.append,
                status=lambda _line: None,
            )

    def test_enabled_wake_word_gates_capture_and_greets(self) -> None:
        config.set_wake_word(True)
        config.set_wake_model_path("/models/yo-claude.onnx")
        spoken: list[str] = []
        with mock.patch(
            "talktomeclaude.wakeword.wait_for_wake_word", return_value="yo claude"
        ) as detector:
            self._run_once(spoken)
        detector.assert_called_once()
        self.assertEqual(
            detector.call_args.args[0], "/models/yo-claude.onnx"
        )
        self.assertIn(listen.WAKE_GREETING, spoken)

    def test_disabled_wake_word_leaves_capture_ungated(self) -> None:
        config.set_wake_word(False)
        spoken: list[str] = []
        with mock.patch(
            "talktomeclaude.wakeword.wait_for_wake_word",
            side_effect=AssertionError("must not run the detector"),
        ):
            self._run_once(spoken)
        self.assertNotIn(listen.WAKE_GREETING, spoken)

    def test_missing_model_degrades_to_ungated_capture(self) -> None:
        config.set_wake_word(True)
        config.set_wake_model_path(None)
        spoken: list[str] = []
        self._run_once(spoken)  # must not raise or hang
        self.assertNotIn(listen.WAKE_GREETING, spoken)


class ConveyanceDeliveryTests(_Isolated):
    def _run_delivery(self, checkpoint_words: list[str], reply: str, spoken: list[str]):
        stop_event = threading.Event()
        takes = iter(["ask"] + checkpoint_words)

        def next_take(**_kwargs):
            try:
                return next(takes)
            except StopIteration:
                stop_event.set()
                return None

        transcriber = mock.Mock()
        transcriber.transcribe.side_effect = lambda audio: audio or ""

        cwd = os.getcwd()
        workdir = tempfile.TemporaryDirectory()
        self.addCleanup(workdir.cleanup)
        os.chdir(workdir.name)
        self.addCleanup(os.chdir, cwd)

        with mock.patch.object(os, "name", "posix"), mock.patch.object(
            listen, "_record_always_on", side_effect=next_take
        ), mock.patch.object(
            listen, "UtteranceTranscriber", return_value=transcriber
        ), mock.patch.object(listen, "_prompt_claude", return_value=(reply, "sess-7")):
            listen.run_listen(
                mode="always-on",
                session_id=None,
                tmux_pane=None,
                device="cpu",
                model=None,
                once=False,
                echo=lambda _line: None,
                speak=spoken.append,
                status=lambda _line: None,
                stop_event=stop_event,
            )
        return Path(workdir.name)

    def test_long_reply_is_chunked_with_persisted_checkpoints(self) -> None:
        reply = " ".join(f"Sentence number {index} is here." for index in range(30))
        spoken: list[str] = []
        root = self._run_delivery(["continue", "stop"], reply, spoken)
        self.assertGreater(len(spoken), 1)
        self.assertTrue(all(len(chunk.split()) <= 75 for chunk in spoken))
        pad = root / ".omc" / "state" / "sessions" / "sess-7" / "voice-conveyance.json"
        self.assertTrue(pad.is_file())
        import json

        state = json.loads(pad.read_text())
        self.assertEqual(set(state), {"cursor", "heading", "status"})
        self.assertEqual(state["status"], "stopped")

    def test_content_at_a_checkpoint_resumes_through_the_same_session(self) -> None:
        reply = " ".join(f"Sentence number {index} is here." for index in range(30))
        prompts: list[str] = []
        stop_event = threading.Event()
        takes = iter(["ask", "actually tell me about rust"])

        def next_take(**_kwargs):
            try:
                return next(takes)
            except StopIteration:
                stop_event.set()
                return None

        transcriber = mock.Mock()
        transcriber.transcribe.side_effect = lambda audio: audio or ""

        def fake_prompt(text, session_id, **_kwargs):
            prompts.append(text)
            return (reply if len(prompts) == 1 else "Short answer.", "sess-8")

        cwd = os.getcwd()
        workdir = tempfile.TemporaryDirectory()
        self.addCleanup(workdir.cleanup)
        os.chdir(workdir.name)
        self.addCleanup(os.chdir, cwd)

        with mock.patch.object(os, "name", "posix"), mock.patch.object(
            listen, "_record_always_on", side_effect=next_take
        ), mock.patch.object(
            listen, "UtteranceTranscriber", return_value=transcriber
        ), mock.patch.object(listen, "_prompt_claude", fake_prompt):
            listen.run_listen(
                mode="always-on",
                session_id=None,
                tmux_pane=None,
                device="cpu",
                model=None,
                once=False,
                echo=lambda _line: None,
                speak=lambda _line: None,
                status=lambda _line: None,
                stop_event=stop_event,
            )
        self.assertEqual(prompts, ["ask", "actually tell me about rust"])


class BargeInWiringTests(_Isolated):
    def test_enabled_gate_routes_speech_through_terminable_playback(self) -> None:
        config.set_barge_in(True)
        transcriber = mock.Mock()
        transcriber.transcribe.return_value = "hello"
        interruptible = mock.Mock(return_value=False)
        with mock.patch.object(os, "name", "posix"), mock.patch.object(
            listen, "headphones_present", return_value=True
        ), mock.patch.object(
            listen, "_speak_interruptible", interruptible
        ), mock.patch.object(
            listen, "_record_always_on", return_value=object()
        ), mock.patch.object(
            listen, "UtteranceTranscriber", return_value=transcriber
        ), mock.patch.object(listen, "_prompt_claude", return_value=("Hi.", "s1")):
            listen.run_listen(
                mode="always-on",
                session_id=None,
                tmux_pane=None,
                device="cpu",
                model=None,
                once=True,
                echo=lambda _line: None,
                speak=lambda _line: None,
                status=lambda _line: None,
            )
        interruptible.assert_called_once()

    def test_no_headphones_stays_on_the_sequential_path(self) -> None:
        config.set_barge_in(True)
        transcriber = mock.Mock()
        transcriber.transcribe.return_value = "hello"
        spoken: list[str] = []
        with mock.patch.object(os, "name", "posix"), mock.patch.object(
            listen, "headphones_present", return_value=False
        ), mock.patch.object(
            listen,
            "_speak_interruptible",
            side_effect=AssertionError("half-duplex must not monitor the mic"),
        ), mock.patch.object(
            listen, "_record_always_on", return_value=object()
        ), mock.patch.object(
            listen, "UtteranceTranscriber", return_value=transcriber
        ), mock.patch.object(listen, "_prompt_claude", return_value=("Hi.", "s1")):
            listen.run_listen(
                mode="always-on",
                session_id=None,
                tmux_pane=None,
                device="cpu",
                model=None,
                once=True,
                echo=lambda _line: None,
                speak=spoken.append,
                status=lambda _line: None,
            )
        self.assertEqual(spoken, ["Hi."])


if __name__ == "__main__":
    unittest.main()
