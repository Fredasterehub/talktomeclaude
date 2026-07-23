from __future__ import annotations

import inspect
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from talktomeclaude import tts
from talktomeclaude.catalog import BUNDLED_VOICES
from talktomeclaude.registry import RegisteredVoice


class PiperSynthesisTests(unittest.TestCase):
    def test_piper_receives_unicode_text_as_utf8(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            root = Path(raw_tmp)
            output = root / "reply.wav"
            model = root / "voice.onnx"
            config = root / "voice.onnx.json"

            def run(*_args, **_kwargs):
                output.write_bytes(b"RIFFaudio")
                return SimpleNamespace(returncode=0, stdout="", stderr="")

            with mock.patch.object(
                tts, "default_voice", return_value=BUNDLED_VOICES[0]
            ), mock.patch.object(
                tts, "voice_files", return_value=(model, config)
            ), mock.patch.object(
                tts, "_piper_executable", return_value="piper"
            ), mock.patch.object(
                tts.subprocess, "run", side_effect=run
            ) as subprocess_run:
                tts.synthesize("Hello \U0001f44b caf\u00e9", output)

        self.assertEqual(subprocess_run.call_args.kwargs["input"], "Hello \U0001f44b caf\u00e9")
        self.assertEqual(subprocess_run.call_args.kwargs["encoding"], "utf-8")
        self.assertEqual(subprocess_run.call_args.kwargs["errors"], "replace")


class PublicFacadeCharacterizationTests(unittest.TestCase):
    def test_public_function_signatures_are_stable(self) -> None:
        expected = {
            tts.get_voice: ("name",),
            tts.default_voice: ("directory",),
            tts.synthesize: ("text", "out_path", "voice_name", "on_status"),
            tts.play_wav: ("path",),
            tts.synthesize_and_play: ("text", "voice_name", "on_status"),
        }

        for function, parameters in expected.items():
            with self.subTest(function=function.__name__):
                self.assertEqual(tuple(inspect.signature(function).parameters), parameters)

    def test_registered_clone_resolution_preserves_identity_and_reference(self) -> None:
        registered = RegisteredVoice(
            name="rick",
            engine="clone",
            params={"reference": "C:/voices/rick.wav", "cfg_weight": 0.5},
            language="en",
            license="user-provided",
            provenance="local reference",
        )

        with mock.patch.object(tts.registry, "get", return_value=registered):
            voice = tts.get_voice("rick")

        self.assertEqual(voice.name, "rick")
        self.assertEqual(voice.engine, "clone")
        self.assertEqual(voice.params["reference"], "C:/voices/rick.wav")

    def test_combined_helper_removes_temp_file_when_synthesis_fails(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            handle, raw_path = tempfile.mkstemp(dir=raw_tmp, suffix=".wav")
            path = Path(raw_path)
            with mock.patch("tempfile.mkstemp", return_value=(handle, raw_path)), mock.patch.object(
                tts, "synthesize", side_effect=tts.TTSError("engine unavailable")
            ), mock.patch.object(tts, "play_wav") as play:
                with self.assertRaisesRegex(tts.TTSError, "engine unavailable"):
                    tts.synthesize_and_play("hello", "rick")

            self.assertFalse(path.exists())
            play.assert_not_called()


if __name__ == "__main__":
    unittest.main()
