from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from talktomeclaude import tts
from talktomeclaude.catalog import BUNDLED_VOICES


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


if __name__ == "__main__":
    unittest.main()
