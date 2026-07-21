"""Tests for guided voice creation (headless: engine + mic are mocked)."""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
import wave
from pathlib import Path
from unittest import mock

import numpy as np

from talktomeclaude import registry, wizard


class WizardTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.env = mock.patch.dict(
            os.environ, {"CLAUDE_PLUGIN_DATA": str(self.root)}, clear=False
        )
        self.env.start()
        self.addCleanup(self.env.stop)
        self.ref = self.root / "ref.wav"
        self.ref.write_bytes(b"RIFFfakewav")

    def test_registers_and_skips_sample_without_engine(self) -> None:
        messages: list[str] = []
        with mock.patch.object(wizard, "clone_available", return_value=False):
            voice, sample = wizard.create_clone_voice(
                "rick", self.ref, sample_text="hi", on_status=messages.append
            )
        self.assertEqual(voice.name, "rick")
        self.assertIsNone(sample)
        self.assertIsNotNone(registry.get("rick"))  # registered even without the engine
        self.assertTrue(any("doctor" in m for m in messages))

    def test_no_sample_text_means_no_render(self) -> None:
        with mock.patch.object(wizard, "clone_available", return_value=True) as available:
            voice, sample = wizard.create_clone_voice("gimli", self.ref)
        self.assertEqual(voice.name, "gimli")
        self.assertIsNone(sample)
        available.assert_not_called()  # short-circuits before checking the engine

    def test_renders_sample_when_engine_available(self) -> None:
        out = self.root / "sample.wav"
        with mock.patch.object(wizard, "clone_available", return_value=True), \
             mock.patch("talktomeclaude.tts.synthesize") as synth:
            voice, sample = wizard.create_clone_voice(
                "rick", self.ref, sample_text="testing", sample_out=out
            )
        self.assertEqual(sample, out)
        synth.assert_called_once()
        _, kwargs = synth.call_args
        self.assertEqual(kwargs["voice_name"], "rick")

    def test_record_reference_writes_wav_from_microphone(self) -> None:
        fake_sd = mock.Mock()
        fake_sd.rec.return_value = np.zeros((24000, 1), dtype="int16")
        out = self.root / "captured.wav"
        with mock.patch.dict(sys.modules, {"sounddevice": fake_sd}):
            result = wizard.record_reference(out, seconds=1.0, sample_rate=24000)
        self.assertEqual(result, out)
        with wave.open(str(out)) as reader:
            self.assertEqual(reader.getframerate(), 24000)
            self.assertEqual(reader.getnframes(), 24000)
            self.assertEqual(reader.getnchannels(), 1)
        fake_sd.wait.assert_called_once()

    def test_record_reference_rejects_nonpositive_length(self) -> None:
        with self.assertRaises(wizard.WizardError):
            wizard.record_reference(self.root / "x.wav", seconds=0)

    def test_record_reference_wraps_capture_failure(self) -> None:
        fake_sd = mock.Mock()
        fake_sd.rec.side_effect = RuntimeError("no input device")
        with mock.patch.dict(sys.modules, {"sounddevice": fake_sd}):
            with self.assertRaises(wizard.WizardError):
                wizard.record_reference(self.root / "x.wav", seconds=1.0)


if __name__ == "__main__":
    unittest.main()
