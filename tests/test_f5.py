"""Model-free tests for the optional F5-TTS adapter and TTS dispatch."""

from __future__ import annotations

import builtins
import importlib
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from talktomeclaude import f5, tts
from talktomeclaude.catalog import Voice
from talktomeclaude.tts import TTSError


class F5ModelFreeTests(unittest.TestCase):
    def tearDown(self) -> None:
        f5._model = None

    def test_module_import_does_not_attempt_heavy_imports(self) -> None:
        real_import = builtins.__import__

        def guarded_import(name, *args, **kwargs):
            if name == "torch" or name.startswith("f5_tts"):
                raise AssertionError(f"heavy import at module load: {name}")
            return real_import(name, *args, **kwargs)

        with mock.patch("builtins.__import__", side_effect=guarded_import):
            importlib.reload(f5)

    def test_f5_error_is_a_tts_error(self) -> None:
        self.assertTrue(issubclass(f5.F5Error, TTSError))

    def test_f5_available_reports_missing_optional_stack(self) -> None:
        real_import = builtins.__import__

        def missing_f5(name, *args, **kwargs):
            if name.startswith("f5_tts"):
                raise ImportError("not installed")
            return real_import(name, *args, **kwargs)

        with mock.patch("builtins.__import__", side_effect=missing_f5):
            self.assertFalse(f5.f5_available())

    def test_synthesize_f5_dispatches_reference_text_and_output(self) -> None:
        fake = mock.Mock()

        def infer(**kwargs):
            Path(kwargs["file_wave"]).write_bytes(b"RIFFaudio")
            return None

        fake.infer.side_effect = infer
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reference = root / "reference.wav"
            reference.write_bytes(b"RIFFreference")
            output = root / "result.wav"
            with mock.patch.object(f5, "_load_model", return_value=fake):
                result = f5.synthesize_f5(
                    "Generated speech.",
                    output,
                    reference,
                    "Reference transcript.",
                )
        self.assertEqual(result, output)
        fake.infer.assert_called_once_with(
            ref_file=str(reference),
            ref_text="Reference transcript.",
            gen_text="Generated speech.",
            file_wave=str(output),
        )

    def test_synthesize_f5_rejects_invalid_input_before_model_load(self) -> None:
        with mock.patch.object(f5, "_load_model") as load:
            with self.assertRaises(f5.F5Error):
                f5.synthesize_f5(" ", Path("/tmp/out.wav"), "/missing.wav", "text")
        load.assert_not_called()


class F5TTSDispatchTests(unittest.TestCase):
    def _voice(self, reference: Path) -> Voice:
        return Voice(
            name="f5_voice",
            language="en",
            quality="n/a",
            license="personal",
            provenance="test",
            engine="f5",
            params={
                "reference": str(reference),
                "ref_text": "Reference transcript.",
            },
        )

    def test_is_available_uses_f5_adapter(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            reference = Path(directory) / "reference.wav"
            reference.write_bytes(b"RIFFreference")
            voice = self._voice(reference)
            with mock.patch.object(f5, "f5_available", return_value=True) as available:
                self.assertTrue(tts.is_available(voice))
        available.assert_called_once_with()

    def test_synthesize_dispatches_f5_params(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reference = root / "reference.wav"
            reference.write_bytes(b"RIFFreference")
            output = root / "output.wav"
            voice = self._voice(reference)
            with mock.patch.object(tts, "get_voice", return_value=voice), mock.patch.object(
                f5, "synthesize_f5", return_value=output
            ) as synthesize:
                result = tts.synthesize("Hello.", output, voice_name=voice.name)
        self.assertIs(result, voice)
        synthesize.assert_called_once_with(
            "Hello.",
            output,
            str(reference),
            "Reference transcript.",
        )


if __name__ == "__main__":
    unittest.main()
