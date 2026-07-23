"""Tests for the user voice registry (bring-your-own Piper + cloned voices)."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from talktomeclaude import registry


class RegistryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        # Route config_dir() (and therefore the registry + voice-refs) here.
        self.env = mock.patch.dict(
            os.environ, {"TALKTOMECLAUDE_CONFIG_DIR": str(self.root)}, clear=False
        )
        self.env.start()
        self.addCleanup(self.env.stop)

    def _make_piper(self, stem: str = "myvoice") -> Path:
        model = self.root / f"{stem}.onnx"
        model.write_bytes(b"onnx")
        model.with_name(model.name + ".json").write_text("{}")
        return model

    def _make_ref(self, name: str = "ref.wav") -> Path:
        ref = self.root / name
        ref.write_bytes(b"RIFFfakewav")
        return ref

    def test_empty_registry_lists_nothing(self) -> None:
        self.assertEqual(registry.list_voices(), [])
        self.assertIsNone(registry.get("nope"))

    def test_add_piper_references_model_in_place(self) -> None:
        model = self._make_piper()
        voice = registry.add_piper("myvoice", model)
        self.assertEqual(voice.engine, "piper")
        self.assertEqual(voice.params["model"], str(model.resolve()))
        self.assertEqual(voice.params["config"], str(model.resolve()) + ".json")
        self.assertEqual([v.name for v in registry.list_voices()], ["myvoice"])
        saved = json.loads((self.root / "voices.json").read_text())
        self.assertEqual(saved["voices"]["myvoice"]["engine"], "piper")

    def test_add_piper_rejects_missing_or_non_onnx(self) -> None:
        with self.assertRaises(registry.RegistryError):
            registry.add_piper("v1", self.root / "absent.onnx")
        wrong = self.root / "voice.bin"
        wrong.write_bytes(b"x")
        with self.assertRaises(registry.RegistryError):
            registry.add_piper("v2", wrong)

    def test_add_piper_requires_config_beside_model(self) -> None:
        model = self.root / "noconfig.onnx"
        model.write_bytes(b"onnx")
        with self.assertRaises(registry.RegistryError):
            registry.add_piper("noconfig", model)

    def test_add_clone_copies_reference_into_voice_refs(self) -> None:
        ref = self._make_ref()
        voice = registry.add_clone("rick", ref, exaggeration=0.7)
        stored = Path(voice.params["reference"])
        self.assertTrue(stored.is_file())
        self.assertEqual(stored.parent, registry.refs_dir())
        self.assertEqual(stored.name, "rick.wav")
        self.assertEqual(voice.params["exaggeration"], 0.7)
        # Original still present; the registry copied rather than moved.
        self.assertTrue(ref.is_file())

    def test_add_f5_copies_reference_and_stores_transcript(self) -> None:
        ref = self._make_ref()
        voice = registry.add_f5("rick_f5", ref, "Wubba lubba dub dub.")
        stored = Path(voice.params["reference"])
        self.assertEqual(voice.engine, "f5")
        self.assertEqual(voice.params["ref_text"], "Wubba lubba dub dub.")
        self.assertEqual(stored, registry.refs_dir() / "rick_f5.wav")
        self.assertTrue(stored.is_file())
        saved = json.loads(registry.registry_path().read_text())
        self.assertEqual(saved["voices"]["rick_f5"]["params"]["reference"], "rick_f5.wav")

    def test_add_f5_requires_reference_text(self) -> None:
        with self.assertRaises(registry.RegistryError):
            registry.add_f5("silent", self._make_ref(), "   ")

    def test_add_f5_rolls_back_clip_when_save_fails(self) -> None:
        with mock.patch.object(registry, "_save", side_effect=OSError("disk full")):
            with self.assertRaises(OSError):
                registry.add_f5("rick_f5", self._make_ref(), "Reference transcript.")
        self.assertFalse((registry.refs_dir() / "rick_f5.wav").exists())

    def test_remove_clone_deletes_copied_reference(self) -> None:
        ref = self._make_ref()
        voice = registry.add_clone("gimli", ref)
        stored = Path(voice.params["reference"])
        self.assertTrue(stored.is_file())
        registry.remove("gimli")
        self.assertIsNone(registry.get("gimli"))
        self.assertFalse(stored.is_file())

    def test_remove_f5_deletes_copied_reference(self) -> None:
        voice = registry.add_f5("gimli_f5", self._make_ref(), "And my axe.")
        stored = Path(voice.params["reference"])
        registry.remove("gimli_f5")
        self.assertIsNone(registry.get("gimli_f5"))
        self.assertFalse(stored.is_file())

    def test_remove_piper_keeps_external_model_file(self) -> None:
        model = self._make_piper()
        registry.add_piper("myvoice", model)
        registry.remove("myvoice")
        self.assertIsNone(registry.get("myvoice"))
        self.assertTrue(model.is_file())  # BYO model lives outside the registry

    def test_remove_unknown_raises(self) -> None:
        with self.assertRaises(registry.RegistryError):
            registry.remove("ghost")

    def test_invalid_names_rejected(self) -> None:
        ref = self._make_ref()
        for bad in ("", ".hidden", "has space", "slash/name", "a" * 65):
            with self.assertRaises(registry.RegistryError):
                registry.add_clone(bad, ref)

    def test_reserved_and_bundled_names_rejected(self) -> None:
        ref = self._make_ref()
        with self.assertRaises(registry.RegistryError):
            registry.add_clone("default", ref)
        with self.assertRaises(registry.RegistryError):
            registry.add_clone("en_US-ljspeech-high", ref)  # a bundled voice

    def test_duplicate_name_rejected(self) -> None:
        ref = self._make_ref()
        registry.add_clone("dup", ref)
        with self.assertRaises(registry.RegistryError):
            registry.add_clone("dup", ref)

    def test_corrupt_registry_file_is_tolerated(self) -> None:
        (self.root / "voices.json").write_text("{ not json")
        self.assertEqual(registry.list_voices(), [])
        # A subsequent add still works and overwrites the corrupt file.
        registry.add_clone("fresh", self._make_ref())
        self.assertIsNotNone(registry.get("fresh"))

    # --- hardening (from the cross-family review) --------------------------- #
    def test_name_with_trailing_newline_rejected(self) -> None:
        with self.assertRaises(registry.RegistryError):
            registry.add_clone("safe\n", self._make_ref())

    def test_windows_device_names_rejected(self) -> None:
        for device in ("con", "CON", "nul", "com1", "LPT9"):
            with self.assertRaises(registry.RegistryError):
                registry.add_clone(device, self._make_ref())

    def test_trailing_dot_rejected(self) -> None:
        with self.assertRaises(registry.RegistryError):
            registry.add_clone("rick.", self._make_ref())

    def test_case_insensitive_collision_rejected(self) -> None:
        registry.add_clone("Rick", self._make_ref())
        with self.assertRaises(registry.RegistryError):
            registry.add_clone("rick", self._make_ref())

    def test_non_audio_reference_rejected(self) -> None:
        bad = self.root / "notes.txt"
        bad.write_text("not audio")
        with self.assertRaises(registry.RegistryError):
            registry.add_clone("voice", bad)

    def test_weight_out_of_range_or_non_finite_rejected(self) -> None:
        ref = self._make_ref()
        for bad in (1.5, -0.1, float("nan"), float("inf")):
            with self.assertRaises(registry.RegistryError):
                registry.add_clone("voice", ref, exaggeration=bad)

    def test_params_mapping_is_read_only(self) -> None:
        voice = registry.add_clone("rick", self._make_ref())
        with self.assertRaises(TypeError):
            voice.params["reference"] = "x"  # type: ignore[index]

    def test_add_piper_license_name_kwarg(self) -> None:
        voice = registry.add_piper("byo", self._make_piper(), license_name="MIT")
        self.assertEqual(voice.license, "MIT")

    def test_null_params_record_does_not_crash(self) -> None:
        registry._save({"version": 1, "voices": {"x": {"engine": "clone", "params": None}}})
        self.assertEqual([v.name for v in registry.list_voices()], ["x"])
        self.assertEqual(dict(registry.get("x").params), {})

    def test_unknown_engine_skipped_in_list_and_raises_on_get(self) -> None:
        registry._save({"version": 1, "voices": {"weird": {"engine": "mystery", "params": {}}}})
        self.assertEqual(registry.list_voices(), [])
        with self.assertRaises(registry.RegistryError):
            registry.get("weird")

    def test_corrupt_file_is_quarantined_before_overwrite(self) -> None:
        registry.add_clone("keep", self._make_ref("a.wav"))
        (self.root / "voices.json").write_text("{ broken")
        registry.add_clone("new", self._make_ref("b.wav"))
        backups = list(self.root.glob("voices.json.corrupt*"))
        self.assertTrue(backups, "the corrupt file should be quarantined, not silently dropped")
        self.assertIsNotNone(registry.get("new"))

    def test_add_clone_rolls_back_clip_when_save_fails(self) -> None:
        ref = self._make_ref()
        with mock.patch.object(registry, "_save", side_effect=OSError("disk full")):
            with self.assertRaises(OSError):
                registry.add_clone("rick", ref)
        self.assertFalse((registry.refs_dir() / "rick.wav").exists())

    # --- second-pass review fixes ----------------------------------------- #
    @unittest.skipUnless(os.name == "posix", "Unix file-permission bits")
    def test_clone_reference_stored_private_0600(self) -> None:
        voice = registry.add_clone("rick", self._make_ref())
        stored = Path(voice.params["reference"])
        self.assertEqual(stored.stat().st_mode & 0o777, 0o600)

    def test_copy_into_refs_survives_source_equals_destination(self) -> None:
        refs = registry._safe_refs_dir()
        clip = refs / "rick.wav"
        clip.write_bytes(b"RIFFdata")
        registry._copy_into_refs(clip, clip)  # aliased: must not destroy the source
        self.assertTrue(clip.is_file())
        self.assertEqual(clip.read_bytes(), b"RIFFdata")

    def test_invalid_utf8_read_tolerated_and_write_quarantines(self) -> None:
        (self.root / "voices.json").write_bytes(b"\xff\xfe not valid utf-8")
        self.assertEqual(registry.list_voices(), [])  # tolerant read, no crash
        registry.add_clone("fresh", self._make_ref())
        self.assertTrue(list(self.root.glob("voices.json.corrupt*")))
        self.assertIsNotNone(registry.get("fresh"))

    def test_dotted_windows_device_name_rejected(self) -> None:
        with self.assertRaises(registry.RegistryError):
            registry.add_clone("CON.voice", self._make_ref())

    def test_quarantine_failure_aborts_mutation(self) -> None:
        (self.root / "voices.json").write_text("{ broken")
        with mock.patch.object(
            registry, "_quarantine", side_effect=registry.RegistryError("cannot move")
        ):
            with self.assertRaises(registry.RegistryError):
                registry.add_clone("x", self._make_ref())


if __name__ == "__main__":
    unittest.main()
