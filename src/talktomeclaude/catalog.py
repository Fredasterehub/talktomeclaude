"""Bundled voice catalog — the leaf data layer for voices.

This module holds the :class:`Voice` value type and the three bundled
public-domain Piper voices, and imports nothing else from the package. Both the
synthesis layer (:mod:`tts`) and the user registry (:mod:`registry`) depend on
it, which is what lets the registry validate names against the bundled set
without importing the synthesis engines (no circular dependency).
"""

from dataclasses import dataclass, field
from pathlib import Path

# Voices are public-domain Piper trains hosted canonically on the Hugging Face
# Hub, fetched on first use and cached (see tts.voice_files).
HF_VOICES_REPO = "rhasspy/piper-voices"


@dataclass(frozen=True)
class Voice:
    name: str
    language: str
    quality: str
    license: str
    provenance: str
    engine: str = "piper"
    params: dict = field(default_factory=dict)

    def model_path(self, voices_dir: Path) -> Path:
        return voices_dir / f"{self.name}.onnx"

    def config_path(self, voices_dir: Path) -> Path:
        return voices_dir / f"{self.name}.onnx.json"

    def is_installed(self, voices_dir: Path) -> bool:
        return self.model_path(voices_dir).is_file() and self.config_path(voices_dir).is_file()


BUNDLED_VOICES = (
    Voice(
        name="en_US-ljspeech-high",
        language="en_US",
        quality="high",
        license="public domain",
        provenance="LJ Speech dataset, trained from scratch",
        engine="piper",
        params={"hf_path": "en/en_US/ljspeech/high/en_US-ljspeech-high.onnx"},
    ),
    Voice(
        name="en_GB-cori-medium",
        language="en_GB",
        quality="medium",
        license="public domain",
        provenance="LibriVox recordings, trained from scratch",
        engine="piper",
        params={"hf_path": "en/en_GB/cori/medium/en_GB-cori-medium.onnx"},
    ),
    Voice(
        name="en_US-bryce-medium",
        language="en_US",
        quality="medium",
        license="public domain",
        provenance="author's own voice recordings",
        engine="piper",
        params={"hf_path": "en/en_US/bryce/medium/en_US-bryce-medium.onnx"},
    ),
)

BUNDLED_VOICE_NAMES = frozenset(voice.name for voice in BUNDLED_VOICES)
