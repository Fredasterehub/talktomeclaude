"""Local text-to-speech built on the Piper engine.

Piper's lineage is GPL-3.0, so it is never imported as a library: the plugin
stays MIT by driving the ``piper`` executable through a subprocess boundary
only. Voices are bundled from-scratch public-domain trains (see each voice's
MODEL_CARD next to its model under ``voices/``).
"""

import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Voice:
    name: str
    language: str
    quality: str
    license: str
    provenance: str

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
    ),
    Voice(
        name="en_GB-cori-medium",
        language="en_GB",
        quality="medium",
        license="public domain",
        provenance="LibriVox recordings, trained from scratch",
    ),
    Voice(
        name="en_US-bryce-medium",
        language="en_US",
        quality="medium",
        license="public domain",
        provenance="author's own voice recordings",
    ),
)

_QUALITY_RANK = {"low": 0, "x_low": 0, "medium": 1, "high": 2}


class TTSError(RuntimeError):
    """Raised when speech synthesis cannot proceed."""


def voices_dir() -> Path:
    override = os.environ.get("TALKTOMECLAUDE_VOICES_DIR")
    if override:
        return Path(override)
    return Path(__file__).resolve().parents[2] / "voices"


def get_voice(name: str) -> Voice:
    for voice in BUNDLED_VOICES:
        if voice.name == name:
            return voice
    known = ", ".join(v.name for v in BUNDLED_VOICES)
    raise TTSError(f"unknown voice {name!r} (bundled voices: {known})")


def _hardware_allows_high_tier() -> bool:
    """Hardware-tier auto-detection (directive D-1).

    Piper's high-quality profile stays faster than realtime on any
    multi-core CPU, and a visible CUDA GPU always carries it; only very
    small machines drop to the medium profile. One code path either way.
    """
    if shutil.which("nvidia-smi"):
        return True
    return (os.cpu_count() or 1) >= 4


def default_voice(directory: Path | None = None) -> Voice:
    directory = directory or voices_dir()
    installed = [v for v in BUNDLED_VOICES if v.is_installed(directory)]
    candidates = installed or list(BUNDLED_VOICES)
    max_rank = 2 if _hardware_allows_high_tier() else 1
    eligible = [v for v in candidates if _QUALITY_RANK.get(v.quality, 1) <= max_rank]
    if not eligible:
        eligible = candidates
    return max(eligible, key=lambda v: _QUALITY_RANK.get(v.quality, 1))


def _piper_executable() -> str:
    local = Path(sys.executable).parent / "piper"
    if local.is_file() and os.access(local, os.X_OK):
        return str(local)
    found = shutil.which("piper")
    if found:
        return found
    raise TTSError(
        "piper executable not found; install it into the project venv "
        "(pip install piper-tts)"
    )


def synthesize(text: str, out_path: Path, voice_name: str | None = None) -> Voice:
    """Render *text* to a WAV file at *out_path*, fully locally.

    Returns the voice used. Piper runs as a subprocess only (GPL isolation).
    """
    directory = voices_dir()
    voice = get_voice(voice_name) if voice_name else default_voice(directory)
    if not voice.is_installed(directory):
        raise TTSError(
            f"voice {voice.name!r} is not installed under {directory}; "
            f"expected {voice.model_path(directory).name} and "
            f"{voice.config_path(directory).name}"
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        [
            _piper_executable(),
            "--model", str(voice.model_path(directory)),
            "--config", str(voice.config_path(directory)),
            "--output-file", str(out_path),
        ],
        input=text,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip() or "no output"
        raise TTSError(f"piper failed (exit {result.returncode}): {detail}")
    if not out_path.is_file() or out_path.stat().st_size == 0:
        raise TTSError(f"piper produced no audio at {out_path}")
    return voice
