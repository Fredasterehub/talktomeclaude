"""Local text-to-speech built on the Piper engine.

Piper's lineage is GPL-3.0, so it is never imported as a library: the plugin
stays MIT by driving the ``piper`` executable through a subprocess boundary
only. Voices are from-scratch public-domain trains, fetched on first use from
the Hugging Face Hub and cached locally (a bundled/override ``voices/``
directory is honored first for offline use).
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


# Voices are public-domain Piper trains hosted canonically on the Hugging Face
# Hub. They are fetched on first use and cached (mirroring the STT model), so
# the repository stays small and installs never depend on git large-file limits.
HF_VOICES_REPO = "rhasspy/piper-voices"
_HF_VOICE_PATHS = {
    "en_US-ljspeech-high": "en/en_US/ljspeech/high/en_US-ljspeech-high.onnx",
    "en_GB-cori-medium": "en/en_GB/cori/medium/en_GB-cori-medium.onnx",
    "en_US-bryce-medium": "en/en_US/bryce/medium/en_US-bryce-medium.onnx",
}


def cache_voices_dir() -> Path:
    """Local cache where downloaded voices are materialized (flat names)."""
    override = os.environ.get("TALKTOMECLAUDE_VOICES_CACHE")
    if override:
        return Path(override)
    return Path.home() / ".cache" / "talktomeclaude" / "voices"


def _download_voice(voice: "Voice", on_status=None) -> tuple[Path, Path]:
    rel = _HF_VOICE_PATHS.get(voice.name)
    if rel is None:
        raise TTSError(f"no download source registered for voice {voice.name!r}")
    dest = cache_voices_dir()
    model_dst = voice.model_path(dest)
    config_dst = voice.config_path(dest)
    if model_dst.is_file() and config_dst.is_file():
        return model_dst, config_dst
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise TTSError(
            "huggingface_hub is required to download voices (pip install huggingface_hub), "
            "or place the voice files under TALKTOMECLAUDE_VOICES_DIR"
        ) from exc
    status = on_status or (lambda _message: None)
    status(f"downloading voice {voice.name} from {HF_VOICES_REPO} (once; caching to {dest})")
    dest.mkdir(parents=True, exist_ok=True)
    try:
        model_src = hf_hub_download(repo_id=HF_VOICES_REPO, filename=rel)
        config_src = hf_hub_download(repo_id=HF_VOICES_REPO, filename=rel + ".json")
    except Exception as exc:  # network / hub errors surface as a clean TTS error
        raise TTSError(f"failed to download voice {voice.name!r} from {HF_VOICES_REPO}: {exc}") from exc
    shutil.copyfile(model_src, model_dst)
    shutil.copyfile(config_src, config_dst)
    return model_dst, config_dst


def voice_files(voice: "Voice", on_status=None) -> tuple[Path, Path]:
    """Resolve (model, config) paths for *voice*, fetching on demand.

    Resolution order: a bundled/override ``voices/`` directory (offline use),
    then the local download cache, then a one-time download from the Hub.
    """
    bundled = voices_dir()
    if voice.is_installed(bundled):
        return voice.model_path(bundled), voice.config_path(bundled)
    cache = cache_voices_dir()
    if voice.is_installed(cache):
        return voice.model_path(cache), voice.config_path(cache)
    return _download_voice(voice, on_status)


def is_available(voice: "Voice") -> bool:
    """True if the voice is already on disk (bundled or cached) — no download needed."""
    return voice.is_installed(voices_dir()) or voice.is_installed(cache_voices_dir())


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
    cache = cache_voices_dir()
    # Prefer a voice already on disk (bundled or cached) so the default never
    # forces a download when a usable voice is present; otherwise consider all
    # bundled voices and let the best one fetch on demand.
    available = [v for v in BUNDLED_VOICES if v.is_installed(directory) or v.is_installed(cache)]
    candidates = available or list(BUNDLED_VOICES)
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


def synthesize(
    text: str,
    out_path: Path,
    voice_name: str | None = None,
    on_status=None,
) -> Voice:
    """Render *text* to a WAV file at *out_path*, fully locally.

    Returns the voice used. The voice files are resolved on demand (bundled,
    cached, or downloaded once from the Hub); Piper then runs as a subprocess
    only (GPL isolation).
    """
    voice = get_voice(voice_name) if voice_name else default_voice()
    model_path, config_path = voice_files(voice, on_status)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        [
            _piper_executable(),
            "--model", str(model_path),
            "--config", str(config_path),
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
