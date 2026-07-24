"""Compatibility facade for the local speech implementation.

Existing callers keep importing :mod:`talktomeclaude.tts`; implementation and
warm-worker reuse live under :mod:`talktomeclaude.speech.voices`.
"""

from pathlib import Path

from talktomeclaude.catalog import Voice
from talktomeclaude.speech import voices as _voices

BUNDLED_VOICES = _voices.BUNDLED_VOICES
HF_VOICES_REPO = _voices.HF_VOICES_REPO
TTSError = _voices.TTSError

voices_dir = _voices.voices_dir
cache_voices_dir = _voices.cache_voices_dir
voice_files = _voices.voice_files
is_available = _voices.is_available
get_voice = _voices.get_voice
default_voice = _voices.default_voice
play_wav = _voices.play_wav

# Existing model-free tests patch these names to avoid downloads, executables,
# and hardware. Keep that seam for the lifetime of the compatibility facade.
registry = _voices.registry
subprocess = _voices.subprocess
_download_voice = _voices._download_voice
_hardware_allows_high_tier = _voices._hardware_allows_high_tier
_piper_executable = _voices._piper_executable
_synthesize_clone = _voices._synthesize_clone
_synthesize_f5 = _voices._synthesize_f5


def synthesize(
    text: str,
    out_path: Path,
    voice_name: str | None = None,
    on_status=None,
) -> Voice:
    """Render text through the speech implementation using facade test seams."""
    return _voices._synthesize_with(
        text,
        out_path,
        voice_name,
        on_status,
        get_voice_fn=get_voice,
        default_voice_fn=default_voice,
        voice_files_fn=voice_files,
        piper_executable_fn=_piper_executable,
    )


def synthesize_and_play(
    text: str,
    voice_name: str | None = None,
    on_status=None,
) -> Voice:
    """Render and play through the speech implementation."""
    return _voices._synthesize_and_play_with(
        text,
        voice_name,
        on_status,
        synthesize_fn=synthesize,
        play_wav_fn=play_wav,
    )


__all__ = [
    "BUNDLED_VOICES",
    "HF_VOICES_REPO",
    "TTSError",
    "Voice",
    "cache_voices_dir",
    "default_voice",
    "get_voice",
    "is_available",
    "play_wav",
    "synthesize",
    "synthesize_and_play",
    "voice_files",
    "voices_dir",
]
