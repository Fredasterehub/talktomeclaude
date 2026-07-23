"""Optional F5-TTS voice-cloning engine.

The F5 stack is intentionally imported only inside functions. Importing this
module therefore remains safe on the dependency-light core installation, with
no F5-TTS package or GPU stack present.
"""

from pathlib import Path

from talktomeclaude.tts import TTSError

_model = None


class F5Error(TTSError):
    """Raised when F5-TTS synthesis cannot proceed."""


def f5_available() -> bool:
    """Return whether the optional F5-TTS API can be imported."""
    try:
        from f5_tts.api import F5TTS  # noqa: F401
    except Exception:
        return False
    return True


def _load_model():
    global _model
    if _model is not None:
        return _model
    if not f5_available():
        raise F5Error(
            "the F5-TTS engine is not installed; install the optional F5-TTS "
            "stack before using an F5 voice"
        )
    from f5_tts.api import F5TTS

    try:
        _model = F5TTS()
    except Exception as exc:
        raise F5Error(f"failed to load the F5-TTS model: {exc}") from exc
    return _model


def synthesize_f5(
    text: str,
    out_path: Path,
    reference_path: str | Path,
    ref_text: str,
) -> Path:
    """Render *text* with F5-TTS using a transcribed reference clip."""
    if not text.strip():
        raise F5Error("nothing to speak")
    reference = Path(reference_path)
    if not reference.is_file():
        raise F5Error(f"reference clip not found: {reference}")
    if not isinstance(ref_text, str) or not ref_text.strip():
        raise F5Error("F5 reference text must be a non-empty string")

    output = Path(out_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    model = _load_model()
    try:
        model.infer(
            ref_file=str(reference),
            ref_text=ref_text,
            gen_text=text,
            file_wave=str(output),
        )
    except Exception as exc:
        output.unlink(missing_ok=True)
        raise F5Error(f"F5-TTS synthesis failed: {exc}") from exc
    if not output.is_file() or output.stat().st_size == 0:
        raise F5Error(f"F5-TTS produced no audio at {output}")
    return output
