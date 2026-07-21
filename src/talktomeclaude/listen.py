"""Microphone listen loop: hear the operator, transcribe locally, drive Claude Code.

Injection strategy (directive D-3): the primary path drives ``claude -p`` with
``--resume`` so the voice loop owns its own session and every reply comes back
as structured JSON for the dialogue-only filter — no tmux requirement, works
over SSH and headless. The alternate path types the transcript into a live
interactive Claude Code TUI pane via ``tmux send-keys``. One driver per
session: a session owned by the voice loop is never simultaneously driven
from a live interactive window.

Recording modes (locked vocabulary): ``always-on`` segments hands-free at
pauses with VAD-gated transcription; ``push-to-talk`` records while a key is
held (terminal raw-mode reads, no extra dependencies); ``push-toggle`` starts
on one tap and sends on the next. The microphone stream is opened per
utterance and closed before any reply is spoken, so the listener never hears
the TTS voice and re-transcribes it as a prompt.
"""

import json
import os
import select
import shutil
import subprocess
import sys
import termios
import time
import tty
from typing import Callable

from talktomeclaude.stt import HOTWORDS, detect_tier, models_dir
from talktomeclaude.transcript import speakable

SAMPLE_RATE = 16000
_BLOCK_SECONDS = 0.05
_PREROLL_SECONDS = 0.3
_CALIBRATION_SECONDS = 0.4
_SILENCE_HANG_SECONDS = 0.9
_KEY_RELEASE_SECONDS = 0.6
_MAX_UTTERANCE_SECONDS = 60.0
_MIN_UTTERANCE_SECONDS = 0.3


class ListenError(RuntimeError):
    """Raised when the listen loop cannot proceed."""


class UtteranceTranscriber:
    """One loaded Whisper model reused for every utterance of the session.

    Tier selection follows directive D-1 (auto-detected, GPU first) and any
    fallback from the detected tier is reported through *on_status* — never
    a silent quality cut (D-2). Transcription runs with the Silero VAD
    filter so silence never hallucinates phantom phrases.
    """

    def __init__(
        self,
        device: str = "auto",
        model: str | None = None,
        on_status: Callable[[str], None] | None = None,
    ) -> None:
        status = on_status or (lambda message: None)
        tier = detect_tier(device, model)
        try:
            self._whisper = self._load(tier)
        except Exception as exc:
            if tier.device != "cuda" or device == "cuda":
                raise ListenError(
                    f"could not load STT tier ({tier.describe()}): {exc}"
                ) from exc
            fallback = detect_tier("cpu", model)
            status(
                f"stt tier degraded: {tier.describe()} failed ({exc}); "
                f"falling back to {fallback.describe()}"
            )
            try:
                self._whisper = self._load(fallback)
            except Exception as fallback_exc:
                raise ListenError(
                    f"could not load STT tier ({fallback.describe()}): {fallback_exc}"
                ) from fallback_exc
            tier = fallback
        self.tier = tier
        status(f"stt tier: {tier.describe()}")

    @staticmethod
    def _load(tier):
        from faster_whisper import WhisperModel

        return WhisperModel(
            tier.model,
            device=tier.device,
            compute_type=tier.compute_type,
            download_root=str(models_dir()),
        )

    def transcribe(self, audio) -> str:
        segments, _info = self._whisper.transcribe(
            audio,
            beam_size=5,
            hotwords=HOTWORDS,
            vad_filter=True,
        )
        return " ".join(
            part for part in (segment.text.strip() for segment in segments) if part
        )


def _sounddevice():
    try:
        import sounddevice
    except (ImportError, OSError) as exc:
        raise ListenError(f"microphone capture unavailable ({exc})") from exc
    return sounddevice


def _numpy():
    import numpy

    return numpy


class _RawKeys:
    """Cbreak-mode key reads on the listen process's own terminal."""

    def __init__(self) -> None:
        if not sys.stdin.isatty():
            raise ListenError(
                "push-to-talk and push-toggle need an interactive terminal; "
                "use --mode always-on when running without one"
            )
        self._fd = sys.stdin.fileno()
        self._saved = None

    def __enter__(self) -> "_RawKeys":
        self._saved = termios.tcgetattr(self._fd)
        tty.setcbreak(self._fd)
        return self

    def __exit__(self, *exc_info) -> None:
        if self._saved is not None:
            termios.tcsetattr(self._fd, termios.TCSADRAIN, self._saved)

    def read_key(self, timeout: float | None) -> str | None:
        ready, _, _ = select.select([self._fd], [], [], timeout)
        if not ready:
            return None
        data = os.read(self._fd, 1)
        if data in (b"\x03", b"\x04"):
            raise KeyboardInterrupt
        return data.decode("utf-8", errors="ignore")

    def drain(self) -> None:
        while self.read_key(0) is not None:
            pass


def _rms(block) -> float:
    numpy = _numpy()
    return float(numpy.sqrt(numpy.mean(numpy.square(block.astype(numpy.float64)))))


def _finish(chunks, minimum_seconds: float = _MIN_UTTERANCE_SECONDS):
    numpy = _numpy()
    if not chunks:
        return None
    audio = numpy.concatenate(chunks).reshape(-1)
    if audio.shape[0] < int(minimum_seconds * SAMPLE_RATE):
        return None
    return audio


def _record_push_to_talk(keys: _RawKeys) -> "object | None":
    """Record while a key is held: terminal auto-repeat keeps the take alive,
    and a repeat gap longer than the release window ends it."""
    keys.drain()
    if keys.read_key(None) is None:
        return None
    sounddevice = _sounddevice()
    blocksize = int(_BLOCK_SECONDS * SAMPLE_RATE)
    chunks = []
    last_key = time.monotonic()
    started = last_key
    with sounddevice.InputStream(
        samplerate=SAMPLE_RATE, channels=1, dtype="float32", blocksize=blocksize
    ) as stream:
        while True:
            block, _overflowed = stream.read(blocksize)
            chunks.append(block.copy())
            while keys.read_key(0) is not None:
                last_key = time.monotonic()
            now = time.monotonic()
            if now - last_key > _KEY_RELEASE_SECONDS:
                break
            if now - started > _MAX_UTTERANCE_SECONDS:
                break
    keys.drain()
    return _finish(chunks)


def _record_push_toggle(keys: _RawKeys) -> "object | None":
    """Tap to start recording, tap again to send."""
    keys.drain()
    if keys.read_key(None) is None:
        return None
    keys.drain()
    sounddevice = _sounddevice()
    blocksize = int(_BLOCK_SECONDS * SAMPLE_RATE)
    chunks = []
    started = time.monotonic()
    with sounddevice.InputStream(
        samplerate=SAMPLE_RATE, channels=1, dtype="float32", blocksize=blocksize
    ) as stream:
        while True:
            block, _overflowed = stream.read(blocksize)
            chunks.append(block.copy())
            if keys.read_key(0) is not None:
                break
            if time.monotonic() - started > _MAX_UTTERANCE_SECONDS:
                break
    keys.drain()
    return _finish(chunks)


def _record_always_on() -> "object | None":
    """Hands-free capture: calibrate the noise floor, trigger on speech
    energy, and end the utterance after a trailing pause."""
    sounddevice = _sounddevice()
    blocksize = int(_BLOCK_SECONDS * SAMPLE_RATE)
    preroll_blocks = max(1, int(_PREROLL_SECONDS / _BLOCK_SECONDS))
    hang_blocks = max(1, int(_SILENCE_HANG_SECONDS / _BLOCK_SECONDS))
    max_blocks = int(_MAX_UTTERANCE_SECONDS / _BLOCK_SECONDS)
    with sounddevice.InputStream(
        samplerate=SAMPLE_RATE, channels=1, dtype="float32", blocksize=blocksize
    ) as stream:
        ambient_samples = []
        for _ in range(max(1, int(_CALIBRATION_SECONDS / _BLOCK_SECONDS))):
            block, _overflowed = stream.read(blocksize)
            ambient_samples.append(_rms(block))
        noise_floor = sum(ambient_samples) / len(ambient_samples)
        threshold = max(noise_floor * 3.0, 0.01)
        preroll: list = []
        chunks: list = []
        silent_blocks = 0
        while True:
            block, _overflowed = stream.read(blocksize)
            level = _rms(block)
            if not chunks:
                preroll.append(block.copy())
                if len(preroll) > preroll_blocks:
                    preroll.pop(0)
                if level >= threshold:
                    chunks = list(preroll)
                    silent_blocks = 0
                else:
                    noise_floor = 0.95 * noise_floor + 0.05 * level
                    threshold = max(noise_floor * 3.0, 0.01)
                continue
            chunks.append(block.copy())
            silent_blocks = silent_blocks + 1 if level < threshold else 0
            if silent_blocks >= hang_blocks or len(chunks) >= max_blocks:
                break
    return _finish(chunks)


def _prompt_claude(text: str, session_id: str | None) -> tuple[str, str | None]:
    """Primary injection path (D-3): drive a claude -p session, resuming it
    across turns; the reply arrives as structured JSON."""
    claude = shutil.which("claude")
    if claude is None:
        raise ListenError(
            "the claude CLI is not on PATH; install Claude Code or use --tmux-pane"
        )
    command = [claude, "-p", text, "--output-format", "json"]
    if session_id:
        command += ["--resume", session_id]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        detail = result.stderr.strip() or f"exit {result.returncode}"
        raise ListenError(f"claude -p failed: {detail}")
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise ListenError(f"claude -p returned non-JSON output: {exc}") from exc
    if not isinstance(payload, dict):
        raise ListenError("claude -p returned an unexpected JSON shape")
    reply = payload.get("result")
    new_session = payload.get("session_id")
    return (
        reply if isinstance(reply, str) else "",
        new_session if isinstance(new_session, str) else session_id,
    )


def _send_tmux(pane: str, text: str) -> None:
    """Alternate injection path (D-3): type into a live interactive TUI."""
    if shutil.which("tmux") is None:
        raise ListenError("tmux is not on PATH; drop --tmux-pane to drive claude -p")
    try:
        subprocess.run(["tmux", "send-keys", "-t", pane, "-l", text], check=True)
        subprocess.run(["tmux", "send-keys", "-t", pane, "Enter"], check=True)
    except subprocess.CalledProcessError as exc:
        raise ListenError(f"tmux send-keys to pane {pane!r} failed: {exc}") from exc


def run_listen(
    mode: str,
    session_id: str | None,
    tmux_pane: str | None,
    device: str,
    model: str | None,
    once: bool,
    echo: Callable[[str], None],
    speak: Callable[[str], None],
    status: Callable[[str], None],
) -> None:
    """Drive the capture → transcribe → inject → reply loop until interrupted."""
    transcriber = UtteranceTranscriber(device, model, on_status=status)
    prompts = {
        "always-on": "listening (hands-free); Ctrl-C to stop",
        "push-to-talk": "hold any key to talk; release to send; Ctrl-C to stop",
        "push-toggle": "tap any key to talk; tap again to send; Ctrl-C to stop",
    }
    status(prompts[mode])
    keys = _RawKeys() if mode in ("push-to-talk", "push-toggle") else None

    def capture():
        if mode == "always-on":
            return _record_always_on()
        with keys:
            if mode == "push-to-talk":
                return _record_push_to_talk(keys)
            return _record_push_toggle(keys)

    while True:
        audio = capture()
        if audio is None:
            continue
        text = transcriber.transcribe(audio).strip()
        if not text:
            continue
        echo(f"you: {text}")
        if tmux_pane:
            _send_tmux(tmux_pane, text)
            status(f"sent to tmux pane {tmux_pane}; the live TUI owns the reply")
        else:
            reply, session_id = _prompt_claude(text, session_id)
            dialogue = speakable(reply)
            if dialogue:
                echo(f"claude: {dialogue}")
                speak(dialogue)
        if once:
            return
