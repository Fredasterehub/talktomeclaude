"""Microphone listen loop: hear the operator, transcribe locally, drive Claude Code.

Injection strategy (directive D-3): the primary path drives ``claude -p`` with
``--resume`` so the voice loop owns its own session and every reply comes back
as structured JSON for the dialogue-only filter — no tmux requirement. The
alternate path types the transcript into a live interactive Claude Code TUI
pane via ``tmux send-keys``. One driver per session: a session owned by the
voice loop is never simultaneously driven from a live interactive window.

Remote/SSH (``remote=user@host``): the microphone, transcription and spoken
reply stay on the machine the operator sits at, while Claude Code runs on the
server — either injection path is tunnelled over SSH. Multiplexing is used on
POSIX clients; native Windows uses its in-box OpenSSH without Unix
control-socket options. This is the headless-server pattern (e.g. a laptop
driving Claude Code on a Proxmox box): the server needs no audio hardware, the
client needs no Claude.

Recording modes (locked vocabulary): ``always-on`` segments hands-free at
pauses with VAD-gated transcription; ``push-to-talk`` records while a key is
held (terminal raw-mode reads, no extra dependencies); ``push-toggle`` starts
on one tap and sends on the next. The microphone stream is opened per
utterance and closed before any reply is spoken, so the listener never hears
the TTS voice and re-transcribes it as a prompt.
"""

import json
import os
import shlex
import shutil
import subprocess
import sys
import time
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
_WINDOWS_KEY_POLL_SECONDS = 0.01


class ListenError(RuntimeError):
    """Raised when the listen loop cannot proceed."""


def _is_windows() -> bool:
    """Return whether native Windows console/SSH behavior is required."""
    return os.name == "nt"


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
    """Dependency-free key reads on the listen process's own terminal.

    POSIX terminals use cbreak mode and ``select``. Native Windows consoles use
    ``msvcrt`` and need no terminal-mode changes. Imports stay inside their
    platform branches so importing this module works on either platform.
    """

    def __init__(self) -> None:
        if not sys.stdin.isatty():
            raise ListenError(
                "push-to-talk and push-toggle need an interactive terminal; "
                "use --mode always-on when running without one"
            )
        self._windows = _is_windows()
        self._fd = -1 if self._windows else sys.stdin.fileno()
        self._saved = None

    def __enter__(self) -> "_RawKeys":
        if self._windows:
            return self
        import termios
        import tty

        self._saved = termios.tcgetattr(self._fd)
        tty.setcbreak(self._fd)
        return self

    def __exit__(self, *exc_info) -> None:
        if self._saved is not None:
            import termios

            termios.tcsetattr(self._fd, termios.TCSADRAIN, self._saved)

    def read_key(self, timeout: float | None) -> str | None:
        if self._windows:
            return self._read_windows_key(timeout)

        import select

        ready, _, _ = select.select([self._fd], [], [], timeout)
        if not ready:
            return None
        data = os.read(self._fd, 1)
        if data in (b"\x03", b"\x04"):
            raise KeyboardInterrupt
        return data.decode("utf-8", errors="ignore")

    @staticmethod
    def _read_windows_key(timeout: float | None) -> str | None:
        import msvcrt

        if timeout is None:
            key = msvcrt.getwch()
        else:
            deadline = time.monotonic() + max(timeout, 0.0)
            while not msvcrt.kbhit():
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None
                time.sleep(min(_WINDOWS_KEY_POLL_SECONDS, remaining))
            key = msvcrt.getwch()

        if key in ("\x03", "\x04"):
            raise KeyboardInterrupt
        # Function/arrow keys arrive as a prefix plus scan code. Consume both
        # bytes as one event so push-toggle does not immediately stop again.
        if key in ("\x00", "\xe0"):
            key += msvcrt.getwch()
        return key

    def drain(self) -> None:
        while self.read_key(0) is not None:
            pass

    def is_pressed(self, key: str) -> bool | None:
        """Return whether *key* is physically held on Windows.

        Native Windows exposes key-up state through ``GetAsyncKeyState``.  The
        POSIX terminal stream has no key-up events, so callers receive ``None``
        there and retain the existing repeat-gap fallback.
        """
        if not self._windows:
            return None

        import ctypes

        user32 = getattr(ctypes, "windll").user32
        vk_key_scan = user32.VkKeyScanW
        vk_key_scan.argtypes = [ctypes.c_wchar]
        vk_key_scan.restype = ctypes.c_short
        map_virtual_key = user32.MapVirtualKeyW
        map_virtual_key.argtypes = [ctypes.c_uint, ctypes.c_uint]
        map_virtual_key.restype = ctypes.c_uint
        get_async_key_state = user32.GetAsyncKeyState
        get_async_key_state.argtypes = [ctypes.c_int]
        get_async_key_state.restype = ctypes.c_short

        if len(key) == 1:
            mapped = vk_key_scan(key)
            if mapped in (-1, 0xFFFF):
                return None
            virtual_key = mapped & 0xFF
        elif len(key) == 2 and key[0] in ("\x00", "\xe0"):
            # The second character returned by getwch() is the scan code.
            scan_code = ord(key[1])
            if key[0] == "\xe0":
                # MAPVK_VSC_TO_VK_EX expects the extended-key marker in the
                # high byte (for example Up is E0 48, represented as 0xE048).
                scan_code |= 0xE000
            virtual_key = map_virtual_key(scan_code, 3)
            if not virtual_key:
                return None
        else:
            return None
        return bool(get_async_key_state(virtual_key) & 0x8000)


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
    and a repeat gap longer than the release window ends it on POSIX. Native
    Windows polls the physical key state so its configurable repeat delay cannot
    truncate the start of an utterance."""
    keys.drain()
    trigger_key = keys.read_key(None)
    if trigger_key is None:
        return None
    physical_state = keys.is_pressed(trigger_key)
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
            if physical_state is None:
                while keys.read_key(0) is not None:
                    last_key = time.monotonic()
            else:
                physical_state = keys.is_pressed(trigger_key)
            now = time.monotonic()
            if physical_state is False:
                break
            if physical_state is None and now - last_key > _KEY_RELEASE_SECONDS:
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


def _ssh_base(remote: str) -> list[str]:
    """Build a platform-safe SSH invocation.

    POSIX OpenSSH keeps its low-latency control socket. Native Windows omits
    Unix control-socket settings, which are not consistently supported by the
    in-box OpenSSH client.
    """
    command = ["ssh", "-o", "ConnectTimeout=10"]
    if not _is_windows():
        command += [
            "-o", "ControlMaster=auto",
            "-o", "ControlPath=~/.ssh/cm-talktomeclaude-%r@%h:%p",
            "-o", "ControlPersist=600",
        ]
    return command + [remote]


def _remote_shell_command(inner: str, remote_cwd: str | None = None) -> str:
    """Wrap *inner* for a remote login shell, optionally changing directory.

    The target directory and complete shell program are quoted independently:
    this preserves spaces/metacharacters without allowing a configured path to
    become shell syntax.
    """
    if remote_cwd:
        inner = f"cd -- {shlex.quote(remote_cwd)} && {inner}"
    return f"bash -lc {shlex.quote(inner)}"


def _prompt_claude(
    text: str,
    session_id: str | None,
    remote: str | None = None,
    remote_cwd: str | None = None,
) -> tuple[str, str | None]:
    """Primary injection path (D-3): drive a claude -p session, resuming it
    across turns; the reply arrives as structured JSON.

    With *remote* set (``user@host``) the claude process runs on that host over
    SSH — the microphone, speech-to-text and spoken reply stay local, while
    Claude Code runs on the server. The remote command runs through a login
    shell so the server's normal PATH finds the ``claude`` CLI. When
    *remote_cwd* is set, Claude starts in that safely quoted project directory.
    """
    if remote:
        inner = f"claude -p {shlex.quote(text)} --output-format json"
        if session_id:
            inner += f" --resume {shlex.quote(session_id)}"
        command = _ssh_base(remote) + [_remote_shell_command(inner, remote_cwd)]
    else:
        claude = shutil.which("claude")
        if claude is None:
            raise ListenError(
                "the claude CLI is not on PATH; install Claude Code, pass "
                "--remote user@host to run it on a server, or use --tmux-pane"
            )
        command = [claude, "-p", text, "--output-format", "json"]
        if session_id:
            command += ["--resume", session_id]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        detail = result.stderr.strip() or f"exit {result.returncode}"
        where = f" on {remote}" if remote else ""
        raise ListenError(
            f"claude -p failed{where}: {detail}"
            + (
                "  (remote needs passwordless SSH keys and the claude CLI installed there)"
                if remote
                else ""
            )
        )
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


def _send_tmux(pane: str, text: str, remote: str | None = None) -> None:
    """Alternate injection path (D-3): type into a live interactive TUI.

    With *remote* set, the ``tmux send-keys`` runs on that host over SSH, so a
    local voice loop can type into a Claude Code TUI running on the server.
    """
    if remote:
        inner = (
            f"tmux send-keys -t {shlex.quote(pane)} -l {shlex.quote(text)} && "
            f"tmux send-keys -t {shlex.quote(pane)} Enter"
        )
        command = _ssh_base(remote) + [_remote_shell_command(inner)]
        try:
            subprocess.run(command, check=True)
        except subprocess.CalledProcessError as exc:
            raise ListenError(
                f"remote tmux send-keys to pane {pane!r} on {remote} failed: {exc}"
            ) from exc
        return
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
    remote: str | None = None,
    remote_cwd: str | None = None,
) -> None:
    """Drive the capture → transcribe → inject → reply loop until interrupted.

    When *remote* (``user@host``) is set, Claude Code runs on that host over
    SSH while the microphone, transcription and spoken reply stay on this
    machine — the remote/SSH pattern (voice where you sit, compute on the
    server). *remote_cwd* selects the project directory for remote ``claude``
    sessions.
    """
    transcriber = UtteranceTranscriber(device, model, on_status=status)
    if remote:
        status(f"remote: Claude Code runs on {remote} over SSH; voice stays local")
        if remote_cwd:
            status(f"remote project directory: {remote_cwd}")
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
            _send_tmux(tmux_pane, text, remote=remote)
            status(f"sent to tmux pane {tmux_pane}; the live TUI owns the reply")
        else:
            reply, session_id = _prompt_claude(
                text, session_id, remote=remote, remote_cwd=remote_cwd
            )
            dialogue = speakable(reply)
            if dialogue:
                echo(f"claude: {dialogue}")
                speak(dialogue)
        if once:
            return
