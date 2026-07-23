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
import queue
import re
import shlex
import shutil
import subprocess
import sys
import threading
import time
from typing import Callable

from talktomeclaude import command_catalog, config, conveyance, intent
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


CONVEYANCE_VERBS = (
    "continue",
    "repeat",
    "back",
    "skip",
    "stop",
    "slower",
    "expand",
    "steer",
)


def detect_verb(utterance: str) -> str | None:
    """Return a closed conveyance verb only for an exact spoken match."""
    normalized = utterance.strip().casefold()
    return normalized if normalized in CONVEYANCE_VERBS else None


def _is_windows() -> bool:
    """Return whether native Windows console/SSH behavior is required."""
    return os.name == "nt"


def barge_in_active(config_on: bool, headphones: bool) -> bool:
    """Return whether opted-in barge-in has capable audio hardware."""
    return config_on and headphones


# ── voice-activated commands ─────────────────────────────────────────────────
WAKE_GREETING = "Yeah, I'm listening."
INTENT_MODEL = "haiku"
_INTENT_CONFIDENCE_FLOOR = 0.75
_COMMAND_TRIGGERS = ("run command ", "fire command ", "command ")
_FIRE_WORDS = frozenset({"go", "yes", "fire", "confirm", "do it"})
_CANCEL_WORDS = frozenset({"cancel", "no", "stop", "never mind", "nevermind"})

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


def _catalog_by_id(catalog: list[dict], command_id) -> dict | None:
    for record in catalog:
        if record.get("id") == command_id:
            return record
    return None


def _fire_text(record: dict, args: str) -> str:
    namespace = record.get("namespace") or ""
    name = f"{namespace}:{record['id']}" if namespace else str(record["id"])
    return f"/{name} {args}".strip()


def _intent_prompt(request: str, catalog: list[dict]) -> str:
    lines = [
        "Classify this spoken request against the available commands. Answer "
        "with ONLY a JSON object shaped as "
        '{"command_id": string or null, "args": string, "missing_slots": [], '
        '"confidence": number between 0 and 1, "alternatives": []}.',
        f"Request: {request}",
        "Commands:",
    ]
    for record in catalog:
        namespace = record.get("namespace") or "top-level"
        description = record.get("description") or "no description"
        lines.append(f"- {record['id']} ({namespace}): {description}")
    return "\n".join(lines)


def _classify_intent(
    request: str, catalog: list[dict], status: Callable[[str], None]
) -> "tuple[dict, str] | None":
    """Resolve *request* through the isolated intent sub-call — its own
    throwaway ``claude -p`` session, never resumed into the working history."""
    command = intent.intent_subcall_command(_intent_prompt(request, catalog), INTENT_MODEL)
    try:
        result = _run_captured(command)
    except (OSError, subprocess.SubprocessError) as exc:
        status(f"intent classification unavailable ({exc}); treating the request as content")
        return None
    if result.returncode != 0 or not result.stdout.strip():
        status("intent classification failed; treating the request as content")
        return None
    try:
        envelope = json.loads(result.stdout)
        raw = envelope.get("result", "") if isinstance(envelope, dict) else ""
        raw = raw.strip() if isinstance(raw, str) else ""
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        parsed = intent.parse_intent_response(raw)
    except (ValueError, KeyError, TypeError, IndexError):
        status("intent response unreadable; treating the request as content")
        return None
    confidence = parsed.confidence if isinstance(parsed.confidence, (int, float)) else 0.0
    record = _catalog_by_id(catalog, parsed.command_id) if parsed.command_id else None
    if record is None or confidence < _INTENT_CONFIDENCE_FLOOR:
        status("no confident command match; treating the request as content")
        return None
    args = parsed.args if isinstance(parsed.args, str) else ""
    return record, args


def _resolve_command(
    utterance: str, catalog: list[dict], status: Callable[[str], None]
) -> "tuple[dict, str] | None":
    """Resolve a spoken utterance to a fireable command.

    The deterministic keyword prefilter runs first (exact command names,
    no model round-trip). Only an explicit command request ("command …",
    "run command …", "fire command …") escalates to the isolated intent
    sub-call; ordinary content never does.
    """
    enabled = [record for record in catalog if record.get("enabled", True)]
    if not enabled:
        return None
    command_id = intent.keyword_prefilter(utterance, enabled)
    if command_id is not None:
        record = _catalog_by_id(enabled, command_id)
        return (record, "") if record is not None else None
    lowered = utterance.strip().casefold()
    for trigger in _COMMAND_TRIGGERS:
        if lowered.startswith(trigger):
            request = utterance.strip()[len(trigger):].strip()
            if not request:
                return None
            command_id = intent.keyword_prefilter(request, enabled)
            if command_id is not None:
                record = _catalog_by_id(enabled, command_id)
                return (record, "") if record is not None else None
            return _classify_intent(request, enabled, status)
    return None


# ── wake word, barge-in, and gradual conveyance ──────────────────────────────
def _setting(reader: Callable[[], object], default):
    """Read a persisted setting, degrading to *default* when the config store
    is unreachable (for example, no resolvable home directory)."""
    try:
        return reader()
    except Exception:
        return default


def _wake_gate(
    speak: Callable[[str], None],
    status: Callable[[str], None],
    stop_event: "threading.Event | None",
    state: dict,
) -> bool:
    """Honor the opt-in wake word before a hands-free capture.

    Returns False when the session is stopping. When wake mode is off, no
    detector model is configured, or the optional wake-word runtime is
    unavailable, capture stays ungated (reported once, never silent).
    """
    if not _setting(config.wake_word_enabled, False) or state.get("degraded"):
        return True
    model_path = _setting(config.wake_model_path, None)
    phrase = _setting(config.wake_phrase, config.DEFAULT_WAKE_PHRASE)
    if not model_path:
        state["degraded"] = True
        status(
            "wake word is on but no detector model is configured "
            "(config set wake-model /path/to/model.onnx); listening ungated"
        )
        return True
    from talktomeclaude import wakeword

    status(f"waiting for the wake phrase {phrase!r}")
    try:
        heard = wakeword.wait_for_wake_word(
            model_path, phrase=phrase, stop_event=stop_event
        )
    except wakeword.WakeWordError as exc:
        state["degraded"] = True
        status(f"wake word degraded: {exc}; listening ungated")
        return True
    if heard is None:
        return False  # stopping
    speak(WAKE_GREETING)  # local canned greeting — never a Claude round-trip
    return True


def headphones_present() -> bool:
    """Best-effort detection of headphone-class output hardware.

    Any failure reads as absent, auto-degrading barge-in to half-duplex.
    """
    try:
        sounddevice = _sounddevice()
        device = sounddevice.query_devices(kind="output")
        name = str(device["name"]).casefold()
    except Exception:
        return False
    return any(
        marker in name for marker in ("headphone", "headset", "earbud", "airpod")
    )


_BARGE_IN_CALIBRATION_BLOCKS = 6
_BARGE_IN_VOICED_BLOCKS = 3


def _speak_interruptible(
    speak: Callable[[str], None],
    text: str,
    stop_event: "threading.Event | None" = None,
) -> bool:
    """Play *text* while watching the microphone, halting playback the moment
    the operator speaks. Returns True when the operator barged in.

    Playback runs on a worker thread; detection watches microphone energy
    against a floor calibrated just before playback starts (the headphone
    gate keeps the TTS voice out of the microphone). Playback is terminated
    through ``sounddevice.stop()``, which halts the active output stream.
    """
    sounddevice = _sounddevice()
    blocksize = int(_BLOCK_SECONDS * SAMPLE_RATE)
    failure: list[BaseException] = []
    done = threading.Event()

    def playback() -> None:
        try:
            speak(text)
        except BaseException as exc:  # surfaced to the caller after join
            failure.append(exc)
        finally:
            done.set()

    interrupted = False
    with sounddevice.InputStream(
        samplerate=SAMPLE_RATE, channels=1, dtype="float32", blocksize=blocksize
    ) as stream:
        floors = []
        for _ in range(_BARGE_IN_CALIBRATION_BLOCKS):
            block, _overflowed = stream.read(blocksize)
            floors.append(_rms(block))
        threshold = max(sum(floors) / len(floors) * 4.0, 0.02)
        worker = threading.Thread(target=playback, daemon=True)
        worker.start()
        voiced = 0
        while not done.is_set():
            if stop_event is not None and stop_event.is_set():
                break
            block, _overflowed = stream.read(blocksize)
            voiced = voiced + 1 if _rms(block) >= threshold else 0
            if voiced >= _BARGE_IN_VOICED_BLOCKS:
                interrupted = True
                break
    if interrupted or (stop_event is not None and stop_event.is_set()):
        try:
            sounddevice.stop()
        except Exception:
            pass
    worker.join(5.0)
    if failure:
        raise failure[0]
    return interrupted


def _chunk_heading(text: str) -> str:
    return " ".join(text.split()[:6])


def _write_checkpoint(
    session_id: str | None, cursor: int, heading: str, status_value: str
) -> None:
    if not session_id:
        return
    try:
        conveyance.write_scratchpad(
            session_id, cursor=cursor, heading=heading, status=status_value
        )
    except OSError:
        pass


def _resegment(chunks: list[str]) -> list[str]:
    """Split remaining chunks to sentence granularity for the slower verb."""
    finer: list[str] = []
    for chunk in chunks:
        parts = [part for part in _SENTENCE_SPLIT.split(chunk) if part.strip()]
        finer.extend(parts or [chunk])
    return finer


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
        self._requested_device = device
        self._model_override = model
        self._status = status
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
        try:
            segments = self._decode(audio)
        except Exception as exc:
            if self.tier.device != "cuda" or self._requested_device == "cuda":
                raise ListenError(
                    f"transcription failed on {self.tier.describe()}: {exc}"
                ) from exc
            fallback = detect_tier("cpu", self._model_override)
            self._status(
                f"stt tier degraded: {self.tier.describe()} failed ({exc}); "
                f"falling back to {fallback.describe()}"
            )
            try:
                self._whisper = self._load(fallback)
                self.tier = fallback
                segments = self._decode(audio)
            except Exception as fallback_exc:
                raise ListenError(
                    f"transcription failed on {fallback.describe()}: {fallback_exc}"
                ) from fallback_exc
        return " ".join(
            part for part in (segment.text.strip() for segment in segments) if part
        )

    def _decode(self, audio):
        segments, _info = self._whisper.transcribe(
            audio,
            beam_size=5,
            hotwords=HOTWORDS,
            vad_filter=True,
        )
        return segments


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


def _wait_for_trigger(keys: _RawKeys, trigger_key: str | None) -> str | None:
    while True:
        key = keys.read_key(None)
        if key is None or trigger_key is None or key == trigger_key:
            return key


def _report_level(block, on_level: Callable[[float], None] | None) -> None:
    if on_level is not None:
        on_level(_rms(block))


def _record_push_to_talk(
    keys: _RawKeys,
    trigger_key: str | None = None,
    on_level: Callable[[float], None] | None = None,
    on_recording: Callable[[], None] | None = None,
) -> "object | None":
    """Record while a key is held: terminal auto-repeat keeps the take alive,
    and a repeat gap longer than the release window ends it on POSIX. Native
    Windows polls the physical key state so its configurable repeat delay cannot
    truncate the start of an utterance."""
    keys.drain()
    active_key = _wait_for_trigger(keys, trigger_key)
    if active_key is None:
        return None
    physical_state = keys.is_pressed(active_key)
    sounddevice = _sounddevice()
    blocksize = int(_BLOCK_SECONDS * SAMPLE_RATE)
    chunks = []
    last_key = time.monotonic()
    started = last_key
    if on_recording is not None:
        on_recording()
    with sounddevice.InputStream(
        samplerate=SAMPLE_RATE, channels=1, dtype="float32", blocksize=blocksize
    ) as stream:
        while True:
            block, _overflowed = stream.read(blocksize)
            chunks.append(block.copy())
            _report_level(block, on_level)
            if physical_state is None:
                while keys.read_key(0) is not None:
                    last_key = time.monotonic()
            else:
                physical_state = keys.is_pressed(active_key)
            now = time.monotonic()
            if physical_state is False:
                break
            if physical_state is None and now - last_key > _KEY_RELEASE_SECONDS:
                break
            if now - started > _MAX_UTTERANCE_SECONDS:
                break
    keys.drain()
    return _finish(chunks)


def _record_push_toggle(
    keys: _RawKeys,
    trigger_key: str | None = None,
    on_level: Callable[[float], None] | None = None,
    on_recording: Callable[[], None] | None = None,
    start_immediately: bool = False,
) -> "object | None":
    """Tap to start recording, tap again to send."""
    keys.drain()
    if not start_immediately and _wait_for_trigger(keys, trigger_key) is None:
        return None
    keys.drain()
    sounddevice = _sounddevice()
    blocksize = int(_BLOCK_SECONDS * SAMPLE_RATE)
    chunks = []
    started = time.monotonic()
    if on_recording is not None:
        on_recording()
    with sounddevice.InputStream(
        samplerate=SAMPLE_RATE, channels=1, dtype="float32", blocksize=blocksize
    ) as stream:
        while True:
            block, _overflowed = stream.read(blocksize)
            chunks.append(block.copy())
            _report_level(block, on_level)
            key = keys.read_key(0)
            if key is not None and (trigger_key is None or key == trigger_key):
                break
            if time.monotonic() - started > _MAX_UTTERANCE_SECONDS:
                break
    keys.drain()
    return _finish(chunks)


def _record_always_on(
    on_level: Callable[[float], None] | None = None,
    on_recording: Callable[[], None] | None = None,
    stop_event: "threading.Event | None" = None,
) -> "object | None":
    """Hands-free capture: calibrate the noise floor, trigger on speech
    energy, and end the utterance after a trailing pause.

    A set *stop_event* aborts at the next audio block so an idle hands-free
    session — blocked on the microphone rather than on a key source — can still
    unwind promptly when the dashboard asks it to stop.
    """
    stopping = lambda: stop_event is not None and stop_event.is_set()
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
            if stopping():
                return None
            block, _overflowed = stream.read(blocksize)
            ambient_samples.append(_rms(block))
        noise_floor = sum(ambient_samples) / len(ambient_samples)
        threshold = max(noise_floor * 3.0, 0.01)
        preroll: list = []
        chunks: list = []
        silent_blocks = 0
        while True:
            if stopping():
                return None
            block, _overflowed = stream.read(blocksize)
            level = _rms(block)
            if chunks:
                _report_level(block, on_level)
            if not chunks:
                preroll.append(block.copy())
                if len(preroll) > preroll_blocks:
                    preroll.pop(0)
                if level >= threshold:
                    chunks = list(preroll)
                    silent_blocks = 0
                    if on_recording is not None:
                        on_recording()
                    _report_level(block, on_level)
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
    command = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10"]
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


def build_claude_command(
    text,
    session_id,
    *,
    remote=None,
    remote_cwd=None,
    stream=False,
    permission="off",
) -> list[str]:
    fmt = "stream-json" if stream else "json"
    if remote:
        inner = f"claude -p {shlex.quote(text)} --output-format {fmt}"
        if stream:
            inner += " --verbose"
        if session_id:
            inner += f" --resume {shlex.quote(session_id)}"
        if permission == "skip":
            inner += " --dangerously-skip-permissions"
        elif permission in ("acceptEdits", "bypassPermissions"):
            inner += f" --permission-mode {permission}"
        return _ssh_base(remote) + [_remote_shell_command(inner, remote_cwd)]

    claude = shutil.which("claude")
    if claude is None:
        raise ListenError(
            "the claude CLI is not on PATH; install Claude Code, pass "
            "--remote user@host to run it on a server, or use --tmux-pane"
        )
    command = [claude, "-p", text, "--output-format", fmt]
    if stream:
        command += ["--verbose"]
    if session_id:
        command += ["--resume", session_id]
    if permission == "skip":
        command += ["--dangerously-skip-permissions"]
    elif permission in ("acceptEdits", "bypassPermissions"):
        command += ["--permission-mode", permission]
    return command


def _prompt_claude(
    text: str,
    session_id: str | None,
    remote: str | None = None,
    remote_cwd: str | None = None,
    on_wait: Callable[[], None] | None = None,
    on_event: "Callable[[dict], None] | None" = None,
    stop_event: "threading.Event | None" = None,
    permission: str = "off",
) -> tuple[str, str | None]:
    """Primary injection path (D-3): drive a claude -p session, resuming it
    across turns; the reply arrives as structured JSON.

    With *remote* set (``user@host``) the claude process runs on that host over
    SSH — the microphone, speech-to-text and spoken reply stay local, while
    Claude Code runs on the server. The remote command runs through a login
    shell so the server's normal PATH finds the ``claude`` CLI. When
    *remote_cwd* is set, Claude starts in that safely quoted project directory.

    When *on_event* is given, the session runs with ``--output-format
    stream-json`` and every activity event (tool calls, thinking, results) is
    surfaced live for the dashboard's session mirror; the spoken reply still
    comes only from the authoritative final ``result`` event.
    """
    command = build_claude_command(
        text,
        session_id,
        remote=remote,
        remote_cwd=remote_cwd,
        stream=on_event is not None,
        permission=permission,
    )
    if on_event is not None:
        return _consume_stream(command, on_event, on_wait, remote, stop_event)
    try:
        result = _run_captured(command, on_wait=on_wait)
    except (OSError, subprocess.SubprocessError) as exc:
        raise ListenError(f"could not run claude -p: {exc}") from exc
    stdout = result.stdout if isinstance(result.stdout, str) else ""
    stderr = result.stderr if isinstance(result.stderr, str) else ""
    if result.returncode != 0:
        detail = stderr.strip() or f"exit {result.returncode}"
        where = f" on {remote}" if remote else ""
        raise ListenError(
            f"claude -p failed{where}: {detail}"
            + (
                "  (remote needs passwordless SSH keys and the claude CLI installed there)"
                if remote
                else ""
            )
        )
    if not stdout.strip():
        raise ListenError("claude -p returned no JSON output")
    try:
        payload = json.loads(stdout)
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


def _run_captured(
    command: list[str], on_wait: Callable[[], None] | None = None
) -> subprocess.CompletedProcess:
    kwargs = {
        "stdin": subprocess.DEVNULL,
        "stdout": subprocess.PIPE,
        "stderr": subprocess.PIPE,
        "text": True,
        "encoding": "utf-8",
        "errors": "replace",
    }
    if on_wait is None:
        result = subprocess.run(command, **kwargs)
        return subprocess.CompletedProcess(
            command,
            result.returncode,
            result.stdout if isinstance(result.stdout, str) else "",
            result.stderr if isinstance(result.stderr, str) else "",
        )

    process = subprocess.Popen(command, **kwargs)
    captured: list[str | None] = [None, None]
    failure: list[BaseException] = []

    def communicate() -> None:
        try:
            captured[:] = process.communicate()
        except BaseException as exc:
            failure.append(exc)

    worker = threading.Thread(target=communicate, daemon=True)
    worker.start()
    try:
        while worker.is_alive():
            on_wait()
            worker.join(0.1)
    except BaseException:
        process.terminate()
        worker.join(2.0)
        if worker.is_alive():
            process.kill()
            worker.join()
        raise
    if failure:
        raise failure[0]
    return subprocess.CompletedProcess(
        command,
        process.returncode,
        captured[0] if isinstance(captured[0], str) else "",
        captured[1] if isinstance(captured[1], str) else "",
    )


def _reap(process: "subprocess.Popen", timeout: float = 2.0) -> None:
    """Terminate a child and guarantee it is reaped: SIGTERM, a bounded wait,
    then SIGKILL so a signal-resistant process can never hang the caller."""
    process.terminate()
    try:
        process.wait(timeout)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


def _consume_stream(
    command: list[str],
    on_event: "Callable[[dict], None]",
    on_wait: Callable[[], None] | None,
    remote: str | None,
    stop_event: "threading.Event | None" = None,
) -> tuple[str, str | None]:
    """Run a ``claude -p --output-format stream-json`` session, surfacing each
    NDJSON event live via *on_event*, and return only the authoritative final
    ``result`` text plus its session id.

    stdout is read on a side thread and fed through a queue so a set
    *stop_event* interrupts a mid-stream turn (the reader would otherwise block
    until Claude finished). stderr is drained on its own thread so a full pipe
    can never deadlock; unknown event types and malformed lines are tolerated;
    the child is always reaped with a bounded wait. No PTY is used, so the
    stream stays clean NDJSON even over SSH.
    """
    try:
        process = subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise ListenError(f"could not run claude -p: {exc}") from exc

    stderr_chunks: list[str] = []

    def drain_stderr() -> None:
        if process.stderr is not None:
            for chunk in process.stderr:
                stderr_chunks.append(chunk)

    sentinel = object()
    stdout_lines: "queue.Queue[object]" = queue.Queue()

    def read_stdout() -> None:
        try:
            if process.stdout is not None:
                for line in process.stdout:
                    stdout_lines.put(line)
        finally:
            stdout_lines.put(sentinel)

    stderr_worker = threading.Thread(target=drain_stderr, daemon=True)
    reader = threading.Thread(target=read_stdout, daemon=True)
    stderr_worker.start()
    reader.start()

    result_text = ""
    session_id: str | None = None
    is_error = False
    outcome = "eof"
    try:
        while True:
            if stop_event is not None and stop_event.is_set():
                outcome = "stopped"
                break
            try:
                line = stdout_lines.get(timeout=0.1)
            except queue.Empty:
                if on_wait is not None:
                    on_wait()
                continue
            if line is sentinel:
                outcome = "eof"
                break
            if on_wait is not None:
                on_wait()
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(event, dict):
                continue
            try:
                on_event(event)
            except Exception:
                pass
            etype = event.get("type")
            new_session = event.get("session_id")
            if isinstance(new_session, str) and new_session:
                session_id = new_session
            if etype == "result":
                is_error = bool(event.get("is_error"))
                reply = event.get("result")
                result_text = reply if isinstance(reply, str) else ""
                outcome = "result"
                break
    except BaseException:
        _reap(process)
        reader.join(1.0)
        stderr_worker.join(1.0)
        raise

    if outcome == "eof":
        process.wait()  # ended on its own; keep the real exit code
    else:
        _reap(process)  # stopped or done early — the child may still be running
    reader.join(1.0)
    stderr_worker.join(1.0)

    if outcome == "stopped":
        return "", session_id
    if outcome == "eof":
        detail = "".join(stderr_chunks).strip() or f"exit {process.returncode}"
        where = f" on {remote}" if remote else ""
        if process.returncode not in (0, None):
            raise ListenError(
                f"claude -p failed{where}: {detail}"
                + (
                    "  (remote needs passwordless SSH keys and the claude CLI installed there)"
                    if remote
                    else ""
                )
            )
        raise ListenError("claude -p stream ended without a result event")
    if is_error:
        raise ListenError(result_text.strip() or "claude reported an error")
    return result_text, session_id


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
    on_level: Callable[[float], None] | None = None,
    on_phase: Callable[[str], None] | None = None,
    on_progress: Callable[[], None] | None = None,
    trigger_key: str | None = None,
    start_recording: bool = False,
    keys: "_RawKeys | None" = None,
    stop_event: "threading.Event | None" = None,
    on_event: "Callable[[dict], None] | None" = None,
    on_session: "Callable[[str], None] | None" = None,
    permission: str = "off",
) -> None:
    """Drive the capture → transcribe → inject → reply loop until interrupted.

    When *remote* (``user@host``) is set, Claude Code runs on that host over
    SSH while the microphone, transcription and spoken reply stay on this
    machine — the remote/SSH pattern (voice where you sit, compute on the
    server). *remote_cwd* selects the project directory for remote ``claude``
    sessions.

    *keys* lets a caller inject its own key source instead of opening a raw
    terminal reader. The interactive dashboard passes a queue-backed source so
    Textual keeps sole ownership of the TTY while still feeding the trigger key.
    *stop_event*, when set, unwinds the loop cleanly between (and during)
    captures — the graceful path for hands-free ``always-on`` sessions, whose
    microphone read never sees the key source's shutdown sentinel.
    """
    set_phase = on_phase or (lambda _phase: None)
    set_phase("starting")
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
    if keys is None and mode in ("push-to-talk", "push-toggle"):
        keys = _RawKeys()
    first_capture = True
    wake_state: dict = {}

    barge_opted_in = _setting(config.barge_in_enabled, False)
    barge = barge_in_active(barge_opted_in, headphones_present()) if barge_opted_in else False
    if barge:
        status("barge-in armed: speak over a reply to interrupt it (headphones detected)")
    elif barge_opted_in:
        status("barge-in is on but no headphones were detected; staying half-duplex")

    catalog: list[dict] = []
    pending_fire: "tuple[dict, str] | None" = None
    pending_text: str | None = None

    def handle_event(event: dict) -> None:
        # The session's system/init event advertises the voice-fireable
        # commands; refresh the catalog while preserving the user-owned flags.
        if (
            isinstance(event, dict)
            and event.get("type") == "system"
            and event.get("subtype") == "init"
        ):
            try:
                catalog[:] = command_catalog.load_catalog(event)
                command_catalog.save_flags(catalog)
            except Exception:
                pass
        if on_event is not None:
            on_event(event)

    stream_events = handle_event if on_event is not None else None

    def capture():
        nonlocal first_capture
        set_phase("ready")
        recording = lambda: set_phase("recording")
        if mode == "always-on":
            if not _wake_gate(speak, status, stop_event, wake_state):
                return None
            return _record_always_on(
                on_level=on_level, on_recording=recording, stop_event=stop_event
            )
        with keys:
            if mode == "push-to-talk":
                return _record_push_to_talk(
                    keys,
                    trigger_key=trigger_key,
                    on_level=on_level,
                    on_recording=recording,
                )
            record_now = start_recording and first_capture
            first_capture = False
            return _record_push_toggle(
                keys,
                trigger_key=trigger_key,
                on_level=on_level,
                on_recording=recording,
                start_immediately=record_now,
            )

    def deliver(text: str) -> bool:
        """Speak *text*; True when the operator barged in over the playback."""
        if barge:
            return _speak_interruptible(speak, text, stop_event=stop_event)
        speak(text)
        return False

    def deliver_reply(dialogue: str, active_session: str | None) -> str | None:
        """The voice-conveyance checkpoint loop.

        Long replies are spoken gradually: after each chunk the cursor,
        heading, and status persist to the session scratchpad, the microphone
        reopens, and an exact feedback verb steers delivery. A content-bearing
        utterance heard at a checkpoint is returned so it resumes through the
        same session as the next ordinary turn.
        """
        chunks = conveyance.chunk(dialogue)
        if once or len(chunks) <= 1:
            set_phase("speaking")
            deliver(dialogue)
            return None
        cursor = 0
        while cursor < len(chunks):
            current = chunks[cursor]
            _write_checkpoint(active_session, cursor, _chunk_heading(current), "delivering")
            set_phase("speaking")
            deliver(current)
            if cursor == len(chunks) - 1:
                break
            if stop_event is not None and stop_event.is_set():
                _write_checkpoint(active_session, cursor, _chunk_heading(current), "stopped")
                return None
            audio = capture()  # the checkpoint: the mic reopens between chunks
            if audio is None:
                if stop_event is not None and stop_event.is_set():
                    _write_checkpoint(
                        active_session, cursor, _chunk_heading(current), "stopped"
                    )
                    return None
                cursor += 1
                continue
            set_phase("transcribing")
            heard = transcriber.transcribe(audio).strip()
            if not heard:
                cursor += 1
                continue
            verb = detect_verb(heard)
            if verb is None:
                _write_checkpoint(active_session, cursor, _chunk_heading(current), "steered")
                return heard
            echo(f"you: {verb}")
            if verb == "continue":
                cursor += 1
            elif verb == "repeat":
                continue
            elif verb == "back":
                cursor = max(0, cursor - 1)
            elif verb == "skip":
                cursor += 2
            elif verb == "stop":
                _write_checkpoint(active_session, cursor, _chunk_heading(current), "stopped")
                return None
            elif verb == "slower":
                chunks[cursor:] = _resegment(chunks[cursor:])
            elif verb == "expand":
                _write_checkpoint(active_session, cursor, _chunk_heading(current), "steered")
                return "Expand on the point you just made, in more detail."
            elif verb == "steer":
                _write_checkpoint(active_session, cursor, _chunk_heading(current), "steered")
                return None
        _write_checkpoint(
            active_session, len(chunks) - 1, _chunk_heading(chunks[-1]), "done"
        )
        return None

    while True:
        if stop_event is not None and stop_event.is_set():
            return
        if pending_text is not None:
            text = pending_text
            pending_text = None
        else:
            audio = capture()
            if audio is None:
                continue
            set_phase("transcribing")
            text = transcriber.transcribe(audio).strip()
            if not text:
                continue
        echo(f"you: {text}")
        if tmux_pane:
            set_phase("thinking")
            _send_tmux(tmux_pane, text, remote=remote)
            status(f"sent to tmux pane {tmux_pane}; the live TUI owns the reply")
            set_phase("ready")
            if once:
                return
            continue
        # Voice-activated commands: resolve, confirm, then fire the confirmed
        # command through the same resumed session with the operator posture.
        normalized = " ".join(text.casefold().split())
        if pending_fire is not None:
            record, args = pending_fire
            pending_fire = None
            if normalized in _CANCEL_WORDS:
                status(f"cancelled /{record['id']}")
                set_phase("ready")
                if once:
                    return
                continue
            if normalized in _FIRE_WORDS:
                text = _fire_text(record, args)
                record["fire_count"] = int(record.get("fire_count", 0)) + 1
                try:
                    command_catalog.save_flags(catalog)
                except OSError:
                    pass
                echo(f"you: {text}")
                status(f"firing {text} into the session")
            # Anything else is ordinary content — fall through unchanged.
        else:
            resolved = _resolve_command(text, catalog, status)
            if resolved is not None:
                pending_fire = resolved
                record, args = resolved
                confirmation = f"Firing {_fire_text(record, args)}. Say go, or cancel."
                status(confirmation)
                set_phase("speaking")
                speak(confirmation)
                set_phase("ready")
                if once:
                    return
                continue
        set_phase("thinking")
        reply, session_id = _prompt_claude(
            text,
            session_id,
            remote=remote,
            remote_cwd=remote_cwd,
            on_wait=on_progress,
            on_event=stream_events,
            stop_event=stop_event,
            permission=permission,
        )
        if on_session is not None and session_id is not None:
            on_session(session_id)
        dialogue = speakable(reply)
        if dialogue:
            echo(f"claude: {dialogue}")
            pending_text = deliver_reply(dialogue, session_id)
        set_phase("ready")
        if once:
            return
