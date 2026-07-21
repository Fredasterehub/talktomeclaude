"""Command-line entry point for talktomeclaude."""

from pathlib import Path

import click

from talktomeclaude.transcript import iter_dialogue
from talktomeclaude.tts import (
    BUNDLED_VOICES,
    TTSError,
    default_voice,
    synthesize,
    voices_dir,
)


@click.group()
def main() -> None:
    """Use voice as a medium for Claude Code."""


@main.command()
@click.argument("text")
@click.option(
    "--out",
    "out_path",
    type=click.Path(dir_okay=False, writable=True, path_type=Path),
    default=None,
    help="Write the synthesized audio to this WAV file instead of playing it.",
)
@click.option(
    "--voice",
    "voice_name",
    default=None,
    help="Bundled voice to use (see the voices command); defaults to the best tier the hardware carries.",
)
def speak(text: str, out_path: Path | None, voice_name: str | None) -> None:
    """Synthesize TEXT to speech, fully locally, via the Piper engine."""
    playback = out_path is None
    if playback:
        import tempfile

        out_path = Path(tempfile.mkstemp(prefix="talktomeclaude-", suffix=".wav")[1])
    try:
        voice = synthesize(
            text, out_path, voice_name, on_status=lambda message: click.echo(message, err=True)
        )
    except TTSError as exc:
        raise click.ClickException(str(exc)) from exc
    if not playback:
        click.echo(f"wrote {out_path} (voice: {voice.name})")
        return
    try:
        _play_wav(out_path)
    finally:
        out_path.unlink(missing_ok=True)


def _play_wav(path: Path) -> None:
    import wave

    try:
        import numpy
        import sounddevice
    except (ImportError, OSError) as exc:
        raise click.ClickException(
            f"audio playback unavailable ({exc}); use --out to write a WAV file"
        ) from exc
    with wave.open(str(path), "rb") as wav:
        frames = wav.readframes(wav.getnframes())
        samples = numpy.frombuffer(frames, dtype=numpy.int16)
        channels = wav.getnchannels()
        if channels > 1:
            samples = samples.reshape(-1, channels)
        sounddevice.play(samples, samplerate=wav.getframerate(), blocking=True)


@main.command()
@click.option(
    "--download",
    is_flag=True,
    help="Fetch all voices into the local cache now, instead of on first use.",
)
def voices(download: bool) -> None:
    """List the available voices and their licenses.

    Voices are fetched from the Hugging Face Hub on first use and cached
    locally; run with --download to pre-fetch them all up front.
    """
    from talktomeclaude.tts import cache_voices_dir, is_available, voice_files

    directory = voices_dir()
    cache = cache_voices_dir()

    if download:
        for voice in BUNDLED_VOICES:
            try:
                model_path, _config = voice_files(
                    voice, on_status=lambda message: click.echo(message, err=True)
                )
            except TTSError as exc:
                raise click.ClickException(str(exc)) from exc
            click.echo(f"{voice.name}: ready  ({model_path})")
        return

    default = default_voice(directory)
    for voice in BUNDLED_VOICES:
        if voice.is_installed(directory):
            status = "bundled"
        elif voice.is_installed(cache):
            status = "cached"
        else:
            status = "on-demand"
        marker = "  (default)" if voice.name == default.name else ""
        click.echo(
            f"{voice.name}  [{voice.language}, {voice.quality}, {status}]  "
            f"license: {voice.license}  ({voice.provenance}){marker}"
        )


@main.command()
@click.argument(
    "audio",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.option(
    "--device",
    type=click.Choice(["auto", "cuda", "cpu"]),
    default="auto",
    show_default=True,
    help="Hardware tier: auto-detect picks the GPU tier when CUDA is present.",
)
@click.option(
    "--model",
    "model_name",
    default=None,
    help="Override the Whisper model for the active tier (e.g. large-v3, small.en).",
)
@click.option(
    "--show-tier",
    is_flag=True,
    help="Print the active hardware tier without transcribing, then exit.",
)
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    help="Report the active tier and any degradation on stderr.",
)
def transcribe(audio: Path, device: str, model_name: str | None, show_tier: bool, verbose: bool) -> None:
    """Transcribe an AUDIO file locally with Whisper-class speech-to-text.

    Auto-detects the hardware tier: a CUDA GPU carries the largest fluid
    model; CPU-only machines use the most accurate model that stays fluid.
    Tier fallbacks are always reported on stderr — never silent.
    """
    from talktomeclaude.stt import STTError, detect_tier, transcribe_file

    if show_tier:
        try:
            click.echo(detect_tier(device, model_name).describe())
        except STTError as exc:
            raise click.ClickException(str(exc)) from exc
        return

    def report(message: str) -> None:
        if verbose or "degraded" in message:
            click.echo(message, err=True)

    try:
        text, _tier = transcribe_file(audio, device, model_name, on_status=report)
    except STTError as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(text)


@main.command(name="filter")
@click.argument("transcript", type=click.File("r", encoding="utf-8"))
def filter_command(transcript) -> None:
    """Extract spoken dialogue from a transcript.

    Reads a Claude Code JSONL transcript (or - for stdin) and prints only
    Claude's assistant prose — never tool calls, tool results, code blocks,
    or thinking.
    """
    for dialogue in iter_dialogue(transcript):
        click.echo(dialogue)


@main.command()
@click.option(
    "--mode",
    type=click.Choice(["always-on", "push-to-talk", "push-toggle"]),
    default=None,
    help="Recording mode for this run; defaults to the persisted recording-mode setting.",
)
@click.option(
    "--session",
    "session_id",
    default=None,
    help="Claude Code session id to resume (claude -p --resume); omit to start a fresh session.",
)
@click.option(
    "--tmux-pane",
    default=None,
    help="Type transcripts into this live tmux pane via `tmux send-keys` instead of driving claude -p.",
)
@click.option(
    "--remote",
    "remote",
    default=None,
    help="Run Claude Code on a remote host over SSH (user@host): the mic, transcription and "
    "spoken reply stay local; Claude runs on the server. Persist it with "
    "`config set remote user@host`. Needs passwordless SSH keys and the claude CLI on the remote.",
)
@click.option(
    "--remote-cwd",
    default=None,
    help="Start remote Claude Code in this project directory for this run; "
    "defaults to the persisted remote-cwd setting.",
)
@click.option(
    "--device",
    type=click.Choice(["auto", "cuda", "cpu"]),
    default="auto",
    show_default=True,
    help="Hardware tier: auto-detect picks the GPU tier when CUDA is present.",
)
@click.option(
    "--model",
    "model_name",
    default=None,
    help="Override the Whisper model for the active tier (e.g. large-v3, small.en).",
)
@click.option("--once", is_flag=True, help="Handle a single utterance, then exit.")
def listen(
    mode: str | None,
    session_id: str | None,
    tmux_pane: str | None,
    remote: str | None,
    remote_cwd: str | None,
    device: str,
    model_name: str | None,
    once: bool,
) -> None:
    """Listen to the microphone and drive Claude Code by voice.

    Captures speech in the selected recording mode, transcribes it locally,
    and injects the text into Claude Code. The primary path drives the
    loop's own `claude -p --resume` session and speaks each reply; with
    --tmux-pane the transcript is typed into a live interactive TUI via
    `tmux send-keys` instead. One driver per session: never point the voice
    loop and a live interactive window at the same session.

    Runs anywhere: on the machine you sit at (mic, Claude, speakers all local —
    the default), or split with --remote user@host so the voice stays local
    while Claude Code runs on a server (the remote/SSH pattern).

    \b
    Recording modes (persist a default with `config set recording-mode`):
      always-on     hands-free capture, VAD-gated; an utterance ends at a pause
      push-to-talk  hold any key to record; releasing it sends the utterance
      push-toggle   tap a key to start recording, tap again to send
    """
    from talktomeclaude import config
    from talktomeclaude.listen import ListenError, run_listen

    active_mode = mode or config.recording_mode()
    active_remote = remote or config.remote()
    if remote_cwd is not None and not active_remote:
        raise click.ClickException("--remote-cwd requires --remote or a persisted remote")
    active_remote_cwd = (
        remote_cwd if remote_cwd is not None else config.remote_cwd()
    ) if active_remote else None

    def speak_reply(text: str) -> None:
        if not config.voice_assist_enabled():
            return
        import tempfile

        wav_path = Path(tempfile.mkstemp(prefix="talktomeclaude-listen-", suffix=".wav")[1])
        try:
            synthesize(text, wav_path, None)
            _play_wav(wav_path)
        except (TTSError, click.ClickException):
            pass
        finally:
            wav_path.unlink(missing_ok=True)

    try:
        run_listen(
            mode=active_mode,
            session_id=session_id,
            tmux_pane=tmux_pane,
            device=device,
            model=model_name,
            once=once,
            echo=click.echo,
            speak=speak_reply,
            status=lambda message: click.echo(message, err=True),
            remote=active_remote,
            remote_cwd=active_remote_cwd,
        )
    except ListenError as exc:
        raise click.ClickException(str(exc)) from exc
    except KeyboardInterrupt:
        click.echo("listen stopped", err=True)


@main.group()
def config() -> None:
    """Read and write settings persisted across invocations."""


@config.command("set")
@click.argument("key")
@click.argument("value")
def config_set(key: str, value: str) -> None:
    """Persist KEY = VALUE.

    Known keys: recording-mode (always-on, push-to-talk, push-toggle),
    voice-assist (on, off), remote (user@host, or "local"/"none" to clear),
    and remote-cwd (remote project path, or "home"/"none" to clear).
    """
    from talktomeclaude import config as settings

    if key == "recording-mode":
        try:
            settings.set_recording_mode(value)
        except ValueError as exc:
            raise click.ClickException(str(exc)) from exc
    elif key == "voice-assist":
        if value not in ("on", "off"):
            raise click.ClickException(
                f"invalid voice-assist value {value!r}: expected on or off"
            )
        settings.set_voice_assist(value == "on")
    elif key == "remote":
        settings.set_remote(None if value.lower() in ("", "local", "none", "off") else value)
    elif key == "remote-cwd":
        settings.set_remote_cwd(
            None if value.lower() in ("", "home", "none", "off") else value
        )
    else:
        raise click.ClickException(
            f"unknown setting {key!r}: expected recording-mode, voice-assist, remote, or remote-cwd"
        )
    click.echo(f"{key} = {value}")


@config.command("get")
@click.argument("key")
def config_get(key: str) -> None:
    """Print the persisted value of KEY (or its default)."""
    from talktomeclaude import config as settings

    if key == "recording-mode":
        click.echo(settings.recording_mode())
    elif key == "voice-assist":
        click.echo("on" if settings.voice_assist_enabled() else "off")
    elif key == "remote":
        click.echo(settings.remote() or "local")
    elif key == "remote-cwd":
        click.echo(settings.remote_cwd() or "home")
    else:
        raise click.ClickException(
            f"unknown setting {key!r}: expected recording-mode, voice-assist, remote, or remote-cwd"
        )


@main.command()
@click.argument("state", type=click.Choice(["on", "off", "status"]))
def assist(state: str) -> None:
    """Switch voice-assist (spoken replies) on or off, or query its state.

    With voice-assist off the Stop hook stays silent; this is the full-mute
    switch, persisted across invocations.
    """
    from talktomeclaude import config

    if state == "status":
        click.echo("on" if config.voice_assist_enabled() else "off")
        return
    config.set_voice_assist(state == "on")
    click.echo(f"voice-assist {state}")


@main.group()
def hook() -> None:
    """Entry points invoked by the Claude Code plugin hooks."""


@hook.command()
@click.option(
    "--dry-run",
    is_flag=True,
    help="Print what would be spoken as a SPEAK: line instead of playing audio.",
)
def stop(dry_run: bool) -> None:
    """Handle a Stop event: speak Claude's final reply.

    Reads the hook event JSON from stdin and speaks the dialogue of
    last_assistant_message through the local TTS voice. Honors the
    voice-assist mute switch and never blocks Claude Code — every
    outcome exits 0.
    """
    import sys

    from talktomeclaude import config
    from talktomeclaude.hook import read_stop_event, stop_dialogue

    event = read_stop_event(sys.stdin)
    if event is None or not config.voice_assist_enabled():
        return
    dialogue = stop_dialogue(event)
    if not dialogue:
        return
    if dry_run:
        click.echo("SPEAK: " + " ".join(dialogue.split()))
        return
    import tempfile

    wav_path = Path(tempfile.mkstemp(prefix="talktomeclaude-hook-", suffix=".wav")[1])
    try:
        synthesize(dialogue, wav_path, None)
        _play_wav(wav_path)
    except (TTSError, click.ClickException):
        pass
    finally:
        wav_path.unlink(missing_ok=True)
