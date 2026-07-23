"""Voice-clone creation UI: source selection, acquisition, mandatory review.

Reachable from the dashboard (C) and from onboarding's first-voice step. Every
reference — explicit file, fresh recording, or an automatically selected
YouTube segment — passes the audition-and-confirm review before any voice is
registered.
"""

from __future__ import annotations

import subprocess
import tempfile
import wave
from pathlib import Path
from typing import Callable

from textual.app import ComposeResult
from textual.binding import Binding
from textual.screen import Screen
from textual.widgets import Input, OptionList, Static

from talktomeclaude import registry

review_required: bool = True

_SEGMENT_SECONDS = 15.0
_RECORD_SECONDS = 15.0

_SOURCE_OPTIONS = (
    "Audio file on disk",
    "Record from the microphone",
    "YouTube link (agent cut)",
    "Cancel",
)


def play_reference(path: Path) -> None:
    """Default audition player: local WAV playback through sounddevice."""
    try:
        import numpy
        import sounddevice
    except (ImportError, OSError) as exc:
        raise RuntimeError(f"audio playback unavailable ({exc})") from exc
    with wave.open(str(path), "rb") as handle:
        frames = handle.readframes(handle.getnframes())
        samples = numpy.frombuffer(frames, dtype=numpy.int16)
        channels = handle.getnchannels()
        if channels > 1:
            samples = samples.reshape(-1, channels)
        sounddevice.play(samples, samplerate=handle.getframerate(), blocking=True)


def download_youtube_audio(url: str, workdir: Path) -> Path:
    """Download the audio track of a consented YouTube source with yt-dlp."""
    from talktomeclaude.clone import ytdlp_command

    dest = workdir / "source.m4a"
    command = ytdlp_command(url, str(dest))
    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
    except FileNotFoundError as exc:
        raise RuntimeError("yt-dlp is required for the YouTube source") from exc
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or "").strip() or f"exit {exc.returncode}"
        raise RuntimeError(f"could not download the source audio: {detail}") from exc
    if not dest.is_file():
        raise RuntimeError("the YouTube download produced no audio file")
    return dest


def probe_duration(path: Path) -> float | None:
    """Media duration in seconds via ffprobe, or None when unknown."""
    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                str(path),
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        return float(result.stdout.strip())
    except Exception:
        return None


def cut_segment(
    source: Path, dest: Path, *, start: float, seconds: float = _SEGMENT_SECONDS
) -> Path:
    """Cut the automatically selected reference segment with ffmpeg."""
    command = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-ss", str(start), "-t", str(seconds),
        "-i", str(source), "-ac", "1", "-ar", "24000",
        str(dest),
    ]
    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
    except FileNotFoundError as exc:
        raise RuntimeError("ffmpeg is required to cut a reference segment") from exc
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or "").strip() or f"exit {exc.returncode}"
        raise RuntimeError(f"could not cut the reference segment: {detail}") from exc
    if not dest.is_file() or dest.stat().st_size <= 44:
        raise RuntimeError("segment cutting produced no audio")
    return dest


def auto_select_segment(source: Path, workdir: Path) -> Path:
    """Automatically pick a review candidate: a clean-length segment starting
    a tenth of the way in, skipping intros. The pick is only a candidate —
    the review screen still requires audition and confirmation."""
    duration = probe_duration(source)
    start = 0.0
    if duration is not None and duration > _SEGMENT_SECONDS:
        start = min(duration * 0.1, duration - _SEGMENT_SECONDS)
    return cut_segment(source, workdir / "segment.wav", start=start)


class CloneScreen(Screen[bool]):
    """Guided voice creation gated on the mandatory clip review.

    Constructed bare it asks for a name and a source; constructed with a
    name and reference it opens directly on the review step (the path the
    CLI wizard and tests use). No voice is ever created before the selected
    clip is auditioned and confirmed.
    """

    BINDINGS = [
        Binding("a", "audition", "Audition", show=False),
        Binding("c", "confirm", "Confirm", show=False),
        Binding("escape", "cancel", "Cancel", show=False, priority=True),
    ]

    def __init__(
        self,
        name: str | None = None,
        reference_path: str | Path | None = None,
        *,
        engine: str = "clone",
        ref_text: str = "",
        audition: Callable[[Path], None] | None = None,
    ) -> None:
        super().__init__()
        if engine not in {"clone", "f5"}:
            raise ValueError(f"unsupported clone engine: {engine}")
        self.voice_name = name
        self.reference_path = Path(reference_path) if reference_path is not None else None
        self.engine = engine
        self.ref_text = ref_text
        self._audition = audition if audition is not None else play_reference
        self._auditioned = False
        self.created_voice: registry.RegisteredVoice | None = None
        self._workdir: Path | None = None
        self._flow_error = ""
        self._busy_message = ""
        if name and reference_path is not None:
            self._step = "review"
        elif name:
            self._step = "source"
        else:
            self._step = "name"

    # ── step rendering ───────────────────────────────────────────────────────
    def compose(self) -> ComposeResult:
        yield from self._widgets_for(self._step)

    def _widgets_for(self, step: str):
        if step == "name":
            yield Static("Name the new voice")
            yield Input(id="clone-name", placeholder="voice name")
            yield Static(self._flow_error or "Enter Continue   ·   Esc Cancel")
        elif step == "source":
            yield Static(f"Choose a source for {self.voice_name!r} — own or consented audio only")
            yield OptionList(*_SOURCE_OPTIONS, id="clone-source")
            yield Static(
                self._flow_error
                or "Up/Down Choose   ·   Enter Select   ·   Esc Cancel"
            )
        elif step == "file":
            yield Static("Path to the reference audio file (~10-20s, clean, single speaker)")
            yield Input(id="clone-file", placeholder="/path/to/reference.wav")
            yield Static(self._flow_error or "Enter Continue   ·   Esc Cancel")
        elif step == "youtube":
            yield Static("YouTube URL — a segment is cut automatically, then reviewed")
            yield Input(id="clone-url", placeholder="https://youtu.be/…")
            yield Static(self._flow_error or "Enter Download   ·   Esc Cancel")
        elif step == "busy":
            yield Static(self._busy_message, id="clone-busy")
            yield Static("Esc Cancel")
        else:  # review
            yield Static("Review the auto-selected reference clip")
            yield Static(str(self.reference_path))
            yield OptionList(
                "Audition auto-selected clip",
                "Confirm clip and create voice",
                "Cancel",
                id="clone-review-actions",
            )
            yield Static(
                "Audition is mandatory before confirmation.",
                id="clone-review-status",
            )
            yield Static("Up/Down Choose   ·   Enter Select   ·   Esc Cancel")

    def on_mount(self) -> None:
        self._focus_step()

    def _show_step(self, step: str) -> None:
        self._step = step
        self.remove_children()
        self.mount_all(list(self._widgets_for(step)))
        self.call_after_refresh(self._focus_step)

    def _focus_step(self) -> None:
        if self._step == "review":
            actions = self.query_one("#clone-review-actions", OptionList)
            actions.highlighted = 0
            actions.focus()
        elif self._step == "source":
            source = self.query_one("#clone-source", OptionList)
            source.highlighted = 0
            source.focus()
        elif self._step in ("name", "file", "youtube"):
            self.query_one(Input).focus()

    def _fail_back(self, step: str, message: str) -> None:
        self._flow_error = message
        self._show_step(step)

    # ── acquisition ──────────────────────────────────────────────────────────
    def _ensure_workdir(self) -> Path:
        if self._workdir is None:
            self._workdir = Path(tempfile.mkdtemp(prefix="ttmc-clone-"))
        return self._workdir

    def on_input_submitted(self, event: Input.Submitted) -> None:
        value = event.value.strip()
        if event.input.id == "clone-name":
            if not value:
                self._fail_back("name", "A voice needs a name.")
                return
            self.voice_name = value
            self._flow_error = ""
            self._show_step("source")
        elif event.input.id == "clone-file":
            path = Path(value).expanduser()
            if not path.is_file():
                self._fail_back("file", f"Not a file: {path}")
                return
            self.reference_path = path
            self._flow_error = ""
            self._show_step("review")
        elif event.input.id == "clone-url":
            if not value:
                self._fail_back("youtube", "Enter a URL.")
                return
            self._flow_error = ""
            self._busy_message = "Downloading and cutting a reference segment…"
            self._show_step("busy")
            workdir = self._ensure_workdir()
            self.run_worker(
                lambda: self._acquire_youtube(value, workdir),
                thread=True,
                exit_on_error=False,
                group="clone-acquire",
            )

    def _acquire_youtube(self, url: str, workdir: Path) -> None:
        try:
            source = download_youtube_audio(url, workdir)
            segment = auto_select_segment(source, workdir)
        except RuntimeError as exc:
            self.app.call_from_thread(self._acquired, None, str(exc))
            return
        self.app.call_from_thread(self._acquired, segment, None)

    def _start_recording(self) -> None:
        self._flow_error = ""
        self._busy_message = f"Recording {_RECORD_SECONDS:.0f} seconds — speak now…"
        self._show_step("busy")
        workdir = self._ensure_workdir()
        self.run_worker(
            lambda: self._acquire_recording(workdir),
            thread=True,
            exit_on_error=False,
            group="clone-acquire",
        )

    def _acquire_recording(self, workdir: Path) -> None:
        from talktomeclaude import wizard

        try:
            clip = wizard.record_reference(
                workdir / "recording.wav", seconds=_RECORD_SECONDS
            )
        except wizard.WizardError as exc:
            self.app.call_from_thread(self._acquired, None, str(exc))
            return
        self.app.call_from_thread(self._acquired, Path(clip), None)

    def _acquired(self, path: Path | None, error: str | None) -> None:
        if error is not None or path is None:
            self._fail_back("source", error or "Acquisition failed.")
            return
        # Every automatically selected segment goes through the same
        # mandatory audition-and-confirm review before registration.
        self.reference_path = path
        self._auditioned = False
        self._show_step("review")

    # ── selection routing ────────────────────────────────────────────────────
    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        if event.option_list.id == "clone-source":
            if event.option_index == 0:
                self._flow_error = ""
                self._show_step("file")
            elif event.option_index == 1:
                self._start_recording()
            elif event.option_index == 2:
                self._flow_error = ""
                self._show_step("youtube")
            else:
                self.action_cancel()
        elif event.option_list.id == "clone-review-actions":
            if event.option_index == 0:
                self.action_audition()
            elif event.option_index == 1:
                self.action_confirm()
            else:
                self.action_cancel()

    # ── the mandatory review gate ────────────────────────────────────────────
    def _set_status(self, message: str) -> None:
        self.query_one("#clone-review-status", Static).update(message)

    def action_audition(self) -> None:
        if self._step != "review":
            return
        if not self.reference_path.is_file():
            self._set_status(f"Reference clip not found: {self.reference_path}")
            return
        try:
            self._audition(self.reference_path)
        except Exception as exc:
            self._set_status(f"Audition failed: {exc}")
            return
        self._auditioned = True
        self._set_status("Clip auditioned. Select Confirm to create the voice.")
        self.query_one("#clone-review-actions", OptionList).highlighted = 1

    def action_confirm(self) -> None:
        if self._step != "review":
            return
        if not self._auditioned:
            self._set_status("Audition the auto-selected clip before confirming it.")
            self.query_one("#clone-review-actions", OptionList).highlighted = 0
            return
        try:
            if self.engine == "f5":
                self.created_voice = registry.add_f5(
                    self.voice_name,
                    self.reference_path,
                    self.ref_text,
                )
            else:
                self.created_voice = registry.add_clone(
                    self.voice_name,
                    self.reference_path,
                )
        except Exception as exc:
            self._set_status(f"Voice creation failed: {exc}")
            return
        self.dismiss(True)

    def action_cancel(self) -> None:
        self.dismiss(False)
