"""Dependency-free terminal dashboard for the local voice loop."""

from __future__ import annotations

import math
import os
import shlex
import shutil
import subprocess
import sys
import textwrap
import time
from dataclasses import dataclass, field
from pathlib import PurePosixPath
from typing import Callable, TextIO

from talktomeclaude import config
from talktomeclaude.listen import (
    ListenError,
    _RawKeys,
    _remote_shell_command,
    _ssh_base,
    run_listen,
)

_LEVELS = " .:-=+*#"
_MODE_LABELS = {
    "always-on": "Always on",
    "push-to-talk": "Push to talk",
    "push-toggle": "Space toggle",
}
_PHASE_LABELS = {
    "ready": "READY",
    "starting": "STARTING",
    "recording": "RECORDING",
    "transcribing": "TRANSCRIBING",
    "thinking": "CLAUDE WORKING",
    "speaking": "SPEAKING",
    "error": "NEEDS ATTENTION",
}


class TUIError(RuntimeError):
    """Raised when the dashboard cannot start or complete an action."""


@dataclass
class DashboardState:
    remote: str | None
    remote_cwd: str | None
    mode: str
    voice_enabled: bool
    phase: str = "ready"
    notice: str = "Space starts a voice session"
    tier: str = "Hardware detection runs when the session starts"
    levels: list[float] = field(default_factory=lambda: [0.0] * 48)
    dialogue: list[tuple[str, str]] = field(default_factory=list)
    frame: int = 0
    reduced_motion: bool = False
    phase_started_at: float = field(default_factory=time.monotonic, repr=False)

    def add_level(self, level: float) -> None:
        normalized = max(0.0, min(1.0, level * 16.0))
        self.levels = [*self.levels[-47:], normalized]

    def add_dialogue(self, speaker: str, text: str) -> None:
        self.dialogue = [*self.dialogue[-5:], (speaker, text)]


def _fit(text: str, width: int) -> str:
    if width <= 0:
        return ""
    return text if len(text) <= width else text[: max(0, width - 3)] + "..."


def _waveform(state: DashboardState, width: int) -> str:
    width = max(1, width)
    if state.phase == "recording" and any(state.levels):
        values = state.levels[-width:]
        values = [0.0] * (width - len(values)) + values
        return "".join(_LEVELS[min(len(_LEVELS) - 1, int(value * 7))] for value in values)
    if state.reduced_motion:
        return "-" * width
    return "".join(
        _LEVELS[int((math.sin((index + state.frame) / 3.2) + 1.0) * 1.25)]
        for index in range(width)
    )


def _source(state: DashboardState) -> str:
    if state.remote is None:
        return "Local Claude Code"
    return f"{state.remote}:{state.remote_cwd or '~'}"


def _pad_canvas(lines: list[str], width: int, height: int) -> str:
    clipped = [_fit(line, width) for line in lines[:height]]
    clipped.extend("" for _ in range(max(0, height - len(clipped))))
    return "\n".join(clipped)


def render_dashboard(state: DashboardState, width: int = 88, height: int = 24) -> str:
    """Render a deterministic dashboard frame for the active terminal size."""
    width = max(36, width)
    height = max(16, height)
    phase = _PHASE_LABELS.get(state.phase, state.phase.upper())
    title_gap = max(1, width - len("TALK TO ME, CLAUDE") - len(phase) - 3)
    lines = [
        f"TALK TO ME, CLAUDE{' ' * title_gap}* {phase}",
        "Local voice link for Claude Code",
        "-" * width,
        "GOAL  Speak naturally while Claude works in the selected project",
        f"MODE  {_MODE_LABELS.get(state.mode, state.mode)}    SOURCE  {_source(state)}",
        f"VOICE {'On' if state.voice_enabled else 'Muted'}    STT  {state.tier}",
        "",
        "SIGNAL",
        _waveform(state, min(width, 72)),
        f"STATUS  {state.notice}",
        "",
        "DIALOGUE",
    ]
    key_lines = (
        ["KEYS  Space Talk   P Project   M Mode   V Voice   R Remote   Q Quit"]
        if width >= 72
        else [
            "KEYS  Space Talk   P Project   M Mode",
            "      V Voice   R Remote   Q Quit",
        ]
    )
    dialogue_room = max(1, height - len(lines) - len(key_lines) - 1)
    dialogue_groups: list[list[str]] = []
    for speaker, message in state.dialogue:
        prefix = f"{speaker.upper():<7} "
        wrapped = textwrap.wrap(message, max(10, width - len(prefix))) or [""]
        dialogue_groups.append(
            [prefix + wrapped[0], *(" " * len(prefix) + part for part in wrapped[1:])]
        )
    dialogue_lines: list[str] = []
    remaining = dialogue_room
    for group in reversed(dialogue_groups):
        if len(group) <= remaining:
            dialogue_lines = group + dialogue_lines
            remaining -= len(group)
        elif not dialogue_lines:
            dialogue_lines = group[:remaining]
            remaining = 0
        if remaining == 0:
            break
    if not dialogue_lines:
        dialogue_lines = ["The current conversation will appear here."]
    lines.extend(dialogue_lines)
    footer = ["-" * width, *key_lines]
    while len(lines) < height - len(footer):
        lines.append("")
    lines.extend(footer)
    return _pad_canvas(lines, width, height)


def render_project_picker(
    state: DashboardState,
    projects: list[str],
    selected: int,
    width: int = 88,
    height: int = 24,
) -> str:
    width = max(36, width)
    height = max(16, height)
    lines = [
        "TALK TO ME, CLAUDE                                      PROJECTS",
        "Choose where remote Claude Code should work",
        "-" * width,
        "GOAL  Select a working directory",
        f"MODE  Project picker    SOURCE  {state.remote or 'No remote configured'}",
        "KEYS  Up/Down Move   Enter Select   E Enter path   Esc Back",
        "",
    ]
    available = max(3, height - len(lines) - 2)
    start = max(0, min(selected - available // 2, max(0, len(projects) - available)))
    for index in range(start, min(len(projects), start + available)):
        marker = ">" if index == selected else " "
        current = "  current" if projects[index] == state.remote_cwd else ""
        lines.append(f"{marker} {projects[index]}{current}")
    if not projects:
        lines.append("No project directories were found.")
    lines.extend(["", _fit(state.notice, width)])
    return _pad_canvas(lines, width, height)


def render_text_prompt(
    state: DashboardState,
    label: str,
    value: str,
    width: int = 88,
    height: int = 24,
) -> str:
    lines = [
        "TALK TO ME, CLAUDE                                      SETTINGS",
        "-" * max(36, width),
        f"GOAL  Set {label.lower()}",
        f"MODE  Text entry    SOURCE  {_source(state)}",
        "KEYS  Enter Save   Esc Cancel   Backspace Delete",
        "",
        label,
        f"> {value}_",
    ]
    return _pad_canvas(lines, max(36, width), max(16, height))


class TerminalScreen:
    def __init__(self, stream: TextIO = sys.stdout) -> None:
        self.stream = stream
        self.color = "NO_COLOR" not in os.environ

    def __enter__(self) -> "TerminalScreen":
        self.stream.write("\x1b[?1049h\x1b[?25l")
        self.stream.flush()
        return self

    def __exit__(self, *_exc_info) -> None:
        self.stream.write("\x1b[?25h\x1b[?1049l")
        self.stream.flush()

    def size(self) -> tuple[int, int]:
        size = shutil.get_terminal_size((88, 24))
        return size.columns, size.lines

    def draw(self, canvas: str, phase: str = "ready") -> None:
        if self.color:
            canvas = canvas.replace("TALK TO ME, CLAUDE", "\x1b[1;36mTALK TO ME, CLAUDE\x1b[0m", 1)
            label = _PHASE_LABELS.get(phase, phase.upper())
            colors = {
                "recording": "31",
                "transcribing": "33",
                "thinking": "33",
                "speaking": "32",
                "error": "31",
            }
            if phase in colors:
                canvas = canvas.replace(label, f"\x1b[1;{colors[phase]}m{label}\x1b[0m", 1)
        self.stream.write("\x1b[H" + canvas + "\x1b[J")
        self.stream.flush()


def _read_key(keys: _RawKeys, timeout: float | None = None) -> str | None:
    key = keys.read_key(timeout)
    if key == "\x1b":
        second = keys.read_key(0.03)
        if second is None:
            return "escape"
        third = keys.read_key(0.03) if second == "[" else None
        key = key + second + (third or "")
    return {
        "\r": "enter",
        "\n": "enter",
        " ": "space",
        "\x08": "backspace",
        "\x7f": "backspace",
        "\xe0H": "up",
        "\xe0P": "down",
        "\x1b[A": "up",
        "\x1b[B": "down",
    }.get(key, key.lower() if key else key)


def discover_remote_projects(remote: str, root: str = "/DEV") -> list[str]:
    inner = (
        f"find -- {shlex.quote(root)} -mindepth 1 -maxdepth 1 -type d "
        "-not -name '.*' -print"
    )
    command = _ssh_base(remote) + [_remote_shell_command(inner)]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        detail = result.stderr.strip() or f"SSH exited {result.returncode}"
        raise TUIError(f"Could not list {root}: {detail}")
    return sorted(
        {line.strip() for line in result.stdout.splitlines() if line.strip()},
        key=str.casefold,
    )


def remote_directory_exists(remote: str, path: str) -> bool:
    inner = f"test -d -- {shlex.quote(path)}"
    command = _ssh_base(remote) + [_remote_shell_command(inner)]
    return subprocess.run(command, capture_output=True, text=True).returncode == 0


def _prompt_text(
    screen: TerminalScreen,
    state: DashboardState,
    label: str,
    initial: str,
) -> str | None:
    value = initial
    with _RawKeys() as keys:
        while True:
            width, height = screen.size()
            screen.draw(render_text_prompt(state, label, value, width, height))
            key = _read_key(keys)
            if key == "escape":
                return None
            if key == "enter":
                return value.strip()
            if key == "backspace":
                value = value[:-1]
            elif isinstance(key, str) and len(key) == 1 and key.isprintable():
                value += key


def _choose_project(
    screen: TerminalScreen,
    state: DashboardState,
    projects: list[str],
) -> str | None:
    selected = projects.index(state.remote_cwd) if state.remote_cwd in projects else 0
    with _RawKeys() as keys:
        while True:
            width, height = screen.size()
            screen.draw(render_project_picker(state, projects, selected, width, height))
            key = _read_key(keys)
            if key in {"escape", "q"}:
                return None
            if key in {"up", "k"} and projects:
                selected = (selected - 1) % len(projects)
            elif key in {"down", "j"} and projects:
                selected = (selected + 1) % len(projects)
            elif key == "enter" and projects:
                return projects[selected]
            elif key == "e":
                return _prompt_text(
                    screen,
                    state,
                    "Remote project path",
                    state.remote_cwd or "/DEV/",
                )


def _dashboard_state() -> DashboardState:
    return DashboardState(
        remote=config.remote(),
        remote_cwd=config.remote_cwd(),
        mode=config.recording_mode(),
        voice_enabled=config.voice_assist_enabled(),
        reduced_motion=os.environ.get("TALKTOMECLAUDE_REDUCED_MOTION") == "1",
    )


def run_dashboard(speak: Callable[[str], None]) -> None:
    """Run the interactive launcher and live voice-session dashboard."""
    if not sys.stdin.isatty() or not sys.stdout.isatty():
        raise TUIError("the dashboard needs an interactive terminal")
    state = _dashboard_state()
    last_draw = 0.0

    with TerminalScreen() as screen:
        def draw(force: bool = False) -> None:
            nonlocal last_draw
            now = time.monotonic()
            if not force and now - last_draw < 0.08:
                return
            width, height = screen.size()
            screen.draw(render_dashboard(state, width, height), state.phase)
            last_draw = now

        def set_phase(phase: str) -> None:
            if state.phase != phase:
                state.phase_started_at = time.monotonic()
            state.phase = phase
            notices = {
                "ready": "Press Space to record; press Space again to send",
                "starting": "Loading the local speech model",
                "recording": "Listening to your microphone",
                "transcribing": "Turning speech into text locally",
                "thinking": "Claude is working in the selected project",
                "speaking": "Playing Claude's reply",
            }
            state.notice = notices.get(phase, state.notice)
            draw(force=True)

        def report_progress() -> None:
            state.frame += 1
            elapsed = max(0, int(time.monotonic() - state.phase_started_at))
            state.notice = f"Claude is working ({elapsed}s)"
            draw()

        def report_status(message: str) -> None:
            if message.startswith("stt tier:"):
                state.tier = message.removeprefix("stt tier:").strip()
            elif "degraded" in message.lower():
                state.notice = message
            draw(force=True)

        def report_dialogue(message: str) -> None:
            if message.startswith("you:"):
                state.add_dialogue("You", message.removeprefix("you:").strip())
            elif message.startswith("claude:"):
                state.add_dialogue("Claude", message.removeprefix("claude:").strip())
            draw(force=True)

        def report_level(level: float) -> None:
            state.add_level(level)
            draw()

        while True:
            draw(force=True)
            try:
                with _RawKeys() as keys:
                    key = None
                    while key is None:
                        key = _read_key(keys, 0.2)
                        if key is None and not state.reduced_motion:
                            state.frame += 1
                            draw(force=True)
            except KeyboardInterrupt:
                return

            if key in {"q", "escape"}:
                return
            if key == "m":
                modes = list(config.RECORDING_MODES)
                state.mode = modes[(modes.index(state.mode) + 1) % len(modes)]
                config.set_recording_mode(state.mode)
                state.notice = f"Recording mode: {_MODE_LABELS[state.mode]}"
                continue
            if key == "v":
                state.voice_enabled = not state.voice_enabled
                config.set_voice_assist(state.voice_enabled)
                state.notice = "Spoken replies enabled" if state.voice_enabled else "Spoken replies muted"
                continue
            if key == "r":
                remote = _prompt_text(screen, state, "SSH target", state.remote or "")
                if remote is not None:
                    config.set_remote(remote or None)
                    state.remote = remote or None
                    state.notice = "Remote target saved" if remote else "Using local Claude Code"
                continue
            if key == "p":
                if state.remote is None:
                    state.notice = "Set an SSH target with R before choosing a remote project"
                    continue
                state.notice = "Loading remote project directories"
                draw(force=True)
                root = str(PurePosixPath(state.remote_cwd).parent) if state.remote_cwd else "/DEV"
                try:
                    projects = discover_remote_projects(state.remote, root)
                    selected = _choose_project(screen, state, projects)
                    if selected is not None:
                        if not remote_directory_exists(state.remote, selected):
                            raise TUIError(f"Remote directory does not exist: {selected}")
                        config.set_remote_cwd(selected)
                        state.remote_cwd = selected
                        state.notice = f"Project selected: {selected}"
                except TUIError as exc:
                    state.phase = "error"
                    state.notice = str(exc)
                continue
            if key not in {"space", "enter"}:
                continue

            try:
                run_listen(
                    mode=state.mode,
                    session_id=None,
                    tmux_pane=None,
                    device="auto",
                    model=None,
                    once=False,
                    echo=report_dialogue,
                    speak=speak,
                    status=report_status,
                    remote=state.remote,
                    remote_cwd=state.remote_cwd,
                    on_level=report_level,
                    on_phase=set_phase,
                    on_progress=report_progress,
                    trigger_key=" ",
                    start_recording=state.mode == "push-toggle",
                )
            except KeyboardInterrupt:
                state.phase = "ready"
                state.notice = "Voice session paused"
            except ListenError as exc:
                state.phase = "error"
                state.notice = str(exc)
