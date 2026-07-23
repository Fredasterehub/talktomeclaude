"""First-run guided setup: keyboard-driven panes with a defaults fast path.

One decision per pane, every pane skippable, every choice persisted the
moment it is made (Ctrl-C loses nothing). The onboarding version and the
completion timestamp are written only at Finish — or through the explicit
"use recommended defaults" fast path, which is Finish taken early.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Callable

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.screen import Screen
from textual.widgets import Input, OptionList, Static

from talktomeclaude import config

CURRENT_ONBOARDING_VERSION: int = 1

_STEP_ORDER = (
    "welcome",
    "hardware",
    "claude",
    "voice",
    "spoken",
    "recording",
    "wake",
    "permissions",
    "finish",
)

_HINT = "Enter Select   ·   S Skip   ·   B Back   ·   Esc Use defaults and finish"


class OnboardingScreen(Screen[bool]):
    """The guided first-run sequence, keyboard-driven and skippable."""

    BINDINGS = [
        Binding("s", "skip", "Skip", show=False),
        Binding("b", "back", "Back", show=False),
        Binding("escape", "accept_defaults", "Use defaults", show=False, priority=True),
    ]

    def __init__(self) -> None:
        super().__init__()
        self._step = "welcome"
        self._history: list[str] = []
        self._clone_feasible: bool | None = None

    # ── pane rendering ───────────────────────────────────────────────────────
    def compose(self) -> ComposeResult:
        yield from self._widgets_for(self._step)

    def _widgets_for(self, step: str):
        if step == "welcome":
            yield Static("Welcome to Talk To Me, Claude")
            yield Static(
                "Everything runs locally: your voice, the transcription, and "
                "the spoken replies never leave your machines."
            )
            yield OptionList(
                "Use recommended defaults and finish now",
                "Customize step by step",
                id="ob-welcome",
            )
        elif step == "hardware":
            yield Static("Your hardware")
            yield Static(self._hardware_summary())
            yield OptionList("Continue", id="ob-hardware")
        elif step == "claude":
            yield Static("Where does Claude Code run?")
            yield OptionList(
                "On this machine (default)",
                "On a remote server over SSH",
                id="ob-claude",
            )
        elif step == "remote-target":
            yield Static("SSH target (user@host)")
            yield Input(id="ob-remote-target", placeholder="user@host")
        elif step == "remote-cwd":
            yield Static("Remote project directory (blank for the home directory)")
            yield Input(id="ob-remote-cwd", placeholder="/DEV/project")
        elif step == "voice":
            yield Static("Pick a first voice")
            options = ["Auto (recommended)"]
            from talktomeclaude.tts import BUNDLED_VOICES

            options += [voice.name for voice in BUNDLED_VOICES]
            if self._clone_ok():
                options.append("Clone your own voice…")
            yield OptionList(*options, id="ob-voice")
        elif step == "spoken":
            yield Static("Spoken replies")
            yield OptionList(
                "Speak Claude's replies aloud (default)",
                "Stay silent (text only)",
                id="ob-spoken",
            )
        elif step == "recording":
            yield Static("Recording mode")
            yield OptionList(*config.RECORDING_MODES, id="ob-recording")
        elif step == "wake":
            yield Static("Voice commands and wake word (optional — off by default)")
            yield OptionList(
                "Leave the wake word off (default)",
                "Enable the wake word for hands-free listening",
                id="ob-wake",
            )
        elif step == "wake-phrase":
            yield Static("Wake phrase")
            yield Input(
                id="ob-wake-phrase",
                value=config.wake_phrase(),
                placeholder=config.DEFAULT_WAKE_PHRASE,
            )
        elif step == "permissions":
            yield Static("Claude permission posture (off is the safe default)")
            yield OptionList(*config.CLAUDE_PERMISSIONS, id="ob-permissions")
        else:  # finish
            yield Static("All set")
            yield Static(
                "Every choice is already saved. Re-run this any time with "
                "`talktomeclaude setup`."
            )
            yield OptionList("Finish", id="ob-finish")
        yield Static(_HINT)

    def _hardware_summary(self) -> str:
        try:
            from talktomeclaude import advisor

            recommendation = advisor.recommend()
            self._clone_feasible = recommendation.clone_feasible
            cloning = (
                "voice cloning: feasible on this machine"
                if recommendation.clone_feasible
                else "voice cloning: not recommended on this machine"
            )
            return f"{cloning}. Run `talktomeclaude doctor` for the full report."
        except Exception:
            return "Hardware detection unavailable; run `talktomeclaude doctor` later."

    def _clone_ok(self) -> bool:
        if self._clone_feasible is None:
            try:
                from talktomeclaude import advisor

                self._clone_feasible = advisor.recommend().clone_feasible
            except Exception:
                self._clone_feasible = False
        return self._clone_feasible

    def on_mount(self) -> None:
        self._focus_step()

    def _focus_step(self) -> None:
        lists = self.query(OptionList)
        if lists:
            pane = lists.first(OptionList)
            pane.highlighted = self._default_highlight(pane)
            pane.focus()
            return
        inputs = self.query(Input)
        if inputs:
            inputs.first(Input).focus()

    def _default_highlight(self, pane: OptionList) -> int:
        if pane.id == "ob-recording":
            return config.RECORDING_MODES.index(config.DEFAULT_RECORDING_MODE)
        if pane.id == "ob-permissions":
            return config.CLAUDE_PERMISSIONS.index(config.claude_permissions())
        return 0

    # ── navigation ───────────────────────────────────────────────────────────
    def _goto(self, step: str) -> None:
        self._history.append(self._step)
        self._step = step
        self.remove_children()
        self.mount_all(list(self._widgets_for(step)))
        self.call_after_refresh(self._focus_step)

    def _next(self) -> None:
        order = list(_STEP_ORDER)
        index = order.index(self._step) if self._step in order else 0
        self._goto(order[min(index + 1, len(order) - 1)])

    def action_skip(self) -> None:
        if self._step == "finish":
            self._finish()
            return
        if self._step in ("remote-target", "remote-cwd"):
            self._goto("voice")
            return
        if self._step == "wake-phrase":
            self._goto("permissions")
            return
        self._next()

    def action_back(self) -> None:
        if not self._history:
            return
        previous = self._history.pop()
        self._step = previous
        self.remove_children()
        self.mount_all(list(self._widgets_for(previous)))
        self.call_after_refresh(self._focus_step)

    def action_accept_defaults(self) -> None:
        """The defaults fast path: recommended defaults, then Finish."""
        config.set_recording_mode(config.DEFAULT_RECORDING_MODE)
        config.set_claude_permissions("off")
        self._finish()

    def _finish(self) -> None:
        config.set_onboarding_version(CURRENT_ONBOARDING_VERSION)
        config.set_onboarding_completed_at(
            datetime.now(timezone.utc).isoformat(timespec="seconds")
        )
        self.dismiss(True)

    # ── choices (each persisted the moment it is made) ───────────────────────
    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        pane = event.option_list.id
        index = event.option_index
        if pane == "ob-welcome":
            if index == 0:
                self.action_accept_defaults()
            else:
                self._next()
        elif pane == "ob-hardware":
            self._next()
        elif pane == "ob-claude":
            if index == 0:
                config.set_remote(None)
                self._goto("voice")
            else:
                self._goto("remote-target")
        elif pane == "ob-voice":
            label = str(event.option.prompt)
            if label == "Auto (recommended)":
                config.set_default_voice(None)
                self._next()
            elif label == "Clone your own voice…":
                from talktomeclaude.clone_ui import CloneScreen

                self.app.push_screen(CloneScreen(), lambda _created: None)
                self._next()
            else:
                config.set_default_voice(label)
                self._next()
        elif pane == "ob-spoken":
            config.set_voice_assist(index == 0)
            self._next()
        elif pane == "ob-recording":
            config.set_recording_mode(config.RECORDING_MODES[index])
            self._next()
        elif pane == "ob-wake":
            config.set_wake_word(index == 1)
            if index == 1:
                self._goto("wake-phrase")
            else:
                self._next()
        elif pane == "ob-permissions":
            config.set_claude_permissions(config.CLAUDE_PERMISSIONS[index])
            self._next()
        elif pane == "ob-finish":
            self._finish()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        value = event.value.strip()
        if event.input.id == "ob-remote-target":
            config.set_remote(value or None)
            self._goto("remote-cwd" if value else "voice")
        elif event.input.id == "ob-remote-cwd":
            config.set_remote_cwd(value or None)
            self._goto("voice")
        elif event.input.id == "ob-wake-phrase":
            config.set_wake_phrase(value or config.DEFAULT_WAKE_PHRASE)
            self._goto("permissions")


class _OnboardingApp(App[None]):
    def __init__(self, speak: Callable[[str], None] | None) -> None:
        super().__init__()
        self._speak = speak

    def on_mount(self) -> None:
        self.push_screen(OnboardingScreen(), self._on_onboarding_dismissed)

    def _on_onboarding_dismissed(self, _completed: bool | None) -> None:
        self.exit()


def run_onboarding(speak: Callable[[str], None] | None = None) -> None:
    """Run onboarding until the setup screen is dismissed."""
    _OnboardingApp(speak).run()
