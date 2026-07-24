"""Non-graphical recovery presentation for the production companion."""

from __future__ import annotations

import sys
from collections.abc import Callable
from typing import Protocol

from talktomeclaude.companion.contracts import (
    CompanionIntent,
    CompanionSnapshot,
    IntentKind,
)
from talktomeclaude.companion.viewmodel import CompanionViewModel, to_view_model


class IntentUnavailableError(RuntimeError):
    """A recovery presentation cannot expose the requested direct surface."""


class ProductionController(Protocol):
    @property
    def snapshot(self) -> CompanionSnapshot: ...

    def dispatch(self, intent: CompanionIntent) -> CompanionSnapshot: ...

    def subscribe(
        self, listener: Callable[[CompanionSnapshot], None]
    ) -> Callable[[], None]: ...

    def start_background(self) -> None: ...


class HeadlessController:
    """Focus-free adapter over the same controller used by the Tk shell."""

    _UNAVAILABLE_SURFACES = frozenset(
        {
            IntentKind.OPEN_SETTINGS,
            IntentKind.OPEN_VOICE,
            IntentKind.OPEN_REVIEW,
            IntentKind.OPEN_DIAGNOSTICS,
        }
    )

    def __init__(self, controller: ProductionController) -> None:
        self._controller = controller

    @property
    def snapshot(self) -> CompanionSnapshot:
        return self._controller.snapshot

    def view_model(self) -> CompanionViewModel:
        return to_view_model(self.snapshot)

    def subscribe(
        self, listener: Callable[[CompanionSnapshot], None]
    ) -> Callable[[], None]:
        return self._controller.subscribe(listener)

    def start_background(self) -> None:
        self._controller.start_background()

    def dispatch(self, intent: CompanionIntent) -> CompanionSnapshot:
        if intent.kind in self._UNAVAILABLE_SURFACES:
            raise IntentUnavailableError(
                f"{intent.kind.value} requires the desktop companion"
            )
        return self._controller.dispatch(intent)


InputReader = Callable[[], str | None]
OutputWriter = Callable[[str], object]


def _read_stdin() -> str | None:
    line = sys.stdin.readline()
    return None if line == "" else line


class HeadlessCompanionApplication:
    """Host production state and workflow intents without a GUI or hotkey."""

    _COMMANDS = {
        "status": IntentKind.STATUS,
        "start": IntentKind.START_RECORDING,
        "finish": IntentKind.FINISH_RECORDING,
        "cancel": IntentKind.CANCEL,
        "mute": IntentKind.TOGGLE_OUTPUT_MUTE,
        "quit": IntentKind.QUIT,
        "exit": IntentKind.QUIT,
    }
    HELP = "Commands: status, start, finish, cancel, mute, quit"

    def __init__(
        self,
        controller: HeadlessController,
        *,
        read: InputReader = _read_stdin,
        write: OutputWriter = print,
    ) -> None:
        self._controller = controller
        self._read = read
        self._write = write
        self._last_snapshot: CompanionSnapshot | None = None

    def _render(self, snapshot: CompanionSnapshot, *, force: bool = False) -> None:
        if not force and snapshot == self._last_snapshot:
            return
        self._last_snapshot = snapshot
        view = to_view_model(snapshot)
        detail = f" — {view.detail}" if view.detail else ""
        muted = " [MUTED]" if snapshot.output_muted else ""
        self._write(f"{view.cue}: {view.status}{muted}{detail}")

    def _command(self, value: str) -> bool:
        command = value.strip().casefold()
        if not command:
            return True
        if command in {"help", "?"}:
            self._write(self.HELP)
            return True
        kind = self._COMMANDS.get(command)
        if kind is None:
            self._write("ERROR: Unknown command")
            return True
        try:
            snapshot = self._controller.dispatch(CompanionIntent(kind))
        except Exception:
            self._write("ERROR: Action unavailable")
            return True
        self._render(snapshot, force=kind is IntentKind.STATUS)
        return kind is not IntentKind.QUIT

    def run(self) -> int:
        unsubscribe = self._controller.subscribe(self._render)
        try:
            self._controller.start_background()
            self._render(self._controller.snapshot)
            while True:
                try:
                    value = self._read()
                except (EOFError, KeyboardInterrupt):
                    break
                if value is None or not self._command(value):
                    break
            return 0
        finally:
            unsubscribe()
            try:
                self._controller.dispatch(CompanionIntent(IntentKind.QUIT))
            except Exception:
                pass


def run_headless(
    write: OutputWriter = print,
    controller: ProductionController | HeadlessController | None = None,
    *,
    read: InputReader = _read_stdin,
) -> int:
    """Run the production controller through the non-graphical presentation."""

    if controller is None:
        # Keep production graph construction lazy: importing this module never
        # opens audio, initializes speech, changes hooks, or imports Tk.
        from talktomeclaude.companion.app import build_headless_controller

        active = HeadlessController(build_headless_controller())
    elif isinstance(controller, HeadlessController):
        active = controller
    else:
        active = HeadlessController(controller)
    return HeadlessCompanionApplication(active, read=read, write=write).run()


__all__ = [
    "HeadlessCompanionApplication",
    "HeadlessController",
    "IntentUnavailableError",
    "ProductionController",
    "run_headless",
]
