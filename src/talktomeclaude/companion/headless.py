"""Headless recovery presentation for the staged companion command."""

from __future__ import annotations

from collections.abc import Callable

from talktomeclaude.companion.contracts import (
    CompanionIntent,
    CompanionSnapshot,
    IntentKind,
)
from talktomeclaude.companion.viewmodel import CompanionViewModel, to_view_model
from talktomeclaude.core import EventKind, RuntimeCoordinator, RuntimeEvent


class IntentUnavailableError(RuntimeError):
    """Raised when the staged skeleton receives an unwired workflow intent."""


class HeadlessController:
    """Minimal controller that can later be bound to the companion runtime.

    User workflow intents pass through the authoritative runtime coordinator.
    This slice advances lifecycle state only; capture and delivery side effects
    remain unwired and are never simulated.
    """

    _EVENTS = {
        IntentKind.START_RECORDING: EventKind.START_RECORDING,
        IntentKind.FINISH_RECORDING: EventKind.FINISH_RECORDING,
        IntentKind.CANCEL: EventKind.CANCEL,
        IntentKind.QUIT: EventKind.STOP_REQUESTED,
    }

    def __init__(
        self,
        runtime: RuntimeCoordinator | None = None,
        *,
        detail: str = "",
    ) -> None:
        self._runtime = runtime or RuntimeCoordinator()
        self._detail = detail

    @property
    def snapshot(self) -> CompanionSnapshot:
        return CompanionSnapshot(self._runtime.state, self._detail)

    def view_model(self) -> CompanionViewModel:
        return to_view_model(self.snapshot)

    def dispatch(self, intent: CompanionIntent) -> CompanionSnapshot:
        if intent.kind is IntentKind.STATUS:
            return self.snapshot

        event_kind = self._EVENTS.get(intent.kind)
        if event_kind is None:
            raise IntentUnavailableError(
                f"{intent.kind.value} is unavailable in the headless presentation"
            )

        result = self._runtime.dispatch(RuntimeEvent(event_kind))
        if not result.accepted:
            raise IntentUnavailableError(
                f"{intent.kind.value} is unavailable while {result.current.phase.value}"
            )
        return self.snapshot


def run_headless(
    write: Callable[[str], object] = print,
    controller: HeadlessController | None = None,
) -> int:
    """Report the recovery controller state without starting an interactive UI."""

    active = controller or HeadlessController()
    view = active.view_model()
    write(f"{view.cue}: {view.status}")
    return 0
