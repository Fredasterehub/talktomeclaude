"""Supported foreground-terminal capabilities.

Eligibility is based on process and window-class evidence together.  Window
titles, UI Automation contents, tabs, panes, and cursor state are intentionally
outside this boundary.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class TerminalCapability:
    kind: str
    process_names: frozenset[str]
    window_classes: frozenset[str]

    def matches(self, process_name: str, window_class: str) -> bool:
        return (
            process_name.casefold() in self.process_names
            and window_class.casefold() in self.window_classes
        )


SUPPORTED_TERMINALS = (
    TerminalCapability(
        "windows_terminal",
        frozenset({"windowsterminal.exe"}),
        frozenset({"cascadia_hosting_window_class"}),
    ),
    TerminalCapability(
        "console_host",
        frozenset({"conhost.exe", "openconsole.exe"}),
        frozenset({"consolewindowclass"}),
    ),
    TerminalCapability(
        "wezterm",
        frozenset({"wezterm-gui.exe"}),
        frozenset({"org.wezfurlong.wezterm"}),
    ),
    TerminalCapability(
        "alacritty",
        frozenset({"alacritty.exe"}),
        frozenset({"alacritty"}),
    ),
    TerminalCapability(
        "mintty",
        frozenset({"mintty.exe"}),
        frozenset({"mintty"}),
    ),
)


def resolve_terminal_capability(
    process_name: str, window_class: str
) -> TerminalCapability | None:
    """Return a supported process/class pair, never a title-based guess."""

    for capability in SUPPORTED_TERMINALS:
        if capability.matches(process_name, window_class):
            return capability
    return None
