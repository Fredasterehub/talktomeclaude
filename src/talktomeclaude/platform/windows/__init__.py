"""Native Windows focus, clipboard, injection, and hotkey adapters."""

from .injector import TextInjector
from .target import TargetEvidence, WindowsTargetResolver

__all__ = ["TargetEvidence", "TextInjector", "WindowsTargetResolver"]
