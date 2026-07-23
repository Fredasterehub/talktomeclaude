"""Ephemeral foreground-target evidence and validation."""

from __future__ import annotations

import ctypes
import ntpath
import os
from ctypes import wintypes
from dataclasses import dataclass, field
from enum import Enum
from typing import Protocol

from .capabilities import resolve_terminal_capability


class TargetCode(str, Enum):
    VALID = "valid"
    NO_FOREGROUND_WINDOW = "no_foreground_window"
    PID_UNAVAILABLE = "pid_unavailable"
    PROCESS_UNAVAILABLE = "process_unavailable"
    CLASS_UNAVAILABLE = "class_unavailable"
    UNSUPPORTED = "unsupported"
    WINDOW_GONE = "window_gone"
    FOREGROUND_CHANGED = "foreground_changed"
    EVIDENCE_CHANGED = "evidence_changed"


@dataclass(frozen=True, slots=True)
class TargetEvidence:
    """One transaction's native evidence; never persist or emit in diagnostics."""

    hwnd: int = field(repr=False)
    pid: int = field(repr=False)
    process_name: str = field(repr=False)
    window_class: str = field(repr=False)
    terminal_kind: str


@dataclass(frozen=True, slots=True)
class TargetResolution:
    evidence: TargetEvidence | None
    code: TargetCode


@dataclass(frozen=True, slots=True)
class TargetValidation:
    valid: bool
    code: TargetCode


class TargetFacade(Protocol):
    def get_foreground_window(self) -> int: ...
    def get_window_process_id(self, hwnd: int) -> int: ...
    def get_process_name(self, pid: int) -> str: ...
    def get_window_class(self, hwnd: int) -> str: ...
    def is_window(self, hwnd: int) -> bool: ...


class CtypesTargetFacade:
    """Small checked facade over the Win32 APIs used for target evidence."""

    PROCESS_QUERY_LIMITED_INFORMATION = 0x1000

    def __init__(self) -> None:
        if os.name != "nt":
            raise OSError("Windows target APIs are available only on Windows")
        self._user32 = ctypes.WinDLL("user32", use_last_error=True)
        self._kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        self._configure_signatures()

    def _configure_signatures(self) -> None:
        self._user32.GetForegroundWindow.restype = wintypes.HWND
        self._user32.GetWindowThreadProcessId.argtypes = [
            wintypes.HWND,
            ctypes.POINTER(wintypes.DWORD),
        ]
        self._user32.GetWindowThreadProcessId.restype = wintypes.DWORD
        self._user32.GetClassNameW.argtypes = [
            wintypes.HWND,
            wintypes.LPWSTR,
            ctypes.c_int,
        ]
        self._user32.GetClassNameW.restype = ctypes.c_int
        self._user32.IsWindow.argtypes = [wintypes.HWND]
        self._user32.IsWindow.restype = wintypes.BOOL
        self._kernel32.OpenProcess.argtypes = [
            wintypes.DWORD,
            wintypes.BOOL,
            wintypes.DWORD,
        ]
        self._kernel32.OpenProcess.restype = wintypes.HANDLE
        self._kernel32.QueryFullProcessImageNameW.argtypes = [
            wintypes.HANDLE,
            wintypes.DWORD,
            wintypes.LPWSTR,
            ctypes.POINTER(wintypes.DWORD),
        ]
        self._kernel32.QueryFullProcessImageNameW.restype = wintypes.BOOL
        self._kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
        self._kernel32.CloseHandle.restype = wintypes.BOOL

    def get_foreground_window(self) -> int:
        return int(self._user32.GetForegroundWindow() or 0)

    def get_window_process_id(self, hwnd: int) -> int:
        pid = wintypes.DWORD()
        if not self._user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid)):
            return 0
        return int(pid.value)

    def get_process_name(self, pid: int) -> str:
        handle = self._kernel32.OpenProcess(
            self.PROCESS_QUERY_LIMITED_INFORMATION, False, pid
        )
        if not handle:
            return ""
        try:
            capacity = 32768
            size = wintypes.DWORD(capacity)
            buffer = ctypes.create_unicode_buffer(capacity)
            if not self._kernel32.QueryFullProcessImageNameW(
                handle, 0, buffer, ctypes.byref(size)
            ):
                return ""
            return ntpath.basename(buffer.value)
        finally:
            self._kernel32.CloseHandle(handle)

    def get_window_class(self, hwnd: int) -> str:
        buffer = ctypes.create_unicode_buffer(256)
        if not self._user32.GetClassNameW(hwnd, buffer, len(buffer)):
            return ""
        return buffer.value

    def is_window(self, hwnd: int) -> bool:
        return bool(self._user32.IsWindow(hwnd))


class WindowsTargetResolver:
    def __init__(self, facade: TargetFacade | None = None) -> None:
        self._facade = facade or CtypesTargetFacade()

    def snapshot(self) -> TargetResolution:
        try:
            hwnd = self._facade.get_foreground_window()
        except (OSError, RuntimeError):
            hwnd = 0
        if not hwnd:
            return TargetResolution(None, TargetCode.NO_FOREGROUND_WINDOW)
        try:
            pid = self._facade.get_window_process_id(hwnd)
        except (OSError, RuntimeError):
            pid = 0
        if not pid:
            return TargetResolution(None, TargetCode.PID_UNAVAILABLE)
        try:
            process_name = self._facade.get_process_name(pid)
        except (OSError, RuntimeError):
            process_name = ""
        if not process_name:
            return TargetResolution(None, TargetCode.PROCESS_UNAVAILABLE)
        try:
            window_class = self._facade.get_window_class(hwnd)
        except (OSError, RuntimeError):
            window_class = ""
        if not window_class:
            return TargetResolution(None, TargetCode.CLASS_UNAVAILABLE)
        capability = resolve_terminal_capability(process_name, window_class)
        if capability is None:
            return TargetResolution(None, TargetCode.UNSUPPORTED)
        return TargetResolution(
            TargetEvidence(
                hwnd=hwnd,
                pid=pid,
                process_name=process_name,
                window_class=window_class,
                terminal_kind=capability.kind,
            ),
            TargetCode.VALID,
        )

    def validate(self, evidence: TargetEvidence) -> TargetValidation:
        try:
            is_window = self._facade.is_window(evidence.hwnd)
        except (OSError, RuntimeError):
            is_window = False
        if not is_window:
            return TargetValidation(False, TargetCode.WINDOW_GONE)
        try:
            foreground = self._facade.get_foreground_window()
        except (OSError, RuntimeError):
            foreground = 0
        if foreground != evidence.hwnd:
            return TargetValidation(False, TargetCode.FOREGROUND_CHANGED)
        try:
            pid = self._facade.get_window_process_id(evidence.hwnd)
            process_name = self._facade.get_process_name(pid) if pid else ""
            window_class = self._facade.get_window_class(evidence.hwnd)
        except (OSError, RuntimeError):
            return TargetValidation(False, TargetCode.EVIDENCE_CHANGED)
        if (
            pid != evidence.pid
            or process_name.casefold() != evidence.process_name.casefold()
            or window_class.casefold() != evidence.window_class.casefold()
            or resolve_terminal_capability(process_name, window_class) is None
        ):
            return TargetValidation(False, TargetCode.EVIDENCE_CHANGED)
        return TargetValidation(True, TargetCode.VALID)
