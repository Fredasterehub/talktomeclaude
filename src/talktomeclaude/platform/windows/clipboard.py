"""Bounded Win32 Unicode clipboard transactions."""

from __future__ import annotations

import ctypes
import os
import time
from ctypes import wintypes
from dataclasses import dataclass, field
from enum import Enum
from typing import Protocol

from talktomeclaude.platform.contracts import RestoreStatus


class ClipboardCode(str, Enum):
    OK = "ok"
    OPEN_TIMEOUT = "open_timeout"
    UNSUPPORTED_FORMAT = "unsupported_format"
    READ_FAILED = "read_failed"
    SET_FAILED = "set_failed"


@dataclass(frozen=True, slots=True)
class ClipboardSnapshot:
    text: str | None = field(repr=False)
    sequence: int


@dataclass(frozen=True, slots=True)
class ClipboardOperation:
    code: ClipboardCode

    @property
    def succeeded(self) -> bool:
        return self.code is ClipboardCode.OK


class ClipboardFacade(Protocol):
    def open(self) -> bool: ...
    def close(self) -> None: ...
    def read_unicode(self) -> str | None: ...
    def is_unicode_only_or_empty(self) -> bool: ...
    def write_unicode(self, text: str) -> bool: ...
    def clear(self) -> bool: ...
    def sequence_number(self) -> int: ...


class CtypesClipboardFacade:
    CF_TEXT = 1
    CF_OEMTEXT = 7
    CF_UNICODETEXT = 13
    CF_LOCALE = 16
    GMEM_MOVEABLE = 0x0002

    def __init__(self, owner_hwnd: int = 0) -> None:
        if os.name != "nt":
            raise OSError("Windows clipboard APIs are available only on Windows")
        self._owner_hwnd = owner_hwnd
        self._temporary_owner = 0
        self._user32 = ctypes.WinDLL("user32", use_last_error=True)
        self._kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        self._configure_signatures()

    def _configure_signatures(self) -> None:
        self._user32.OpenClipboard.argtypes = [wintypes.HWND]
        self._user32.OpenClipboard.restype = wintypes.BOOL
        self._user32.CloseClipboard.restype = wintypes.BOOL
        self._user32.IsClipboardFormatAvailable.argtypes = [wintypes.UINT]
        self._user32.IsClipboardFormatAvailable.restype = wintypes.BOOL
        self._user32.GetClipboardData.argtypes = [wintypes.UINT]
        self._user32.GetClipboardData.restype = wintypes.HANDLE
        self._user32.EmptyClipboard.restype = wintypes.BOOL
        self._user32.SetClipboardData.argtypes = [wintypes.UINT, wintypes.HANDLE]
        self._user32.SetClipboardData.restype = wintypes.HANDLE
        self._user32.GetClipboardSequenceNumber.restype = wintypes.DWORD
        self._user32.EnumClipboardFormats.argtypes = [wintypes.UINT]
        self._user32.EnumClipboardFormats.restype = wintypes.UINT
        self._user32.CreateWindowExW.argtypes = [
            wintypes.DWORD,
            wintypes.LPCWSTR,
            wintypes.LPCWSTR,
            wintypes.DWORD,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            wintypes.HWND,
            wintypes.HMENU,
            wintypes.HINSTANCE,
            wintypes.LPVOID,
        ]
        self._user32.CreateWindowExW.restype = wintypes.HWND
        self._user32.DestroyWindow.argtypes = [wintypes.HWND]
        self._user32.DestroyWindow.restype = wintypes.BOOL
        self._kernel32.GlobalAlloc.argtypes = [wintypes.UINT, ctypes.c_size_t]
        self._kernel32.GlobalAlloc.restype = wintypes.HGLOBAL
        self._kernel32.GlobalLock.argtypes = [wintypes.HGLOBAL]
        self._kernel32.GlobalLock.restype = wintypes.LPVOID
        self._kernel32.GlobalUnlock.argtypes = [wintypes.HGLOBAL]
        self._kernel32.GlobalUnlock.restype = wintypes.BOOL
        self._kernel32.GlobalFree.argtypes = [wintypes.HGLOBAL]
        self._kernel32.GlobalFree.restype = wintypes.HGLOBAL

    def open(self) -> bool:
        owner = self._owner_hwnd
        if not owner:
            # SetClipboardData needs a real owner after EmptyClipboard.  A
            # message-only built-in STATIC window is private to this operation
            # and can never activate or appear in the taskbar.
            hwnd_message = ctypes.c_void_p(-3).value
            self._temporary_owner = int(
                self._user32.CreateWindowExW(
                    0,
                    "STATIC",
                    "",
                    0,
                    0,
                    0,
                    0,
                    0,
                    hwnd_message,
                    None,
                    None,
                    None,
                )
                or 0
            )
            owner = self._temporary_owner
            if not owner:
                return False
        if self._user32.OpenClipboard(owner):
            return True
        if self._temporary_owner:
            self._user32.DestroyWindow(self._temporary_owner)
            self._temporary_owner = 0
        return False

    def close(self) -> None:
        self._user32.CloseClipboard()
        if self._temporary_owner:
            self._user32.DestroyWindow(self._temporary_owner)
            self._temporary_owner = 0

    def read_unicode(self) -> str | None:
        if not self._user32.IsClipboardFormatAvailable(self.CF_UNICODETEXT):
            return None
        handle = self._user32.GetClipboardData(self.CF_UNICODETEXT)
        if not handle:
            raise OSError(ctypes.get_last_error(), "GetClipboardData failed")
        pointer = self._kernel32.GlobalLock(handle)
        if not pointer:
            raise OSError(ctypes.get_last_error(), "GlobalLock failed")
        try:
            return ctypes.wstring_at(pointer)
        finally:
            self._kernel32.GlobalUnlock(handle)

    def is_unicode_only_or_empty(self) -> bool:
        """Reject rich/binary/custom formats that this adapter cannot restore."""

        formats: set[int] = set()
        current = 0
        ctypes.set_last_error(0)
        while True:
            current = int(self._user32.EnumClipboardFormats(current))
            if not current:
                break
            formats.add(current)
        if ctypes.get_last_error():
            return False
        if not formats:
            return True
        allowed = {self.CF_TEXT, self.CF_OEMTEXT, self.CF_UNICODETEXT, self.CF_LOCALE}
        return formats <= allowed and bool(
            self._user32.IsClipboardFormatAvailable(self.CF_UNICODETEXT)
        )

    def write_unicode(self, text: str) -> bool:
        encoded = (text + "\0").encode("utf-16-le")
        handle = self._kernel32.GlobalAlloc(self.GMEM_MOVEABLE, len(encoded))
        if not handle:
            return False
        transferred = False
        try:
            pointer = self._kernel32.GlobalLock(handle)
            if not pointer:
                return False
            try:
                ctypes.memmove(pointer, encoded, len(encoded))
            finally:
                self._kernel32.GlobalUnlock(handle)
            if not self._user32.EmptyClipboard():
                return False
            if not self._user32.SetClipboardData(self.CF_UNICODETEXT, handle):
                return False
            transferred = True
            return True
        finally:
            if not transferred:
                self._kernel32.GlobalFree(handle)

    def clear(self) -> bool:
        return bool(self._user32.EmptyClipboard())

    def sequence_number(self) -> int:
        return int(self._user32.GetClipboardSequenceNumber())


class ClipboardTransaction:
    """Snapshot, replace, and restore only while our replacement is current."""

    def __init__(
        self,
        facade: ClipboardFacade | None = None,
        *,
        open_timeout: float = 0.25,
        retry_interval: float = 0.01,
        sleep=time.sleep,
        monotonic=time.monotonic,
    ) -> None:
        if open_timeout < 0 or retry_interval < 0:
            raise ValueError("clipboard timing values must be non-negative")
        self._facade = facade or CtypesClipboardFacade()
        self._open_timeout = open_timeout
        self._retry_interval = retry_interval
        self._sleep = sleep
        self._monotonic = monotonic
        self._snapshot: ClipboardSnapshot | None = None
        self._owned_sequence: int | None = None

    def _open_bounded(self) -> bool:
        deadline = self._monotonic() + self._open_timeout
        while True:
            if self._facade.open():
                return True
            if self._monotonic() >= deadline:
                return False
            self._sleep(self._retry_interval)

    def snapshot(self) -> ClipboardOperation:
        if not self._open_bounded():
            return ClipboardOperation(ClipboardCode.OPEN_TIMEOUT)
        try:
            try:
                if not self._facade.is_unicode_only_or_empty():
                    return ClipboardOperation(ClipboardCode.UNSUPPORTED_FORMAT)
                text = self._facade.read_unicode()
            except (OSError, RuntimeError):
                return ClipboardOperation(ClipboardCode.READ_FAILED)
            self._snapshot = ClipboardSnapshot(text, self._facade.sequence_number())
            return ClipboardOperation(ClipboardCode.OK)
        finally:
            self._facade.close()

    def set_text(self, text: str) -> ClipboardOperation:
        if self._snapshot is None:
            raise RuntimeError("clipboard must be snapshotted before replacement")
        if not self._open_bounded():
            return ClipboardOperation(ClipboardCode.OPEN_TIMEOUT)
        try:
            try:
                wrote = self._facade.write_unicode(text)
            except (OSError, RuntimeError):
                wrote = False
            # A failed SetClipboardData (including an exception) may still
            # follow a successful EmptyClipboard. Track the resulting
            # generation so the caller restores instead of assuming no change.
            self._owned_sequence = self._facade.sequence_number()
            return ClipboardOperation(
                ClipboardCode.OK if wrote else ClipboardCode.SET_FAILED
            )
        finally:
            self._facade.close()

    def restore(self) -> RestoreStatus:
        if self._snapshot is None or self._owned_sequence is None:
            return RestoreStatus.NOT_NEEDED
        if self._facade.sequence_number() != self._owned_sequence:
            return RestoreStatus.CONFLICT
        if not self._open_bounded():
            return RestoreStatus.OPEN_TIMEOUT
        try:
            # Recheck after obtaining the clipboard lock so another owner cannot
            # win between the conflict check and our restore.
            if self._facade.sequence_number() != self._owned_sequence:
                return RestoreStatus.CONFLICT
            try:
                if self._snapshot.text is None:
                    restored = self._facade.clear()
                else:
                    restored = self._facade.write_unicode(self._snapshot.text)
                if not restored:
                    return RestoreStatus.FAILED
                return RestoreStatus.RESTORED
            except (OSError, RuntimeError):
                return RestoreStatus.FAILED
        finally:
            self._facade.close()
