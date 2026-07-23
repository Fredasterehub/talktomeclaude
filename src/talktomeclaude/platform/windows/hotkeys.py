"""Global Win32 hotkey registration with no-repeat and deterministic cleanup."""

from __future__ import annotations

import ctypes
import os
import threading
from ctypes import wintypes
from typing import Protocol


MOD_ALT = 0x0001
MOD_CONTROL = 0x0002
MOD_SHIFT = 0x0004
MOD_WIN = 0x0008
MOD_NOREPEAT = 0x4000
WM_HOTKEY = 0x0312


class HotkeyFacade(Protocol):
    def register_hotkey(self, hwnd: int, hotkey_id: int, modifiers: int, vk: int) -> bool: ...
    def unregister_hotkey(self, hwnd: int, hotkey_id: int) -> bool: ...


class HotkeyIntentQueue(Protocol):
    def put_nowait(self, hotkey_id: int) -> None: ...


class CtypesHotkeyFacade:
    def __init__(self) -> None:
        if os.name != "nt":
            raise OSError("Windows hotkey APIs are available only on Windows")
        self._user32 = ctypes.WinDLL("user32", use_last_error=True)
        self._user32.RegisterHotKey.argtypes = [
            wintypes.HWND,
            ctypes.c_int,
            wintypes.UINT,
            wintypes.UINT,
        ]
        self._user32.RegisterHotKey.restype = wintypes.BOOL
        self._user32.UnregisterHotKey.argtypes = [wintypes.HWND, ctypes.c_int]
        self._user32.UnregisterHotKey.restype = wintypes.BOOL

    def register_hotkey(self, hwnd: int, hotkey_id: int, modifiers: int, vk: int) -> bool:
        return bool(self._user32.RegisterHotKey(hwnd, hotkey_id, modifiers, vk))

    def unregister_hotkey(self, hwnd: int, hotkey_id: int) -> bool:
        return bool(self._user32.UnregisterHotKey(hwnd, hotkey_id))


class GlobalHotkeyAdapter:
    """Registration plus a deliberately tiny shell-owned dispatch boundary.

    The shell owns and pumps ``WM_HOTKEY`` on the constructing thread, then
    passes messages to :meth:`dispatch_message`.  This adapter only queues the
    integer hotkey intent; runtime work cannot execute in the Win32 message
    callback.
    """

    def __init__(
        self,
        facade: HotkeyFacade | None = None,
        *,
        hwnd: int = 0,
        intent_queue: HotkeyIntentQueue | None = None,
    ) -> None:
        self._facade = facade or CtypesHotkeyFacade()
        self._hwnd = hwnd
        self._intent_queue = intent_queue
        self._owner_thread = threading.get_ident()
        self._registered: set[int] = set()

    def _require_owner_thread(self) -> None:
        if threading.get_ident() != self._owner_thread:
            raise RuntimeError("hotkey operations must run on the owning message thread")

    def register(self, hotkey_id: int, modifiers: int, vk: int) -> None:
        self._require_owner_thread()
        if hotkey_id in self._registered:
            raise ValueError(f"hotkey id {hotkey_id} is already registered")
        effective_modifiers = modifiers | MOD_NOREPEAT
        if not self._facade.register_hotkey(
            self._hwnd, hotkey_id, effective_modifiers, vk
        ):
            raise OSError(ctypes.get_last_error(), "RegisterHotKey failed")
        self._registered.add(hotkey_id)

    def unregister(self, hotkey_id: int) -> None:
        self._require_owner_thread()
        if hotkey_id not in self._registered:
            return
        if not self._facade.unregister_hotkey(self._hwnd, hotkey_id):
            raise OSError(ctypes.get_last_error(), "UnregisterHotKey failed")
        self._registered.remove(hotkey_id)

    def close(self) -> None:
        self._require_owner_thread()
        failures: list[int] = []
        for hotkey_id in tuple(self._registered):
            if self._facade.unregister_hotkey(self._hwnd, hotkey_id):
                self._registered.remove(hotkey_id)
            else:
                failures.append(hotkey_id)
        if failures:
            raise OSError(f"failed to unregister hotkey ids: {failures}")

    def dispatch_message(self, message: int, wparam: int) -> bool:
        """Queue an owned ``WM_HOTKEY`` and report whether it was consumed."""

        self._require_owner_thread()
        if message != WM_HOTKEY or wparam not in self._registered:
            return False
        if self._intent_queue is None:
            raise RuntimeError("hotkey dispatch requires an intent queue")
        self._intent_queue.put_nowait(wparam)
        return True

    def __enter__(self) -> GlobalHotkeyAdapter:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()
