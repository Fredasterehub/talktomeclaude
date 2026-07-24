"""Thread-owned Win32 global-hotkey pump for the non-activating Tk shell."""

from __future__ import annotations

import ctypes
import queue
import threading
import time
from collections.abc import Callable
from typing import Protocol

from talktomeclaude.platform.windows.hotkeys import (
    MOD_ALT,
    MOD_CONTROL,
    CtypesHotkeyFacade,
    GlobalHotkeyAdapter,
    HotkeyFacade,
)


WM_QUIT = 0x0012
PM_NOREMOVE = 0x0000


class MessagePump(Protocol):
    def prepare(self) -> int: ...

    def next_message(self) -> tuple[int, int] | None: ...

    def post_quit(self, thread_id: int) -> bool: ...


class KeyState(Protocol):
    def is_pressed(self, virtual_key: int) -> bool: ...


class _CtypesKeyState:
    def __init__(self) -> None:
        from ctypes import wintypes

        self._user32 = ctypes.WinDLL("user32", use_last_error=True)
        self._user32.GetAsyncKeyState.argtypes = (wintypes.INT,)
        self._user32.GetAsyncKeyState.restype = wintypes.SHORT

    def is_pressed(self, virtual_key: int) -> bool:
        return bool(self._user32.GetAsyncKeyState(virtual_key) & 0x8000)


class _CtypesMessagePump:
    def __init__(self) -> None:
        from ctypes import wintypes

        class POINT(ctypes.Structure):
            _fields_ = [("x", wintypes.LONG), ("y", wintypes.LONG)]

        class MSG(ctypes.Structure):
            _fields_ = [
                ("hwnd", wintypes.HWND),
                ("message", wintypes.UINT),
                ("wParam", wintypes.WPARAM),
                ("lParam", wintypes.LPARAM),
                ("time", wintypes.DWORD),
                ("pt", POINT),
            ]

        self._message_type = MSG
        self._message = MSG()
        self._user32 = ctypes.WinDLL("user32", use_last_error=True)
        self._kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        self._user32.PeekMessageW.argtypes = (
            ctypes.POINTER(MSG),
            wintypes.HWND,
            wintypes.UINT,
            wintypes.UINT,
            wintypes.UINT,
        )
        self._user32.PeekMessageW.restype = wintypes.BOOL
        self._user32.GetMessageW.argtypes = (
            ctypes.POINTER(MSG),
            wintypes.HWND,
            wintypes.UINT,
            wintypes.UINT,
        )
        self._user32.GetMessageW.restype = ctypes.c_int
        self._user32.PostThreadMessageW.argtypes = (
            wintypes.DWORD,
            wintypes.UINT,
            wintypes.WPARAM,
            wintypes.LPARAM,
        )
        self._user32.PostThreadMessageW.restype = wintypes.BOOL
        self._kernel32.GetCurrentThreadId.restype = wintypes.DWORD

    def prepare(self) -> int:
        # Peek creates this thread's message queue without removing anything.
        self._user32.PeekMessageW(
            ctypes.byref(self._message), None, 0, 0, PM_NOREMOVE
        )
        return int(self._kernel32.GetCurrentThreadId())

    def next_message(self) -> tuple[int, int] | None:
        result = self._user32.GetMessageW(
            ctypes.byref(self._message), None, 0, 0
        )
        if result == 0:
            return None
        if result < 0:
            raise ctypes.WinError(ctypes.get_last_error())
        return int(self._message.message), int(self._message.wParam)

    def post_quit(self, thread_id: int) -> bool:
        return bool(self._user32.PostThreadMessageW(thread_id, WM_QUIT, 0, 0))


class ThreadHotkeyListener:
    """Register and pump one no-repeat hotkey without touching Tk's thread."""

    def __init__(
        self,
        callback: Callable[[], bool | None],
        *,
        release_callback: Callable[[], None] | None = None,
        modifiers: int = MOD_CONTROL | MOD_ALT,
        virtual_key: int = 0x20,
        hotkey_id: int = 1,
        startup_deadline_seconds: float = 2.0,
        shutdown_deadline_seconds: float = 2.0,
        release_poll_seconds: float = 0.01,
        pump_factory: Callable[[], MessagePump] = _CtypesMessagePump,
        facade_factory: Callable[[], HotkeyFacade] = CtypesHotkeyFacade,
        key_state_factory: Callable[[], KeyState] = _CtypesKeyState,
    ) -> None:
        if (
            startup_deadline_seconds <= 0
            or shutdown_deadline_seconds <= 0
            or release_poll_seconds <= 0
        ):
            raise ValueError("hotkey lifecycle deadlines must be positive")
        self._callback = callback
        self._release_callback = release_callback
        self._modifiers = modifiers
        self._virtual_key = virtual_key
        self._hotkey_id = hotkey_id
        self._startup_deadline = startup_deadline_seconds
        self._shutdown_deadline = shutdown_deadline_seconds
        self._release_poll_seconds = release_poll_seconds
        self._pump_factory = pump_factory
        self._facade_factory = facade_factory
        self._key_state_factory = key_state_factory
        self._intents: queue.SimpleQueue[int | None] = queue.SimpleQueue()
        self._ready = threading.Event()
        self._stopped = threading.Event()
        self._stop_requested = threading.Event()
        self._guard = threading.Lock()
        self._pump: MessagePump | None = None
        self._thread_id: int | None = None
        self._error: BaseException | None = None
        self._owner: threading.Thread | None = None
        self._dispatcher: threading.Thread | None = None
        self._started = False

    def start(self) -> None:
        with self._guard:
            if self._started:
                return
            self._started = True
            self._dispatcher = threading.Thread(
                target=self._dispatch,
                name="ttc-hotkey-intent-dispatch",
                daemon=True,
            )
            self._owner = threading.Thread(
                target=self._run,
                name="ttc-win32-hotkey-pump",
                daemon=True,
            )
            self._dispatcher.start()
            self._owner.start()
        if not self._ready.wait(self._startup_deadline):
            self.stop()
            raise TimeoutError("global hotkey registration timed out")
        if self._error is not None:
            error = self._error
            self.stop()
            raise RuntimeError("global hotkey registration failed") from error

    def _run(self) -> None:
        adapter: GlobalHotkeyAdapter | None = None
        try:
            pump = self._pump_factory()
            self._pump = pump
            self._thread_id = pump.prepare()
            adapter = GlobalHotkeyAdapter(
                self._facade_factory(), hwnd=0, intent_queue=self._intents
            )
            adapter.register(self._hotkey_id, self._modifiers, self._virtual_key)
            self._ready.set()
            while True:
                message = pump.next_message()
                if message is None:
                    break
                adapter.dispatch_message(*message)
        except BaseException as exc:
            self._error = exc
            self._ready.set()
        finally:
            if adapter is not None:
                try:
                    adapter.close()
                except Exception as exc:
                    if self._error is None:
                        self._error = exc
            self._intents.put(None)
            self._stopped.set()

    def _dispatch(self) -> None:
        key_state: KeyState | None = None
        while True:
            intent = self._intents.get()
            if intent is None or self._stop_requested.is_set():
                return
            if intent != self._hotkey_id:
                continue
            try:
                monitor_release = self._callback() is True
                if monitor_release and self._release_callback is not None:
                    key_state = key_state or self._key_state_factory()
                    while key_state.is_pressed(self._virtual_key):
                        if self._stop_requested.wait(self._release_poll_seconds):
                            return
                    if not self._stop_requested.is_set():
                        self._release_callback()
            except Exception:
                pass

    def stop(self) -> bool:
        self._stop_requested.set()
        self._intents.put(None)
        with self._guard:
            if not self._started:
                return True
            pump = self._pump
            thread_id = self._thread_id
        if pump is not None and thread_id is not None:
            pump.post_quit(thread_id)
        with self._guard:
            owner = self._owner
            dispatcher = self._dispatcher
        deadline = time.monotonic() + self._shutdown_deadline
        if owner is not None:
            owner.join(max(0.0, deadline - time.monotonic()))
        if dispatcher is not None:
            dispatcher.join(max(0.0, deadline - time.monotonic()))
        return bool(
            self._stopped.is_set()
            and (owner is None or not owner.is_alive())
            and (dispatcher is None or not dispatcher.is_alive())
        )


__all__ = ["ThreadHotkeyListener"]
