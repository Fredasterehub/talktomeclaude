"""Fail-closed text injection into one ephemeral foreground target."""

from __future__ import annotations

import ctypes
import os
import time
from ctypes import wintypes
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Protocol

from talktomeclaude.platform.contracts import (
    DeliveryCode,
    DeliveryMode,
    DeliveryResult,
)

from .clipboard import ClipboardCode, ClipboardTransaction
from .target import TargetEvidence, TargetResolution, WindowsTargetResolver


class KeyboardFacade(Protocol):
    def send_paste(self) -> KeySendOutcome: ...
    def send_enter(self) -> KeySendOutcome: ...


class KeySendCode(str, Enum):
    SENT = "sent"
    FAILED = "failed"
    TIMEOUT = "timeout"


@dataclass(frozen=True, slots=True)
class KeySendOutcome:
    """Content-free result from one synchronous ``SendInput`` call."""

    code: KeySendCode
    inserted: int
    expected: int
    elapsed_seconds: float

    @property
    def completed(self) -> bool:
        return self.inserted == self.expected


def _coerce_key_send(value: object, *, expected: int) -> KeySendOutcome:
    """Accept the pre-outcome boolean seam used by existing injected fakes."""

    if isinstance(value, KeySendOutcome):
        return value
    if isinstance(value, bool):
        return KeySendOutcome(
            KeySendCode.SENT if value else KeySendCode.FAILED,
            expected if value else 0,
            expected,
            0.0,
        )
    raise TypeError("keyboard adapter returned an invalid outcome")


class CtypesKeyboardFacade:
    """Checked, serialized SendInput sequences for paste and Enter."""

    INPUT_KEYBOARD = 1
    KEYEVENTF_KEYUP = 0x0002
    VK_CONTROL = 0x11
    VK_V = 0x56
    VK_RETURN = 0x0D

    class _KEYBDINPUT(ctypes.Structure):
        _fields_ = [
            ("wVk", wintypes.WORD),
            ("wScan", wintypes.WORD),
            ("dwFlags", wintypes.DWORD),
            ("time", wintypes.DWORD),
            ("dwExtraInfo", ctypes.c_size_t),
        ]

    class _MOUSEINPUT(ctypes.Structure):
        _fields_ = [
            ("dx", wintypes.LONG),
            ("dy", wintypes.LONG),
            ("mouseData", wintypes.DWORD),
            ("dwFlags", wintypes.DWORD),
            ("time", wintypes.DWORD),
            ("dwExtraInfo", ctypes.c_size_t),
        ]

    class _HARDWAREINPUT(ctypes.Structure):
        _fields_ = [
            ("uMsg", wintypes.DWORD),
            ("wParamL", wintypes.WORD),
            ("wParamH", wintypes.WORD),
        ]

    class _INPUTUNION(ctypes.Union):
        pass

    class _INPUT(ctypes.Structure):
        pass

    _INPUTUNION._fields_ = [
        ("ki", _KEYBDINPUT),
        ("mi", _MOUSEINPUT),
        ("hi", _HARDWAREINPUT),
    ]
    _INPUT._anonymous_ = ("u",)
    _INPUT._fields_ = [("type", wintypes.DWORD), ("u", _INPUTUNION)]

    def __init__(
        self,
        *,
        paste_deadline: float = 0.1,
        enter_deadline: float = 0.1,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        if os.name != "nt":
            raise OSError("Windows input APIs are available only on Windows")
        if paste_deadline <= 0 or enter_deadline <= 0:
            raise ValueError("input deadlines must be positive")
        self._paste_deadline = paste_deadline
        self._enter_deadline = enter_deadline
        self._monotonic = monotonic
        self._user32 = ctypes.WinDLL("user32", use_last_error=True)
        self._user32.SendInput.argtypes = [
            wintypes.UINT,
            ctypes.POINTER(self._INPUT),
            ctypes.c_int,
        ]
        self._user32.SendInput.restype = wintypes.UINT

    @classmethod
    def _key(cls, vk: int, *, up: bool = False) -> _INPUT:
        return cls._INPUT(
            type=cls.INPUT_KEYBOARD,
            ki=cls._KEYBDINPUT(
                wVk=vk,
                wScan=0,
                dwFlags=cls.KEYEVENTF_KEYUP if up else 0,
                time=0,
                dwExtraInfo=0,
            ),
        )

    def _send(
        self,
        inputs: list[_INPUT],
        cleanup: list[_INPUT],
        deadline: float,
    ) -> KeySendOutcome:
        """Call immediate ``SendInput`` synchronously and measure its deadline.

        Win32 exposes no cancellable asynchronous SendInput operation.  Keeping
        this call on the owning thread prevents a worker from injecting keys
        after the caller has already observed a timeout.  The documented API is
        immediate; elapsed time is nevertheless measured and surfaced.
        """

        started = self._monotonic()
        array_type = self._INPUT * len(inputs)
        array = array_type(*inputs)
        sent = self._user32.SendInput(len(inputs), array, ctypes.sizeof(self._INPUT))
        primary_elapsed = self._monotonic() - started
        if int(sent) == len(inputs):
            code = (
                KeySendCode.TIMEOUT
                if primary_elapsed > deadline
                else KeySendCode.SENT
            )
            return KeySendOutcome(code, int(sent), len(inputs), primary_elapsed)
        if sent:
            cleanup_type = self._INPUT * len(cleanup)
            cleanup_array = cleanup_type(*cleanup)
            self._user32.SendInput(
                len(cleanup), cleanup_array, ctypes.sizeof(self._INPUT)
            )
        elapsed = self._monotonic() - started
        code = KeySendCode.TIMEOUT if primary_elapsed > deadline else KeySendCode.FAILED
        return KeySendOutcome(code, int(sent), len(inputs), elapsed)

    def send_paste(self) -> KeySendOutcome:
        return self._send(
            [
                self._key(self.VK_CONTROL),
                self._key(self.VK_V),
                self._key(self.VK_V, up=True),
                self._key(self.VK_CONTROL, up=True),
            ],
            [self._key(self.VK_V, up=True), self._key(self.VK_CONTROL, up=True)],
            self._paste_deadline,
        )

    def send_enter(self) -> KeySendOutcome:
        return self._send(
            [self._key(self.VK_RETURN), self._key(self.VK_RETURN, up=True)],
            [self._key(self.VK_RETURN, up=True)],
            self._enter_deadline,
        )


class TextInjector:
    """Execute one no-retarget/no-retry delivery transaction.

    Call :meth:`snapshot_target` at finish-toggle and retain the returned evidence
    only until :meth:`deliver` completes.  Results are content-safe.
    """

    def __init__(
        self,
        resolver: WindowsTargetResolver | None = None,
        keyboard: KeyboardFacade | None = None,
        clipboard_factory: Callable[[], ClipboardTransaction] | None = None,
    ) -> None:
        self._resolver = resolver or WindowsTargetResolver()
        self._keyboard = keyboard or CtypesKeyboardFacade()
        self._clipboard_factory = clipboard_factory or ClipboardTransaction

    def snapshot_target(self) -> TargetResolution:
        return self._resolver.snapshot()

    def deliver(
        self,
        text: str,
        evidence: TargetEvidence | None,
        *,
        mode: DeliveryMode,
        auto_submit: bool,
    ) -> DeliveryResult:
        if not text or not text.strip():
            return DeliveryResult(DeliveryCode.EMPTY_TRANSCRIPT)
        if evidence is None:
            return DeliveryResult(DeliveryCode.INVALID_TARGET)

        pre_clipboard = self._resolver.validate(evidence)
        if not pre_clipboard.valid:
            return DeliveryResult(
                DeliveryCode.TARGET_CHANGED_PRE_CLIPBOARD,
                target_reason=pre_clipboard.code.value,
            )

        clipboard = self._clipboard_factory()
        snapshot = clipboard.snapshot()
        if not snapshot.succeeded:
            if snapshot.code is ClipboardCode.OPEN_TIMEOUT:
                code = DeliveryCode.CLIPBOARD_OPEN_TIMEOUT
            elif snapshot.code is ClipboardCode.UNSUPPORTED_FORMAT:
                code = DeliveryCode.CLIPBOARD_UNSUPPORTED_FORMAT
            else:
                code = DeliveryCode.CLIPBOARD_READ_FAILED
            return DeliveryResult(code)

        replacement = clipboard.set_text(text)
        if not replacement.succeeded:
            restore = clipboard.restore()
            code = (
                DeliveryCode.CLIPBOARD_OPEN_TIMEOUT
                if replacement.code is ClipboardCode.OPEN_TIMEOUT
                else DeliveryCode.CLIPBOARD_SET_FAILED
            )
            return DeliveryResult(code, restore_status=restore)

        pre_paste = self._resolver.validate(evidence)
        if not pre_paste.valid:
            return DeliveryResult(
                DeliveryCode.TARGET_CHANGED_PRE_PASTE,
                restore_status=clipboard.restore(),
                target_reason=pre_paste.code.value,
            )

        paste = _coerce_key_send(self._keyboard.send_paste(), expected=4)
        if paste.code is KeySendCode.TIMEOUT:
            return DeliveryResult(
                DeliveryCode.PASTE_TIMEOUT,
                pasted=paste.completed,
                restore_status=clipboard.restore(),
            )
        if paste.code is not KeySendCode.SENT:
            return DeliveryResult(
                DeliveryCode.PASTE_FAILED,
                restore_status=clipboard.restore(),
            )

        should_submit = mode is DeliveryMode.ASSISTANT and auto_submit
        if not should_submit:
            return DeliveryResult(
                DeliveryCode.DELIVERED,
                pasted=True,
                restore_status=clipboard.restore(),
            )

        pre_enter = self._resolver.validate(evidence)
        if not pre_enter.valid:
            return DeliveryResult(
                DeliveryCode.PASTED_NOT_SUBMITTED,
                pasted=True,
                restore_status=clipboard.restore(),
                target_reason=pre_enter.code.value,
            )

        enter = _coerce_key_send(self._keyboard.send_enter(), expected=2)
        if enter.code is KeySendCode.TIMEOUT:
            return DeliveryResult(
                DeliveryCode.ENTER_TIMEOUT,
                pasted=True,
                submitted=enter.completed,
                restore_status=clipboard.restore(),
            )
        if enter.code is not KeySendCode.SENT:
            return DeliveryResult(
                DeliveryCode.ENTER_FAILED,
                pasted=True,
                restore_status=clipboard.restore(),
            )
        return DeliveryResult(
            DeliveryCode.DELIVERED,
            pasted=True,
            submitted=True,
            restore_status=clipboard.restore(),
        )
