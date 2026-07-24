"""Content-safe contracts for text delivery.

These results are suitable for diagnostics.  They deliberately contain neither
transcript text nor native window/process identifiers.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class DeliveryMode(str, Enum):
    GENERIC = "generic"
    ASSISTANT = "assistant"


class DeliveryCode(str, Enum):
    DELIVERED = "delivered"
    CANCELLED = "cancelled"
    EMPTY_TRANSCRIPT = "empty_transcript"
    INVALID_TARGET = "invalid_target"
    TARGET_CHANGED_PRE_CLIPBOARD = "target_changed_pre_clipboard"
    TARGET_CHANGED_PRE_PASTE = "target_changed_pre_paste"
    CLIPBOARD_OPEN_TIMEOUT = "clipboard_open_timeout"
    CLIPBOARD_UNSUPPORTED_FORMAT = "clipboard_unsupported_format"
    CLIPBOARD_READ_FAILED = "clipboard_read_failed"
    CLIPBOARD_SET_FAILED = "clipboard_set_failed"
    PASTE_TIMEOUT = "paste_timeout"
    PASTE_FAILED = "paste_failed"
    PASTED_NOT_SUBMITTED = "pasted_not_submitted"
    ENTER_TIMEOUT = "enter_timeout"
    ENTER_FAILED = "enter_failed"


class RestoreStatus(str, Enum):
    NOT_NEEDED = "not_needed"
    RESTORED = "restored"
    CONFLICT = "conflict"
    OPEN_TIMEOUT = "open_timeout"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class DeliveryResult:
    """Outcome of exactly one explicit delivery transaction."""

    code: DeliveryCode
    pasted: bool = False
    submitted: bool = False
    restore_status: RestoreStatus = RestoreStatus.NOT_NEEDED
    target_reason: str | None = None

    @property
    def succeeded(self) -> bool:
        return self.code is DeliveryCode.DELIVERED
