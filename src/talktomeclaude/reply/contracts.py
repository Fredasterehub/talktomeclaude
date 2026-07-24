"""Versioned, canonical wire contracts for durable assistant replies.

The answer is deliberately excluded from object representations and exception
messages.  Event identifiers and digests are opaque diagnostics-safe values.
"""

from __future__ import annotations

import hmac
import json
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, ClassVar

from talktomeclaude.reply_protocol import PROTOCOL_VERSION, canonical_reply_digest

MAX_WIRE_BYTES = 8 * 1024 * 1024
_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,127}$")
_DIGEST = re.compile(r"^[0-9a-f]{64}$")


class ReplyProtocolError(ValueError):
    """A wire value does not satisfy the reply protocol."""


class ReceiveDisposition(str, Enum):
    COMMITTED = "committed"
    DUPLICATE = "duplicate"
    QUARANTINED = "quarantined"
    CORRUPT = "corrupt"


class AckDisposition(str, Enum):
    COMMITTED = "committed"
    ALREADY_COMMITTED = "already_committed"
    REJECTED = "rejected"


class DiagnosticCode(str, Enum):
    EVENT_COMMITTED = "reply_event_committed"
    EVENT_DUPLICATE = "reply_event_duplicate"
    EVENT_DIGEST_CONFLICT = "reply_event_digest_conflict"
    EVENT_CORRUPT = "reply_event_corrupt"
    ACK_COMMITTED = "reply_ack_committed"
    ACK_DUPLICATE = "reply_ack_duplicate"
    ACK_REJECTED = "reply_ack_rejected"
    SPOOL_FULL = "reply_spool_full"
    READY_REPLAY = "reply_ready_replay"
    CURSOR_RECOVERED = "reply_cursor_recovered"
    RETENTION_APPLIED = "reply_retention_applied"


def _canonical_json(value: dict[str, Any]) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeError) as exc:
        raise ReplyProtocolError("reply value is not canonical JSON") from exc


def _decode_object(wire: bytes, *, maximum: int = MAX_WIRE_BYTES) -> dict[str, Any]:
    if not wire or len(wire) > maximum:
        raise ReplyProtocolError("reply frame size is invalid")
    try:
        text = wire.decode("utf-8", errors="strict")
        value = json.loads(text)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ReplyProtocolError("reply frame is not valid UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise ReplyProtocolError("reply frame must contain an object")
    return value


def _validate_identifier(value: object, name: str) -> str:
    if not isinstance(value, str) or _IDENTIFIER.fullmatch(value) is None:
        raise ReplyProtocolError(f"{name} is invalid")
    return value


def _validate_session(value: object) -> str:
    if not isinstance(value, str) or not (1 <= len(value) <= 256):
        raise ReplyProtocolError("session is invalid")
    if any(ord(character) < 0x20 or ord(character) == 0x7F for character in value):
        raise ReplyProtocolError("session is invalid")
    try:
        value.encode("utf-8", errors="strict")
    except UnicodeError as exc:
        raise ReplyProtocolError("session is invalid") from exc
    return value


def _validate_digest(value: object) -> str:
    if not isinstance(value, str) or _DIGEST.fullmatch(value) is None:
        raise ReplyProtocolError("digest is invalid")
    return value


@dataclass(frozen=True, slots=True)
class ReplyEvent:
    """One immutable authoritative assistant reply."""

    version: int
    session: str
    event_id: str
    answer: str = field(repr=False)
    digest: str

    BODY_KEYS: ClassVar[frozenset[str]] = frozenset(
        {"version", "session", "event_id", "answer"}
    )
    WIRE_KEYS: ClassVar[frozenset[str]] = BODY_KEYS | {"digest"}

    def __post_init__(self) -> None:
        if type(self.version) is not int or self.version != PROTOCOL_VERSION:
            raise ReplyProtocolError("reply protocol version is unsupported")
        _validate_session(self.session)
        _validate_identifier(self.event_id, "event_id")
        if not isinstance(self.answer, str) or not self.answer:
            raise ReplyProtocolError("answer is invalid")
        try:
            self.answer.encode("utf-8", errors="strict")
        except UnicodeError as exc:
            raise ReplyProtocolError("answer is invalid") from exc
        _validate_digest(self.digest)
        if not hmac.compare_digest(self.digest, self.compute_digest()):
            raise ReplyProtocolError("reply digest does not match the canonical payload")

    @classmethod
    def create(cls, *, session: str, event_id: str, answer: str) -> "ReplyEvent":
        if not isinstance(answer, str) or not answer:
            raise ReplyProtocolError("answer is invalid")
        digest = canonical_reply_digest(
            version=PROTOCOL_VERSION,
            session=session,
            event_id=event_id,
            answer=answer,
        )
        return cls(PROTOCOL_VERSION, session, event_id, answer, digest)

    def _body(self) -> dict[str, Any]:
        return {
            "answer": self.answer,
            "event_id": self.event_id,
            "session": self.session,
            "version": self.version,
        }

    def compute_digest(self) -> str:
        return canonical_reply_digest(
            version=self.version,
            session=self.session,
            event_id=self.event_id,
            answer=self.answer,
        )

    def to_bytes(self) -> bytes:
        return _canonical_json({**self._body(), "digest": self.digest})

    @classmethod
    def from_bytes(cls, wire: bytes, *, require_canonical: bool = True) -> "ReplyEvent":
        value = _decode_object(wire)
        if frozenset(value) != cls.WIRE_KEYS:
            raise ReplyProtocolError("reply event fields are invalid")
        event = cls(
            value["version"],
            _validate_session(value["session"]),
            _validate_identifier(value["event_id"], "event_id"),
            value["answer"],
            _validate_digest(value["digest"]),
        )
        if require_canonical and event.to_bytes() != wire:
            raise ReplyProtocolError("reply event is not canonical JSON")
        return event


@dataclass(frozen=True, slots=True)
class ReplyAck:
    version: int
    event_id: str
    digest: str

    WIRE_KEYS: ClassVar[frozenset[str]] = frozenset(
        {"version", "event_id", "digest"}
    )

    def __post_init__(self) -> None:
        if type(self.version) is not int or self.version != PROTOCOL_VERSION:
            raise ReplyProtocolError("reply protocol version is unsupported")
        _validate_identifier(self.event_id, "event_id")
        _validate_digest(self.digest)

    @classmethod
    def for_event(cls, event: ReplyEvent) -> "ReplyAck":
        return cls(PROTOCOL_VERSION, event.event_id, event.digest)

    def to_bytes(self) -> bytes:
        return _canonical_json(
            {
                "digest": self.digest,
                "event_id": self.event_id,
                "version": self.version,
            }
        )

    @classmethod
    def from_bytes(cls, wire: bytes, *, require_canonical: bool = True) -> "ReplyAck":
        value = _decode_object(wire, maximum=1024)
        if frozenset(value) != cls.WIRE_KEYS:
            raise ReplyProtocolError("reply ACK fields are invalid")
        ack = cls(
            value["version"],
            _validate_identifier(value["event_id"], "event_id"),
            _validate_digest(value["digest"]),
        )
        if require_canonical and ack.to_bytes() != wire:
            raise ReplyProtocolError("reply ACK is not canonical JSON")
        return ack


@dataclass(frozen=True, slots=True)
class ReceiveResult:
    disposition: ReceiveDisposition
    ack: ReplyAck | None
    apply: bool
    diagnostic: DiagnosticCode
    event_id: str | None = None
    digest: str | None = None


@dataclass(frozen=True, slots=True)
class AckResult:
    disposition: AckDisposition
    diagnostic: DiagnosticCode
    event_id: str
    digest: str


@dataclass(frozen=True, slots=True)
class ReplyDiagnostic:
    """Content-free durable-transport observability event."""

    code: DiagnosticCode
    event_id: str | None = None
    digest: str | None = None
    count: int | None = None
