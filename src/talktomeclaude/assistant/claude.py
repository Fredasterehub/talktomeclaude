"""Claude Code event semantics and strict Stop-payload validation."""

from __future__ import annotations

import hmac
import json
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Protocol, cast, runtime_checkable

from talktomeclaude.reply_protocol import PROTOCOL_VERSION, canonical_reply_digest

from .suppression import SuppressionRegistry

DEFAULT_MAX_PAYLOAD_BYTES = 8 * 1024 * 1024
MAX_IDENTIFIER_LENGTH = 128
_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,127}$")
_DIGEST = re.compile(r"^[0-9a-f]{64}$")
_REQUIRED_FIELDS = frozenset({"version", "session", "event_id", "answer", "digest"})
_OPTIONAL_SUPPRESSION_FIELDS = frozenset({"role", "correlation_id"})


@dataclass(frozen=True, slots=True)
class ValidatedAssistantEvent:
    """Validated Claude semantics; sensitive answer text is repr-hidden."""

    version: int
    session: str
    event_id: str
    digest: str
    answer: str = field(repr=False)
    role: str | None = None
    correlation_id: str | None = None

    @property
    def session_id(self) -> str:
        """Expose the neutral session attribute consumed by suppression."""
        return self.session


class AssistantEventCode(StrEnum):
    ACCEPTED = "accepted"
    SUPPRESSED_ROLE = "suppressed_role"
    SUPPRESSED_SESSION = "suppressed_session"
    SUPPRESSED_CORRELATION = "suppressed_correlation"
    INVALID_ENCODING = "invalid_encoding"
    INVALID_JSON = "invalid_json"
    INVALID_ROOT = "invalid_root"
    INVALID_VERSION = "invalid_version"
    INVALID_SESSION = "invalid_session"
    INVALID_EVENT_ID = "invalid_event_id"
    INVALID_ANSWER = "invalid_answer"
    INVALID_DIGEST = "invalid_digest"
    PAYLOAD_TOO_LARGE = "payload_too_large"
    PUBLISH_FAILED = "publish_failed"


@dataclass(frozen=True, slots=True)
class AssistantEventResult:
    code: AssistantEventCode
    event: ValidatedAssistantEvent | None = field(default=None, repr=False)

    @property
    def accepted(self) -> bool:
        return self.code is AssistantEventCode.ACCEPTED


@runtime_checkable
class AssistantAdapter(Protocol):
    """Assistant semantics without connection, framing, or retry operations."""

    @property
    def assistant_auto_submit(self) -> bool: ...

    def submit_eligible(self, *, transcript_acceptable: bool) -> bool: ...

    def validate(self, raw: bytes | str) -> AssistantEventResult: ...

    def handle(
        self,
        raw: bytes | str,
        publish: Callable[[ValidatedAssistantEvent], None],
        *,
        environment: Mapping[str, str] | None = None,
    ) -> AssistantEventResult: ...


class _DuplicateKey(ValueError):
    pass


def _object_without_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateKey(key)
        result[key] = value
    return result


def _valid_identifier(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) <= MAX_IDENTIFIER_LENGTH
        and _IDENTIFIER.fullmatch(value) is not None
    )


class ClaudeCodeAdapter:
    """Validate, suppress, then publish authoritative Claude Stop events.

    The publisher is deliberately just a callback.  Connection, framing, ACK,
    retry, and spool policy belong to ``ReplyTransport`` and are absent here.
    """

    def __init__(
        self,
        suppression: SuppressionRegistry,
        *,
        max_payload_bytes: int = DEFAULT_MAX_PAYLOAD_BYTES,
        metric: Callable[[str], None] | None = None,
        assistant_auto_submit: bool = True,
    ) -> None:
        if max_payload_bytes < 1:
            raise ValueError("max_payload_bytes must be positive")
        self._suppression = suppression
        self._max_payload_bytes = max_payload_bytes
        self._metric = metric
        self._assistant_auto_submit = assistant_auto_submit

    @property
    def assistant_auto_submit(self) -> bool:
        """Whether an acceptable assistant transcript may send one Enter."""
        return self._assistant_auto_submit

    def submit_eligible(self, *, transcript_acceptable: bool) -> bool:
        """Return Claude-mode submit eligibility without transport coupling."""
        return transcript_acceptable and self._assistant_auto_submit

    def _count(self, code: AssistantEventCode) -> None:
        if self._metric is not None:
            try:
                self._metric(code.value)
            except Exception:
                # Observability cannot change validation, suppression, or
                # publication semantics.
                pass

    def validate(self, raw: bytes | str) -> AssistantEventResult:
        if isinstance(raw, str):
            try:
                encoded = raw.encode("utf-8")
            except UnicodeEncodeError:
                return AssistantEventResult(AssistantEventCode.INVALID_ENCODING)
        elif isinstance(raw, bytes):
            encoded = raw
        else:
            return AssistantEventResult(AssistantEventCode.INVALID_ENCODING)
        if len(encoded) > self._max_payload_bytes:
            return AssistantEventResult(AssistantEventCode.PAYLOAD_TOO_LARGE)
        try:
            text = encoded.decode("utf-8", errors="strict")
        except UnicodeDecodeError:
            return AssistantEventResult(AssistantEventCode.INVALID_ENCODING)
        try:
            value = json.loads(text, object_pairs_hook=_object_without_duplicates)
        except (json.JSONDecodeError, _DuplicateKey, ValueError):
            return AssistantEventResult(AssistantEventCode.INVALID_JSON)
        if not isinstance(value, dict):
            return AssistantEventResult(AssistantEventCode.INVALID_ROOT)
        version = value.get("version")
        if type(version) is not int or version != PROTOCOL_VERSION:
            return AssistantEventResult(AssistantEventCode.INVALID_VERSION)
        fields = frozenset(value)
        if not fields <= (_REQUIRED_FIELDS | _OPTIONAL_SUPPRESSION_FIELDS):
            return AssistantEventResult(AssistantEventCode.INVALID_ROOT)
        session = value.get("session")
        if (
            not isinstance(session, str)
            or not (1 <= len(session) <= 256)
            or any(
                ord(character) < 0x20 or ord(character) == 0x7F for character in session
            )
        ):
            return AssistantEventResult(AssistantEventCode.INVALID_SESSION)
        event_id = value.get("event_id")
        if not _valid_identifier(event_id):
            return AssistantEventResult(AssistantEventCode.INVALID_EVENT_ID)
        answer = value.get("answer")
        if not isinstance(answer, str) or not answer:
            return AssistantEventResult(AssistantEventCode.INVALID_ANSWER)
        digest = value.get("digest")
        if not isinstance(digest, str) or _DIGEST.fullmatch(digest) is None:
            return AssistantEventResult(AssistantEventCode.INVALID_DIGEST)
        expected = canonical_reply_digest(
            version=version,
            session=session,
            event_id=cast(str, event_id),
            answer=answer,
        )
        if not hmac.compare_digest(digest, expected):
            return AssistantEventResult(AssistantEventCode.INVALID_DIGEST)
        role = value.get("role")
        correlation_id = value.get("correlation_id")
        if role is not None and not _valid_identifier(role):
            return AssistantEventResult(AssistantEventCode.INVALID_ROOT)
        if correlation_id is not None and not _valid_identifier(correlation_id):
            return AssistantEventResult(AssistantEventCode.INVALID_ROOT)
        event = ValidatedAssistantEvent(
            version=version,
            session=session,
            event_id=cast(str, event_id),
            answer=answer,
            digest=digest,
            role=role,
            correlation_id=correlation_id,
        )
        return AssistantEventResult(AssistantEventCode.ACCEPTED, event)

    def handle(
        self,
        raw: bytes | str,
        publish: Callable[[ValidatedAssistantEvent], None],
        *,
        environment: Mapping[str, str] | None = None,
    ) -> AssistantEventResult:
        result = self.validate(raw)
        if not result.accepted or result.event is None:
            self._count(result.code)
            return result
        reason = self._suppression.reason_for(result.event, environment=environment)
        if reason is not None:
            code = AssistantEventCode(reason)
            self._count(code)
            return AssistantEventResult(code)
        try:
            publish(result.event)
        except Exception:
            code = AssistantEventCode.PUBLISH_FAILED
            self._count(code)
            return AssistantEventResult(code)
        self._count(AssistantEventCode.ACCEPTED)
        return result
