"""Content-safe companion diagnostics and deterministic support exports.

The diagnostic surface is intentionally narrow: callers record semantic state,
capability, queue, retry, identity-hash, and error-code metadata.  User content
never belongs in this store.  Export applies a second defensive redaction pass
so a future caller cannot accidentally turn a support bundle into a transcript.
"""

from __future__ import annotations

import hashlib
import math
import re
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from talktomeclaude.storage import AtomicJsonTransaction, AtomicStorageError


DIAGNOSTIC_VERSION = 1
REDACTED = "[redacted]"

_SENSITIVE_KEYS = frozenset(
    {
        "answer",
        "audio",
        "canonical_answer",
        "command",
        "home",
        "prompt",
        "raw_audio",
        "reference",
        "reference_path",
        "secret",
        "spoken_text",
        "text",
        "token",
        "transcript",
    }
)
_SECRET = re.compile(
    r"(?i)(?:bearer\s+[a-z0-9._~+/=-]+|(?:sk|hf)_[a-z0-9_-]{8,})"
)
_HOME_PATH = re.compile(
    r"(?i)(?:[a-z]:[\\/]users[\\/][^\\/\s]+|/(?:home|users)/[^/\s]+)"
)
_SSH_OPTION = re.compile(
    r"(?i)(?:-o\s*(?:identityfile|proxycommand|proxyjump)\s*=?.*|"
    r"(?:identityfile|proxycommand|proxyjump)\s*=\s*.*)"
)
_REFERENCE_PATH = re.compile(
    r"(?i)(?:voice[-_ ]refs?|reference(?:_path)?)\s*[:=]\s*\S+"
)
_EVENT_KIND = re.compile(r"^[a-z][a-z0-9_]{0,63}$")


def opaque_identity(value: str) -> str:
    """Return a stable, content-free identity suitable for diagnostics."""

    if not isinstance(value, str) or not value:
        raise ValueError("diagnostic identity must not be empty")
    return hashlib.sha256(value.encode("utf-8", errors="strict")).hexdigest()


def _redacted_marker(value: str) -> str:
    digest = hashlib.sha256(value.encode("utf-8", errors="replace")).hexdigest()[:12]
    return f"{REDACTED}:{digest}"


def redact(value: Any, *, key: str | None = None) -> Any:
    """Recursively redact content fields and recognizable secret/path forms."""

    normalized_key = key.casefold().replace("-", "_") if key else None
    if normalized_key in _SENSITIVE_KEYS:
        return REDACTED
    if isinstance(value, Mapping):
        return {
            str(item_key): redact(item_value, key=str(item_key))
            for item_key, item_value in sorted(
                value.items(), key=lambda item: str(item[0])
            )
        }
    if isinstance(value, (list, tuple)):
        return [redact(item) for item in value]
    if isinstance(value, str):
        if any(
            pattern.search(value)
            for pattern in (_SECRET, _HOME_PATH, _SSH_OPTION, _REFERENCE_PATH)
        ):
            return _redacted_marker(value)
        return value
    if value is None or isinstance(value, (bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else REDACTED
    return REDACTED


@dataclass(frozen=True, slots=True)
class DiagnosticEvent:
    """One monotonic, content-safe diagnostic event."""

    sequence: int
    monotonic_seconds: float
    kind: str
    fields: Mapping[str, Any]

    def to_document(self) -> dict[str, Any]:
        kind = self.kind if _EVENT_KIND.fullmatch(self.kind) else "redacted_event"
        return {
            "fields": redact(dict(self.fields)),
            "kind": kind,
            "monotonic_seconds": self.monotonic_seconds,
            "sequence": self.sequence,
        }


def _empty_store() -> dict[str, Any]:
    return {"events": [], "next_sequence": 1, "version": DIAGNOSTIC_VERSION}


class DiagnosticStore:
    """Bounded transactional metric store that never raises into product work."""

    def __init__(
        self,
        path: str | Path,
        *,
        maximum_events: int = 2000,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        if maximum_events < 1:
            raise ValueError("diagnostic event capacity must be positive")
        self.path = Path(path)
        self._maximum_events = maximum_events
        self._monotonic = monotonic
        self._transaction = AtomicJsonTransaction(
            self.path, purpose="companion-diagnostics"
        )

    def record(self, kind: str, **fields: Any) -> bool:
        if not isinstance(kind, str) or not kind.strip():
            raise ValueError("diagnostic kind must not be empty")
        if not _EVENT_KIND.fullmatch(kind.strip()):
            raise ValueError("diagnostic kind must be a semantic identifier")
        safe_fields = redact(fields)

        def update(current: dict[str, Any]) -> dict[str, Any]:
            state = self._validated_or_empty(current)
            sequence = state["next_sequence"]
            event = DiagnosticEvent(
                sequence, float(self._monotonic()), kind.strip(), safe_fields
            ).to_document()
            events = [*state["events"], event][-self._maximum_events :]
            return {
                "events": events,
                "next_sequence": sequence + 1,
                "version": DIAGNOSTIC_VERSION,
            }

        try:
            self._transaction.update(update)
        except (OSError, AtomicStorageError, UnicodeError, ValueError):
            return False
        return True

    @staticmethod
    def _validated_or_empty(value: object) -> dict[str, Any]:
        if not isinstance(value, dict) or value.get("version") != DIAGNOSTIC_VERSION:
            return _empty_store()
        events = value.get("events")
        next_sequence = value.get("next_sequence")
        if (
            not isinstance(events, list)
            or isinstance(next_sequence, bool)
            or not isinstance(next_sequence, int)
            or next_sequence < 1
        ):
            return _empty_store()
        valid_events = [
            redact(event)
            for event in events
            if isinstance(event, dict)
            and isinstance(event.get("kind"), str)
            and _EVENT_KIND.fullmatch(event["kind"])
            and isinstance(event.get("sequence"), int)
        ]
        return {
            "events": valid_events,
            "next_sequence": next_sequence,
            "version": DIAGNOSTIC_VERSION,
        }

    def snapshot(self) -> tuple[dict[str, Any], bool]:
        """Return validated events and whether corrupt storage was recovered."""

        try:
            raw = self._transaction.read()
        except (OSError, AtomicStorageError, UnicodeError, ValueError):
            return _empty_store(), True
        state = self._validated_or_empty(raw)
        return state, state == _empty_store() and bool(raw)

    def export(self, destination: str | Path) -> Path:
        state, recovered = self.snapshot()
        document = {
            "diagnostics": redact(state["events"]),
            "manifest": {
                "included": [
                    "capabilities",
                    "content-free error codes",
                    "hashed event identities",
                    "monotonic timings",
                    "queue depths and retry counts",
                    "semantic state transitions",
                ],
                "omitted": [
                    "answers and spoken text",
                    "audio and transcripts",
                    "full environment and home paths",
                    "prompts and tokens",
                    "SSH secrets and options",
                    "voice reference content and paths",
                ],
                "partial_store_recovered": recovered,
            },
            "version": DIAGNOSTIC_VERSION,
        }
        output = Path(destination)
        AtomicJsonTransaction(output, purpose="diagnostic-export").write(document)
        return output


__all__ = [
    "DIAGNOSTIC_VERSION",
    "DiagnosticEvent",
    "DiagnosticStore",
    "REDACTED",
    "opaque_identity",
    "redact",
]
