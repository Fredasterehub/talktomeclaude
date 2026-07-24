"""Neutral canonical identity shared by assistant and reply adapters."""

from __future__ import annotations

import hashlib
import json

PROTOCOL_VERSION = 1


def canonical_reply_body_bytes(
    *, version: int, session: str, event_id: str, answer: str
) -> bytes:
    """Encode every immutable event field except its derived digest."""

    return json.dumps(
        {
            "answer": answer,
            "event_id": event_id,
            "session": session,
            "version": version,
        },
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8", errors="strict")


def canonical_reply_digest(
    *, version: int, session: str, event_id: str, answer: str
) -> str:
    """Bind answer content and all immutable event identity metadata."""

    return hashlib.sha256(
        canonical_reply_body_bytes(
            version=version,
            session=session,
            event_id=event_id,
            answer=answer,
        )
    ).hexdigest()
