"""Model-free subprocess fixture for the Windows speech worker tests."""

from __future__ import annotations

import os
import sys
import time
from multiprocessing.connection import Client
from pathlib import Path
from typing import Any

from talktomeclaude.speech.subprocess_worker import serve


def _synthesize(text: str, path: Path, selected_voice: str) -> None:
    if text.startswith("BLOCK"):
        time.sleep(30)
    path.write_bytes(f"{selected_voice}|{text}".encode("utf-8"))


def main() -> int:
    if len(sys.argv) != 3:
        return 2
    selected_voice, artifact_root = sys.argv[1:]
    address = os.environ.get("TTC_SYNTHESIS_PIPE")
    authkey_hex = os.environ.get("TTC_SYNTHESIS_AUTHKEY")
    if not address or not authkey_hex:
        return 2
    connection: Any = Client(
        address,
        family="AF_PIPE",
        authkey=bytes.fromhex(authkey_hex),
    )
    try:
        serve(
            connection,
            selected_voice,
            artifact_root,
            synthesize_fn=_synthesize,
        )
    finally:
        connection.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
