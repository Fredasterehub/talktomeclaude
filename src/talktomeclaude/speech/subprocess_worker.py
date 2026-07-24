"""Dedicated synthesis subprocess used by the Windows companion.

The uv console launcher can leave ``multiprocessing`` children unable to load
NumPy's native extension modules.  This entry point starts through an ordinary
``python -m`` boundary and communicates over one authenticated local pipe.
"""

from __future__ import annotations

import os
import sys
import tempfile
from collections.abc import Callable
from multiprocessing.connection import Client
from pathlib import Path
from typing import Any, Protocol


class WorkerConnection(Protocol):
    def recv(self) -> object: ...

    def send(self, value: object) -> None: ...

    def close(self) -> None: ...


def _production_synthesize(text: str, path: Path, selected_voice: str) -> None:
    from talktomeclaude.speech import voices

    voices.synthesize(text, path, selected_voice)


def serve(
    connection: WorkerConnection,
    selected_voice: str,
    artifact_root: str | os.PathLike[str],
    *,
    synthesize_fn: Callable[[str, Path, str], None] = _production_synthesize,
) -> None:
    """Serve validated synthesis requests without returning content in faults."""

    root = Path(artifact_root)
    root.mkdir(parents=True, exist_ok=True)
    connection.send({"kind": "ready"})
    while True:
        try:
            message = connection.recv()
        except EOFError:
            return
        if not isinstance(message, dict):
            return
        if message.get("kind") == "stop":
            return
        if message.get("kind") != "synthesize":
            return
        job_id = message.get("job_id")
        text = message.get("text")
        safe_job_id = bool(
            isinstance(job_id, str)
            and 0 < len(job_id) <= 128
            and all(character.isalnum() or character in "-_" for character in job_id)
        )
        if not safe_job_id or not isinstance(text, str) or not text:
            return
        assert isinstance(job_id, str)

        descriptor, raw_temporary = tempfile.mkstemp(
            prefix=f".{job_id}.", suffix=".tmp.wav", dir=root
        )
        os.close(descriptor)
        temporary = Path(raw_temporary)
        final = root / f"{job_id}.wav"
        succeeded = False
        try:
            synthesize_fn(text, temporary, selected_voice)
            if not temporary.is_file() or temporary.stat().st_size <= 0:
                raise OSError("synthesis produced no artifact")
            os.replace(temporary, final)
            succeeded = True
        except BaseException:
            final.unlink(missing_ok=True)
        finally:
            temporary.unlink(missing_ok=True)
        try:
            connection.send(
                {
                    "kind": "reply",
                    "job_id": job_id,
                    "succeeded": succeeded,
                    "artifact_path": str(final) if succeeded else None,
                }
            )
        except (EOFError, OSError):
            final.unlink(missing_ok=True)
            return


def main(argv: list[str] | None = None) -> int:
    arguments = sys.argv[1:] if argv is None else argv
    if len(arguments) != 2:
        return 2
    selected_voice, artifact_root = arguments
    address = os.environ.get("TTC_SYNTHESIS_PIPE")
    authkey_hex = os.environ.get("TTC_SYNTHESIS_AUTHKEY")
    if not address or not authkey_hex:
        return 2
    try:
        authkey = bytes.fromhex(authkey_hex)
        connection: Any = Client(address, family="AF_PIPE", authkey=authkey)
    except (OSError, ValueError):
        return 1
    try:
        serve(connection, selected_voice, artifact_root)
    finally:
        connection.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
