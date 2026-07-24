"""Remote-side durable reply spool streamer.

The helper speaks one protocol on stdout/stdin: canonical NDJSON
``ReplyEvent`` frames out and canonical ``ReplyAck`` frames back.  Diagnostic
callbacks never receive reply text, and the command-line entry point never
writes protocol or reply content anywhere except stdout.
"""

from __future__ import annotations

import argparse
import hmac
import math
import sys
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from queue import Empty, Queue
from typing import BinaryIO, Callable, Sequence

from .contracts import AckDisposition, ReplyAck, ReplyProtocolError
from .spool import ReplySpool, ReplySpoolError


_MAX_ACK_LINE_BYTES = 1025  # 1024-byte canonical ACK plus its newline.


class RemoteStreamCode(str, Enum):
    """Content-free remote helper diagnostics."""

    STARTED = "started"
    EVENT_EMITTED = "event_emitted"
    ACK_COMMITTED = "ack_committed"
    ACK_REJECTED = "ack_rejected"
    ACK_INVALID = "ack_invalid"
    ACK_TIMEOUT = "ack_timeout"
    INPUT_EOF = "input_eof"
    INPUT_ERROR = "input_error"
    OUTPUT_ERROR = "output_error"
    SPOOL_ERROR = "spool_error"
    STOPPED = "stopped"


@dataclass(frozen=True, slots=True)
class RemoteStreamDiagnostic:
    """A diagnostic that cannot contain an answer or protocol frame."""

    code: RemoteStreamCode
    count: int | None = None


@dataclass(frozen=True, slots=True)
class RemoteStreamResult:
    events_emitted: int
    acknowledgements_committed: int
    acknowledgements_rejected: int
    protocol_errors: int
    input_eof: bool
    reader_stopped: bool
    boundary_replacement_required: bool


class _InputKind(str, Enum):
    LINE = "line"
    PARTIAL = "partial"
    EOF = "eof"
    ERROR = "error"


@dataclass(frozen=True, slots=True)
class _InputItem:
    kind: _InputKind
    wire: bytes = field(default=b"", repr=False)


def _read_ack_input(stream: BinaryIO, output: Queue[_InputItem]) -> None:
    """Read ACK lines without ever decoding or logging their contents."""

    try:
        while True:
            line = stream.readline(_MAX_ACK_LINE_BYTES + 1)
            if not line:
                output.put(_InputItem(_InputKind.EOF))
                return
            if not line.endswith(b"\n"):
                output.put(_InputItem(_InputKind.PARTIAL, line))
                # A bounded readline may return a prefix of an oversized line.
                # Treat the connection as unusable rather than trying to
                # resynchronize inside attacker-controlled bytes.
                output.put(_InputItem(_InputKind.EOF))
                return
            output.put(_InputItem(_InputKind.LINE, line[:-1]))
    except (OSError, ValueError):
        output.put(_InputItem(_InputKind.ERROR))


def _write_all(stream: BinaryIO, wire: bytes) -> None:
    view = memoryview(wire)
    offset = 0
    while offset < len(view):
        written = stream.write(view[offset:])
        if written is None or written <= 0:
            raise OSError("protocol output write did not make progress")
        offset += written
    stream.flush()


def _write_bounded(stream: BinaryIO, wire: bytes, deadline_seconds: float) -> bool:
    """Write on an isolated daemon and never wait beyond the connection deadline."""

    completed = threading.Event()
    succeeded = False

    def write() -> None:
        nonlocal succeeded
        try:
            _write_all(stream, wire)
            succeeded = True
        except (OSError, ValueError):
            pass
        finally:
            completed.set()

    threading.Thread(
        target=write,
        name="ttc-remote-reply-writer",
        daemon=True,
    ).start()
    return completed.wait(deadline_seconds) and succeeded


class RemoteSpoolStreamer:
    """Persistently stream ready events and durably apply matching ACKs."""

    def __init__(
        self,
        spool: ReplySpool,
        *,
        poll_interval_seconds: float = 0.1,
        shutdown_timeout_seconds: float = 1.0,
        ack_deadline_seconds: float = 30.0,
        write_deadline_seconds: float = 2.0,
        on_diagnostic: Callable[[RemoteStreamDiagnostic], None] | None = None,
    ) -> None:
        if (
            not math.isfinite(poll_interval_seconds)
            or poll_interval_seconds <= 0
            or poll_interval_seconds > 10
        ):
            raise ValueError("poll interval must be in (0, 10] seconds")
        if (
            not math.isfinite(shutdown_timeout_seconds)
            or shutdown_timeout_seconds < 0
            or shutdown_timeout_seconds > 10
        ):
            raise ValueError("shutdown timeout must be in [0, 10] seconds")
        for name, value in (
            ("ACK deadline", ack_deadline_seconds),
            ("write deadline", write_deadline_seconds),
        ):
            if not math.isfinite(value) or value <= 0 or value > 300:
                raise ValueError(f"{name} must be in (0, 300] seconds")
        self._spool = spool
        self._poll_interval_seconds = poll_interval_seconds
        self._shutdown_timeout_seconds = shutdown_timeout_seconds
        self._ack_deadline_seconds = ack_deadline_seconds
        self._write_deadline_seconds = write_deadline_seconds
        self._on_diagnostic = on_diagnostic

    def _emit(self, code: RemoteStreamCode, *, count: int | None = None) -> None:
        if self._on_diagnostic is None:
            return
        try:
            self._on_diagnostic(RemoteStreamDiagnostic(code, count))
        except Exception:
            # Observability must not control the durable delivery boundary.
            pass

    def run(
        self,
        input_stream: BinaryIO,
        output_stream: BinaryIO,
        *,
        stop_event: threading.Event | None = None,
    ) -> RemoteStreamResult:
        stop = stop_event or threading.Event()
        input_items: Queue[_InputItem] = Queue()
        reader = threading.Thread(
            target=_read_ack_input,
            args=(input_stream, input_items),
            name="ttc-remote-reply-ack-reader",
            daemon=True,
        )
        reader.start()

        events_emitted = 0
        acknowledgements_committed = 0
        acknowledgements_rejected = 0
        protocol_errors = 0
        input_eof = False
        boundary_replacement_required = False
        self._emit(RemoteStreamCode.STARTED)

        try:
            while not stop.is_set() and not input_eof:
                try:
                    pending = self._spool.pending()
                except (OSError, ReplySpoolError):
                    boundary_replacement_required = True
                    self._emit(RemoteStreamCode.SPOOL_ERROR)
                    break

                if not pending:
                    try:
                        idle_item = input_items.get(
                            timeout=self._poll_interval_seconds
                        )
                    except Empty:
                        continue
                    if idle_item.kind is _InputKind.EOF:
                        input_eof = True
                        self._emit(RemoteStreamCode.INPUT_EOF)
                        break
                    if idle_item.kind is _InputKind.ERROR:
                        input_eof = True
                        boundary_replacement_required = True
                        self._emit(RemoteStreamCode.INPUT_ERROR)
                        break
                    # An ACK without an in-flight event is connection-fatal.
                    if idle_item.kind in (_InputKind.LINE, _InputKind.PARTIAL):
                        protocol_errors += 1
                        boundary_replacement_required = True
                        self._emit(RemoteStreamCode.ACK_INVALID, count=protocol_errors)
                        break
                    continue

                # The protocol deliberately permits exactly one event in flight.
                # Its matching durable ACK must arrive before another frame is emitted.
                record = pending[0]
                if not _write_bounded(
                    output_stream,
                    record.wire_bytes + b"\n",
                    self._write_deadline_seconds,
                ):
                    boundary_replacement_required = True
                    self._emit(RemoteStreamCode.OUTPUT_ERROR)
                    break
                events_emitted += 1
                self._emit(RemoteStreamCode.EVENT_EMITTED, count=events_emitted)

                ack_expires_at = time.monotonic() + self._ack_deadline_seconds
                item: _InputItem | None = None
                while not stop.is_set():
                    remaining = ack_expires_at - time.monotonic()
                    if remaining <= 0:
                        break
                    try:
                        item = input_items.get(
                            timeout=min(self._poll_interval_seconds, 0.05, remaining)
                        )
                        break
                    except Empty:
                        continue

                if item is None:
                    if not stop.is_set():
                        boundary_replacement_required = True
                        self._emit(RemoteStreamCode.ACK_TIMEOUT)
                    break
                if item.kind is _InputKind.EOF:
                    input_eof = True
                    self._emit(RemoteStreamCode.INPUT_EOF)
                    break
                if item.kind is _InputKind.ERROR:
                    input_eof = True
                    boundary_replacement_required = True
                    self._emit(RemoteStreamCode.INPUT_ERROR)
                    break
                if item.kind is _InputKind.PARTIAL:
                    protocol_errors += 1
                    boundary_replacement_required = True
                    self._emit(RemoteStreamCode.ACK_INVALID, count=protocol_errors)
                    break

                try:
                    ack = ReplyAck.from_bytes(item.wire)
                except ReplyProtocolError:
                    protocol_errors += 1
                    boundary_replacement_required = True
                    self._emit(RemoteStreamCode.ACK_INVALID, count=protocol_errors)
                    break
                if ack.event_id != record.event.event_id or not hmac.compare_digest(
                    record.event.digest, ack.digest
                ):
                    acknowledgements_rejected += 1
                    boundary_replacement_required = True
                    self._emit(
                        RemoteStreamCode.ACK_REJECTED,
                        count=acknowledgements_rejected,
                    )
                    break
                try:
                    result = self._spool.commit_ack(ack)
                except (OSError, ReplySpoolError):
                    boundary_replacement_required = True
                    self._emit(RemoteStreamCode.SPOOL_ERROR)
                    break
                if result.disposition not in (
                    AckDisposition.COMMITTED,
                    AckDisposition.ALREADY_COMMITTED,
                ):
                    acknowledgements_rejected += 1
                    boundary_replacement_required = True
                    self._emit(
                        RemoteStreamCode.ACK_REJECTED,
                        count=acknowledgements_rejected,
                    )
                    break
                acknowledgements_committed += 1
                self._emit(
                    RemoteStreamCode.ACK_COMMITTED,
                    count=acknowledgements_committed,
                )
        finally:
            self._emit(RemoteStreamCode.STOPPED)

        return self._result(
            events_emitted,
            acknowledgements_committed,
            acknowledgements_rejected,
            protocol_errors,
            input_eof,
            reader,
            boundary_replacement_required,
        )

    def _result(
        self,
        events_emitted: int,
        acknowledgements_committed: int,
        acknowledgements_rejected: int,
        protocol_errors: int,
        input_eof: bool,
        reader: threading.Thread,
        boundary_replacement_required: bool,
    ) -> RemoteStreamResult:
        reader.join(self._shutdown_timeout_seconds)
        reader_stopped = not reader.is_alive()
        return RemoteStreamResult(
            events_emitted=events_emitted,
            acknowledgements_committed=acknowledgements_committed,
            acknowledgements_rejected=acknowledgements_rejected,
            protocol_errors=protocol_errors,
            input_eof=input_eof,
            reader_stopped=reader_stopped,
            boundary_replacement_required=(
                boundary_replacement_required or not reader_stopped
            ),
        )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="talktomeclaude-reply-remote")
    subparsers = parser.add_subparsers(dest="command", required=True)
    stream = subparsers.add_parser("stream")
    stream.add_argument("--spool-root", type=Path, required=True)
    stream.add_argument("--poll-interval", type=float, default=0.1)
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    input_stream: BinaryIO | None = None,
    output_stream: BinaryIO | None = None,
) -> int:
    """Run the remote helper; return a content-safe process status code."""

    arguments = _parser().parse_args(argv)
    if arguments.command != "stream":
        return 2
    source = input_stream if input_stream is not None else sys.stdin.buffer
    destination = output_stream if output_stream is not None else sys.stdout.buffer
    try:
        spool = ReplySpool(arguments.spool_root)
        result = RemoteSpoolStreamer(
            spool,
            poll_interval_seconds=arguments.poll_interval,
        ).run(source, destination)
    except (OSError, ReplySpoolError, ValueError):
        return 2
    return 1 if result.boundary_replacement_required else 0


if __name__ == "__main__":  # pragma: no cover - exercised through ``main`` tests.
    raise SystemExit(main())
