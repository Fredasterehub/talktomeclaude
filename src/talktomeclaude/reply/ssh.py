"""Persistent, replay-safe SSH reply transport.

The remote helper owns enumeration of its durable ``ready/`` spool.  This
module owns only the long-lived duplex SSH process and the commit-before-ACK
boundary on Windows.  A disconnect is never interpreted as delivery: the
remote helper must replay an event until its matching ACK is durably recorded.
"""

from __future__ import annotations

import json
import math
import random
import shlex
import subprocess
import threading
from dataclasses import dataclass, field
from enum import Enum
from queue import Empty, Queue
from typing import BinaryIO, Callable, IO, Mapping, Protocol, Sequence

from talktomeclaude.core.backoff import BackoffPolicy, JitteredBackoff
from talktomeclaude.reply.contracts import MAX_WIRE_BYTES, PROTOCOL_VERSION


DEFAULT_MAX_FRAME_BYTES = MAX_WIRE_BYTES
_HEX_DIGITS = frozenset("0123456789abcdef")


class ProtocolErrorCode(str, Enum):
    PARTIAL_FRAME = "partial_frame"
    PARTIAL_UTF8 = "partial_utf8"
    FRAME_TOO_LARGE = "frame_too_large"
    INVALID_UTF8 = "invalid_utf8"
    INVALID_JSON = "invalid_json"
    NON_CANONICAL_JSON = "non_canonical_json"
    INVALID_ENVELOPE = "invalid_envelope"
    VERSION_MISMATCH = "version_mismatch"
    ACK_MISMATCH = "ack_mismatch"


class TransportStatusCode(str, Enum):
    CONNECTING = "connecting"
    CONNECTED = "connected"
    DISCONNECTED = "disconnected"
    BACKOFF = "backoff"
    PROTOCOL_ERROR = "protocol_error"
    COMMIT_REJECTED = "commit_rejected"
    IO_ERROR = "io_error"
    REAP_FAILED = "reap_failed"
    STOPPED = "stopped"


@dataclass(frozen=True, slots=True)
class TransportStatus:
    """Content-free observable transport state."""

    code: TransportStatusCode
    attempt: int = 0
    delay_seconds: float | None = None
    protocol_error: ProtocolErrorCode | None = None


@dataclass(frozen=True, slots=True)
class IncomingFrame:
    """Validated wire identity plus an opaque, repr-hidden document."""

    event_id: str
    digest: str
    document: Mapping[str, object] = field(repr=False)


class DurableReceiver(Protocol):
    """Duck-typed durable receiver; runtime validates its ACK-shaped result."""

    def receive(self, wire_bytes: bytes) -> object: ...


@dataclass(frozen=True, slots=True)
class SSHConnectionSpec:
    remote: str
    remote_command: tuple[str, ...]
    executable: str = "ssh"
    connect_timeout_seconds: float = 10.0
    server_alive_interval_seconds: int = 15
    server_alive_count_max: int = 2

    def __post_init__(self) -> None:
        if not self.remote or self.remote.startswith("-"):
            raise ValueError("remote must be a non-option SSH target")
        if any(char.isspace() or ord(char) < 32 for char in self.remote):
            raise ValueError("remote must not contain whitespace or control characters")
        if not self.remote_command or any("\x00" in arg for arg in self.remote_command):
            raise ValueError("remote command must contain safe non-NUL arguments")
        if (
            not math.isfinite(self.connect_timeout_seconds)
            or self.connect_timeout_seconds <= 0
        ):
            raise ValueError("connect timeout must be positive")
        if self.server_alive_interval_seconds <= 0 or self.server_alive_count_max <= 0:
            raise ValueError("server-alive values must be positive")

    def argv(self) -> list[str]:
        timeout = max(1, math.ceil(self.connect_timeout_seconds))
        return [
            self.executable,
            "-T",
            "-o",
            "BatchMode=yes",
            "-o",
            f"ConnectTimeout={timeout}",
            "-o",
            f"ServerAliveInterval={self.server_alive_interval_seconds}",
            "-o",
            f"ServerAliveCountMax={self.server_alive_count_max}",
            "--",
            self.remote,
            shlex.join(self.remote_command),
        ]


@dataclass(frozen=True, slots=True)
class TransportResult:
    events_seen: int
    acknowledgements_sent: int
    reconnects: int
    reaped_cleanly: bool
    boundary_replacement_required: bool


class _DuplicateKey(ValueError):
    pass


class _FrameFault(ValueError):
    def __init__(self, code: ProtocolErrorCode) -> None:
        super().__init__(code.value)
        self.code = code


def _unique_object(pairs: Sequence[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateKey
        result[key] = value
    return result


def canonical_json_bytes(document: Mapping[str, object]) -> bytes:
    """Encode protocol JSON deterministically without ASCII-escaping Unicode."""

    return json.dumps(
        document,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


class _NDJSONDecoder:
    def __init__(self, max_frame_bytes: int) -> None:
        if max_frame_bytes <= 0:
            raise ValueError("max frame size must be positive")
        self._max_frame_bytes = max_frame_bytes
        self._buffer = bytearray()

    def feed(self, chunk: bytes) -> list[Mapping[str, object]]:
        self._buffer.extend(chunk)
        if len(self._buffer) > self._max_frame_bytes and b"\n" not in self._buffer:
            raise _FrameFault(ProtocolErrorCode.FRAME_TOO_LARGE)
        documents: list[Mapping[str, object]] = []
        while True:
            newline = self._buffer.find(b"\n")
            if newline < 0:
                break
            line = bytes(self._buffer[:newline])
            del self._buffer[: newline + 1]
            if line.endswith(b"\r"):
                line = line[:-1]
            if len(line) > self._max_frame_bytes:
                raise _FrameFault(ProtocolErrorCode.FRAME_TOO_LARGE)
            documents.append(self._decode_line(line))
        if len(self._buffer) > self._max_frame_bytes:
            raise _FrameFault(ProtocolErrorCode.FRAME_TOO_LARGE)
        return documents

    def finish(self) -> None:
        if not self._buffer:
            return
        try:
            bytes(self._buffer).decode("utf-8", errors="strict")
        except UnicodeDecodeError as exc:
            raise _FrameFault(ProtocolErrorCode.PARTIAL_UTF8) from exc
        raise _FrameFault(ProtocolErrorCode.PARTIAL_FRAME)

    @staticmethod
    def _decode_line(line: bytes) -> Mapping[str, object]:
        try:
            text = line.decode("utf-8", errors="strict")
        except UnicodeDecodeError as exc:
            raise _FrameFault(ProtocolErrorCode.INVALID_UTF8) from exc
        try:
            document = json.loads(text, object_pairs_hook=_unique_object)
        except (json.JSONDecodeError, _DuplicateKey) as exc:
            raise _FrameFault(ProtocolErrorCode.INVALID_JSON) from exc
        if not isinstance(document, dict):
            raise _FrameFault(ProtocolErrorCode.INVALID_ENVELOPE)
        try:
            canonical = canonical_json_bytes(document)
        except (TypeError, ValueError) as exc:
            raise _FrameFault(ProtocolErrorCode.INVALID_JSON) from exc
        if canonical != line:
            raise _FrameFault(ProtocolErrorCode.NON_CANONICAL_JSON)
        return document


_EOF = object()
_READ_FAILED = object()


def _pump(stream: BinaryIO, output: Queue[bytes | object]) -> None:
    try:
        while True:
            chunk = stream.read(64 * 1024)
            if not chunk:
                break
            output.put(chunk)
    except (OSError, ValueError):
        output.put(_READ_FAILED)
    finally:
        output.put(_EOF)


def _drain(stream: BinaryIO) -> None:
    """Drain SSH stderr without retaining or exposing potentially sensitive bytes."""

    try:
        while stream.read(64 * 1024):
            pass
    except (OSError, ValueError):
        pass


def _write_all(stream: IO[bytes], wire: bytes) -> None:
    view = memoryview(wire)
    offset = 0
    while offset < len(view):
        written = stream.write(view[offset:])
        if written is None or written <= 0:
            raise OSError("ACK write did not make progress")
        offset += written
    stream.flush()


def _write_bounded(stream: IO[bytes], wire: bytes, deadline_seconds: float) -> bool:
    """Isolate a potentially stuck pipe write behind a strict connection deadline."""

    completed = threading.Event()
    succeeded = False

    def write() -> None:
        nonlocal succeeded
        try:
            _write_all(stream, wire)
            succeeded = True
        except (BrokenPipeError, OSError, ValueError):
            pass
        finally:
            completed.set()

    threading.Thread(
        target=write,
        name="reply-ssh-ack-writer",
        daemon=True,
    ).start()
    return completed.wait(deadline_seconds) and succeeded


class PersistentSSHReplyTransport:
    """Receive durable remote events and ACK only durable local commits."""

    def __init__(
        self,
        spec: SSHConnectionSpec,
        receiver: DurableReceiver,
        *,
        backoff_policy: BackoffPolicy = BackoffPolicy(),
        random_source: random.Random | None = None,
        max_frame_bytes: int = DEFAULT_MAX_FRAME_BYTES,
        shutdown_deadline_seconds: float = 2.0,
        write_deadline_seconds: float = 2.0,
        popen_factory: Callable[..., subprocess.Popen[bytes]] = subprocess.Popen,
        status: Callable[[TransportStatus], None] | None = None,
        wait_for_stop: Callable[[threading.Event, float], bool] | None = None,
    ) -> None:
        if max_frame_bytes <= 0:
            raise ValueError("max frame size must be positive")
        if not math.isfinite(shutdown_deadline_seconds) or shutdown_deadline_seconds < 0:
            raise ValueError("shutdown deadline must be non-negative")
        if (
            not math.isfinite(write_deadline_seconds)
            or write_deadline_seconds <= 0
            or write_deadline_seconds > 300
        ):
            raise ValueError("write deadline must be in (0, 300] seconds")
        self._spec = spec
        self._receiver = receiver
        self._backoff = JitteredBackoff(backoff_policy, random_source or random.Random())
        self._max_frame_bytes = max_frame_bytes
        self._shutdown_deadline = shutdown_deadline_seconds
        self._write_deadline = write_deadline_seconds
        self._popen = popen_factory
        self._status_callback = status or (lambda _status: None)
        self._wait_for_stop = wait_for_stop or (lambda event, delay: event.wait(delay))

    def run(self, stop: threading.Event) -> TransportResult:
        attempt = 0
        events_seen = 0
        acknowledgements_sent = 0
        reconnects = 0
        reaped_cleanly = True

        while not stop.is_set():
            self._emit(TransportStatusCode.CONNECTING, attempt=attempt)
            process: subprocess.Popen[bytes] | None = None
            write_tainted = False
            try:
                process = self._popen(
                    self._spec.argv(),
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    bufsize=0,
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                )
                if process.stdin is None or process.stdout is None or process.stderr is None:
                    raise OSError("SSH process pipes unavailable")
                self._emit(TransportStatusCode.CONNECTED, attempt=attempt)
                seen, sent, write_tainted = self._serve_connection(process, stop)
                events_seen += seen
                acknowledgements_sent += sent
                if sent:
                    attempt = 0
            except OSError:
                self._emit(TransportStatusCode.IO_ERROR, attempt=attempt)
            finally:
                if process is not None:
                    clean = self._reap(
                        process,
                        close_stdin=not write_tainted,
                    )
                    reaped_cleanly = reaped_cleanly and clean
                    if not clean:
                        self._emit(TransportStatusCode.REAP_FAILED, attempt=attempt)

            # Never multiply an unreaped process boundary with a reconnect.
            # The owner must replace or externally reap this transport.
            if not reaped_cleanly:
                break
            if stop.is_set():
                break
            reconnects += 1
            self._emit(TransportStatusCode.DISCONNECTED, attempt=attempt)
            delay = self._backoff.delay(attempt)
            self._emit(
                TransportStatusCode.BACKOFF,
                attempt=attempt,
                delay_seconds=delay,
            )
            attempt += 1
            if self._wait_for_stop(stop, delay):
                break

        self._emit(TransportStatusCode.STOPPED)
        return TransportResult(
            events_seen=events_seen,
            acknowledgements_sent=acknowledgements_sent,
            reconnects=reconnects,
            reaped_cleanly=reaped_cleanly,
            boundary_replacement_required=not reaped_cleanly,
        )

    def _serve_connection(
        self,
        process: subprocess.Popen[bytes],
        stop: threading.Event,
    ) -> tuple[int, int, bool]:
        assert process.stdin is not None
        assert process.stdout is not None
        assert process.stderr is not None
        output: Queue[bytes | object] = Queue()
        stdout_thread = threading.Thread(
            target=_pump,
            args=(process.stdout, output),
            name="reply-ssh-stdout",
            daemon=True,
        )
        stderr_thread = threading.Thread(
            target=_drain,
            args=(process.stderr,),
            name="reply-ssh-stderr",
            daemon=True,
        )
        stdout_thread.start()
        stderr_thread.start()
        decoder = _NDJSONDecoder(self._max_frame_bytes)
        seen = 0
        sent = 0

        while not stop.is_set():
            try:
                item = output.get(timeout=0.05)
            except Empty:
                if process.poll() is not None and not stdout_thread.is_alive():
                    break
                continue
            if item is _READ_FAILED:
                self._emit(TransportStatusCode.IO_ERROR)
                return seen, sent, False
            if item is _EOF:
                try:
                    decoder.finish()
                except _FrameFault as fault:
                    self._emit(
                        TransportStatusCode.PROTOCOL_ERROR,
                        protocol_error=fault.code,
                    )
                return seen, sent, False
            assert isinstance(item, bytes)
            try:
                documents = decoder.feed(item)
            except _FrameFault as fault:
                self._emit(
                    TransportStatusCode.PROTOCOL_ERROR,
                    protocol_error=fault.code,
                )
                return seen, sent, False
            for document in documents:
                try:
                    frame = self._validate_event(document)
                except _FrameFault as fault:
                    self._emit(
                        TransportStatusCode.PROTOCOL_ERROR,
                        protocol_error=fault.code,
                    )
                    return seen, sent, False
                seen += 1
                try:
                    # Re-encoding is byte-identical because non-canonical input
                    # has already been rejected.  The local receiver validates
                    # payload digest and persists its canonical/dedupe record.
                    result = self._receiver.receive(canonical_json_bytes(frame.document))
                except Exception:
                    self._emit(TransportStatusCode.COMMIT_REJECTED)
                    return seen, sent, False
                try:
                    ack_result = getattr(result, "ack")
                except (AttributeError, TypeError):
                    self._emit(TransportStatusCode.COMMIT_REJECTED)
                    return seen, sent, False
                if ack_result is None:
                    self._emit(TransportStatusCode.COMMIT_REJECTED)
                    return seen, sent, False
                if ack_result.event_id != frame.event_id or ack_result.digest != frame.digest:
                    self._emit(
                        TransportStatusCode.PROTOCOL_ERROR,
                        protocol_error=ProtocolErrorCode.ACK_MISMATCH,
                    )
                    return seen, sent, False
                ack = canonical_json_bytes(
                    {
                        "digest": frame.digest,
                        "event_id": frame.event_id,
                        "version": PROTOCOL_VERSION,
                    }
                ) + b"\n"
                if not _write_bounded(process.stdin, ack, self._write_deadline):
                    self._emit(TransportStatusCode.IO_ERROR)
                    return seen, sent, True
                sent += 1
        return seen, sent, False

    @staticmethod
    def _validate_event(document: Mapping[str, object]) -> IncomingFrame:
        version = document.get("version")
        if isinstance(version, bool) or not isinstance(version, int):
            raise _FrameFault(ProtocolErrorCode.INVALID_ENVELOPE)
        if version != PROTOCOL_VERSION:
            raise _FrameFault(ProtocolErrorCode.VERSION_MISMATCH)
        event_id = document.get("event_id")
        digest = document.get("digest")
        if not isinstance(event_id, str) or not event_id or len(event_id) > 128:
            raise _FrameFault(ProtocolErrorCode.INVALID_ENVELOPE)
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(char not in _HEX_DIGITS for char in digest)
        ):
            raise _FrameFault(ProtocolErrorCode.INVALID_ENVELOPE)
        return IncomingFrame(event_id=event_id, digest=digest, document=document)

    def _reap(
        self,
        process: subprocess.Popen[bytes],
        *,
        close_stdin: bool = True,
    ) -> bool:
        if close_stdin and process.stdin is not None:
            try:
                process.stdin.close()
            except OSError:
                pass
        clean = True
        if process.poll() is not None:
            try:
                process.wait(timeout=0)
            except (OSError, subprocess.TimeoutExpired):
                clean = False
        else:
            try:
                process.terminate()
            except OSError:
                pass
            try:
                process.wait(timeout=self._shutdown_deadline)
            except subprocess.TimeoutExpired:
                try:
                    process.kill()
                except OSError:
                    pass
                try:
                    process.wait(timeout=self._shutdown_deadline)
                except (OSError, subprocess.TimeoutExpired):
                    clean = False
            except OSError:
                clean = False
        # Closing read pipes only after process termination avoids an unbounded
        # cross-thread close waiting on a blocked Windows pipe read.  If even
        # kill/reap timed out, leave the daemon pumps isolated and report the
        # boundary tainted instead of risking an unbounded close or reconnect.
        if clean or process.poll() is not None:
            for stream in (process.stdout, process.stderr):
                if stream is not None:
                    try:
                        stream.close()
                    except OSError:
                        pass
        return clean

    def _emit(
        self,
        code: TransportStatusCode,
        *,
        attempt: int = 0,
        delay_seconds: float | None = None,
        protocol_error: ProtocolErrorCode | None = None,
    ) -> None:
        try:
            self._status_callback(
                TransportStatus(
                    code=code,
                    attempt=attempt,
                    delay_seconds=delay_seconds,
                    protocol_error=protocol_error,
                )
            )
        except Exception:
            # Status observers are non-authoritative and content-free.
            pass
