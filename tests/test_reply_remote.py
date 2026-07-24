from __future__ import annotations

import io
import tempfile
import threading
import time
import unittest
from pathlib import Path
from typing import BinaryIO, cast

from talktomeclaude.reply import ReplyAck, ReplyEvent, ReplySpool
from talktomeclaude.reply.remote import (
    RemoteSpoolStreamer,
    RemoteStreamCode,
    RemoteStreamDiagnostic,
    RemoteStreamResult,
    main,
)


class RemoteSpoolStreamerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name) / "explicit-remote-spool"

    def _spool(self) -> ReplySpool:
        return ReplySpool(self.root)

    @staticmethod
    def _ack_input(*events: ReplyEvent) -> io.BytesIO:
        return io.BytesIO(
            b"".join(ReplyAck.for_event(event).to_bytes() + b"\n" for event in events)
        )

    def _run(
        self, spool: ReplySpool, source: bytes
    ) -> tuple[RemoteStreamResult, tuple[ReplyEvent, ...]]:
        output = io.BytesIO()
        result = RemoteSpoolStreamer(
            spool,
            poll_interval_seconds=0.001,
            shutdown_timeout_seconds=0.1,
            ack_deadline_seconds=0.1,
            write_deadline_seconds=0.1,
        ).run(io.BytesIO(source), output)
        events = tuple(
            ReplyEvent.from_bytes(line) for line in output.getvalue().splitlines()
        )
        return result, events

    def test_streams_unicode_as_canonical_ndjson_and_commits_exact_ack(self) -> None:
        event = ReplyEvent.create(
            session="unicode-session",
            event_id="unicode-event",
            answer="café 🙂 שלום é\r\nnext",
        )
        spool = self._spool()
        record = spool.enqueue(event)
        output = io.BytesIO()

        result = RemoteSpoolStreamer(spool, poll_interval_seconds=0.001).run(
            self._ack_input(event), output
        )

        self.assertEqual(output.getvalue(), record.wire_bytes + b"\n")
        self.assertEqual(1, result.events_emitted)
        self.assertEqual(1, result.acknowledgements_committed)
        self.assertEqual((), ReplySpool(self.root).pending())

    def test_bad_and_partial_ack_frames_never_clean_up_ready_event(self) -> None:
        event = ReplyEvent.create(
            session="session", event_id="event-bad-ack", answer="private answer"
        )
        spool = self._spool()
        spool.enqueue(event)
        invalid = b'{"not":"an ack"}\n'
        partial = ReplyAck.for_event(event).to_bytes()[:-3]

        result, streamed = self._run(spool, invalid + partial)

        self.assertEqual((event,), streamed)
        self.assertEqual(1, result.protocol_errors)
        self.assertTrue(result.boundary_replacement_required)
        self.assertEqual(0, result.acknowledgements_committed)
        self.assertEqual((event,), tuple(r.event for r in ReplySpool(self.root).pending()))
        self.assertNotIn("private answer", repr(result))

    def test_ack_identity_mismatch_is_rejected_without_cleanup(self) -> None:
        event = ReplyEvent.create(
            session="session", event_id="event-expected", answer="not diagnostic"
        )
        spool = self._spool()
        spool.enqueue(event)
        wrong_id = ReplyAck(event.version, "event-other", event.digest)
        wrong_digest = ReplyAck(event.version, event.event_id, "0" * 64)
        source = wrong_id.to_bytes() + b"\n" + wrong_digest.to_bytes() + b"\n"

        result, _streamed = self._run(spool, source)

        self.assertEqual(1, result.acknowledgements_rejected)
        self.assertTrue(result.boundary_replacement_required)
        self.assertEqual(0, result.acknowledgements_committed)
        self.assertEqual(1, len(ReplySpool(self.root).pending()))

    def test_disconnect_before_ack_replays_on_restart_then_commits(self) -> None:
        event = ReplyEvent.create(
            session="session", event_id="event-replay", answer="replay me"
        )
        self._spool().enqueue(event)

        first, first_events = self._run(ReplySpool(self.root), b"")
        second, second_events = self._run(
            ReplySpool(self.root), ReplyAck.for_event(event).to_bytes() + b"\n"
        )

        self.assertTrue(first.input_eof)
        self.assertEqual((event,), first_events)
        self.assertEqual((event,), second_events)
        self.assertEqual(1, second.acknowledgements_committed)
        self.assertEqual((), ReplySpool(self.root).pending())

    def test_large_backlog_is_strictly_emit_then_matching_ack_ordered(self) -> None:
        spool = self._spool()
        for index in reversed(range(250)):
            spool.enqueue(
                ReplyEvent.create(
                    session="session",
                    event_id=f"event-{index:04d}",
                    answer=f"private-{index}",
                )
            )
        expected = tuple(record.event for record in spool.pending())
        source = b"".join(
            ReplyAck.for_event(event).to_bytes() + b"\n" for event in expected
        )

        result, streamed = self._run(spool, source)

        self.assertEqual(expected, streamed)
        self.assertEqual(250, result.acknowledgements_committed)
        self.assertEqual((), ReplySpool(self.root).pending())

    def test_lost_ack_deadline_leaves_ready_event_for_replay(self) -> None:
        class BlockingInput:
            def readline(self, size: int | None = -1) -> bytes:
                del size
                threading.Event().wait(5)
                return b""

        event = ReplyEvent.create(
            session="session", event_id="event-lost-ack", answer="replay safely"
        )
        spool = self._spool()
        spool.enqueue(event)
        output = io.BytesIO()
        started = time.monotonic()
        first = RemoteSpoolStreamer(
            spool,
            poll_interval_seconds=0.001,
            shutdown_timeout_seconds=0.001,
            ack_deadline_seconds=0.01,
        ).run(cast(BinaryIO, BlockingInput()), output)

        self.assertLess(time.monotonic() - started, 0.25)
        self.assertTrue(first.boundary_replacement_required)
        self.assertEqual(event.to_bytes() + b"\n", output.getvalue())
        self.assertEqual(1, len(spool.pending()))

        second, replayed = self._run(spool, ReplyAck.for_event(event).to_bytes() + b"\n")
        self.assertEqual((event,), replayed)
        self.assertEqual(1, second.acknowledgements_committed)
        self.assertEqual((), spool.pending())

    def test_invalid_ack_is_fatal_and_next_connection_replays(self) -> None:
        event = ReplyEvent.create(
            session="session", event_id="event-invalid-replay", answer="still durable"
        )
        spool = self._spool()
        spool.enqueue(event)

        first, first_streamed = self._run(spool, b'{"invalid":true}\n')
        second, second_streamed = self._run(
            spool, ReplyAck.for_event(event).to_bytes() + b"\n"
        )

        self.assertEqual((event,), first_streamed)
        self.assertTrue(first.boundary_replacement_required)
        self.assertEqual((event,), second_streamed)
        self.assertEqual(1, second.acknowledgements_committed)
        self.assertEqual((), spool.pending())

    def test_permanently_blocking_output_is_bounded_and_taints_connection(self) -> None:
        class BlockingOutput:
            def write(self, data: bytes) -> int:
                del data
                threading.Event().wait(5)
                return 0

            def flush(self) -> None:
                pass

        event = ReplyEvent.create(
            session="session", event_id="event-blocked-output", answer="private"
        )
        spool = self._spool()
        spool.enqueue(event)
        started = time.monotonic()
        result = RemoteSpoolStreamer(
            spool,
            poll_interval_seconds=0.001,
            write_deadline_seconds=0.01,
        ).run(self._ack_input(event), cast(BinaryIO, BlockingOutput()))

        self.assertLess(time.monotonic() - started, 0.25)
        self.assertTrue(result.boundary_replacement_required)
        self.assertEqual(0, result.events_emitted)
        self.assertEqual(1, len(spool.pending()))

    def test_temp_files_are_not_protocol_events(self) -> None:
        spool = self._spool()
        temporary = spool.ready / ".unfinished.tmp"
        temporary.write_bytes(b"must never stream")

        result, streamed = self._run(spool, b"")

        self.assertEqual((), streamed)
        self.assertEqual(0, result.events_emitted)
        self.assertTrue(temporary.exists())

        retention = spool.apply_retention(max_age_seconds=0, max_count=0)
        self.assertEqual(1, retention.stale_temps_removed)
        self.assertFalse(temporary.exists())

    def test_polls_for_new_ready_event_and_shutdown_is_bounded(self) -> None:
        class BlockingInput:
            def readline(self, size: int | None = -1) -> bytes:
                del size
                threading.Event().wait(5)
                return b""

        stop = threading.Event()
        output = io.BytesIO()
        spool = self._spool()
        streamer = RemoteSpoolStreamer(
            spool, poll_interval_seconds=0.005, shutdown_timeout_seconds=0.01
        )
        result_holder: list[object] = []
        runner = threading.Thread(
            target=lambda: result_holder.append(
                streamer.run(
                    cast(BinaryIO, BlockingInput()), output, stop_event=stop
                )
            )
        )
        runner.start()
        event = ReplyEvent.create(
            session="session", event_id="event-later", answer="later"
        )
        spool.enqueue(event)
        deadline = time.monotonic() + 1
        while not output.getvalue() and time.monotonic() < deadline:
            time.sleep(0.005)
        started = time.monotonic()
        stop.set()
        runner.join(0.5)

        self.assertFalse(runner.is_alive())
        self.assertLess(time.monotonic() - started, 0.5)
        self.assertEqual(event.to_bytes() + b"\n", output.getvalue())
        result = result_holder[0]
        self.assertFalse(result.reader_stopped)  # type: ignore[attr-defined]
        self.assertTrue(result.boundary_replacement_required)  # type: ignore[attr-defined]
        self.assertEqual(1, len(ReplySpool(self.root).pending()))

    def test_diagnostics_are_content_free_even_when_observer_fails(self) -> None:
        event = ReplyEvent.create(
            session="session", event_id="event-private", answer="SECRET-ANSWER"
        )
        spool = self._spool()
        spool.enqueue(event)
        diagnostics: list[RemoteStreamDiagnostic] = []

        def observe(item: RemoteStreamDiagnostic) -> None:
            diagnostics.append(item)
            if item.code is RemoteStreamCode.EVENT_EMITTED:
                raise RuntimeError("observer failure")

        output = io.BytesIO()
        result = RemoteSpoolStreamer(
            spool, poll_interval_seconds=0.001, on_diagnostic=observe
        ).run(self._ack_input(event), output)

        self.assertEqual(1, result.acknowledgements_committed)
        self.assertNotIn("SECRET-ANSWER", repr(diagnostics))
        self.assertNotIn(event.event_id, repr(diagnostics))
        self.assertNotIn(event.digest, repr(diagnostics))

    def test_cli_main_uses_only_explicit_spool_root(self) -> None:
        event = ReplyEvent.create(
            session="session", event_id="event-cli", answer="cli unicode 🙂"
        )
        self._spool().enqueue(event)
        output = io.BytesIO()

        status = main(
            ["stream", "--spool-root", str(self.root), "--poll-interval", "0.001"],
            input_stream=self._ack_input(event),
            output_stream=output,
        )

        self.assertEqual(0, status)
        self.assertEqual(event, ReplyEvent.from_bytes(output.getvalue().rstrip(b"\n")))
        self.assertEqual((), ReplySpool(self.root).pending())


if __name__ == "__main__":
    unittest.main()
