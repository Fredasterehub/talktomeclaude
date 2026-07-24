"""Crash/replay matrix for the durable file-per-event reply protocol."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

from talktomeclaude.reply import (
    AckDisposition,
    DiagnosticCode,
    ReceiveDisposition,
    ReplyAck,
    ReplyDiagnostic,
    ReplyEvent,
    ReplyProtocolError,
    ReplyReceiver,
    ReplySpool,
    SpoolConflictError,
    SpoolFullError,
)


def _event(
    event_id: str = "event-001",
    *,
    session: str = "session-001",
    answer: str = "Unicode: café 🙂 مرحبا\r\nnext",
) -> ReplyEvent:
    return ReplyEvent.create(session=session, event_id=event_id, answer=answer)


class ReplyContractTests(unittest.TestCase):
    def test_event_is_exact_canonical_utf8_v1_and_content_private(self) -> None:
        event = _event()
        wire = event.to_bytes()

        self.assertEqual(
            ["answer", "digest", "event_id", "session", "version"],
            sorted(json.loads(wire)),
        )
        changed_session = _event(session="other-session")
        self.assertNotEqual(changed_session.digest, event.digest)
        self.assertEqual(event, ReplyEvent.from_bytes(wire))
        self.assertNotIn(event.answer, repr(event))
        self.assertEqual(wire, ReplyEvent.from_bytes(wire).to_bytes())

    def test_noncanonical_unknown_version_and_unsafe_identity_are_rejected(self) -> None:
        event = _event()
        expanded = json.dumps(json.loads(event.to_bytes()), ensure_ascii=False).encode()
        with self.assertRaises(ReplyProtocolError):
            ReplyEvent.from_bytes(expanded)
        with self.assertRaises(ReplyProtocolError):
            ReplyEvent.create(session="s", event_id="../escape", answer="answer")
        with self.assertRaises(ReplyProtocolError):
            ReplyEvent(True, "session", "event", "answer", "0" * 64)
        value = json.loads(event.to_bytes())
        value["version"] = 2
        with self.assertRaises(ReplyProtocolError):
            ReplyEvent.from_bytes(
                json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
            )

    def test_ack_contract_is_exact_and_canonical(self) -> None:
        ack = ReplyAck.for_event(_event())
        self.assertEqual(ack, ReplyAck.from_bytes(ack.to_bytes()))
        self.assertEqual(["digest", "event_id", "version"], sorted(json.loads(ack.to_bytes())))


class ReplySpoolTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_capabilities_report_actual_separate_guarantees(self) -> None:
        spool = ReplySpool(self.root / "spool")
        self.assertTrue(spool.capabilities.file_fsync)
        self.assertTrue(spool.capabilities.atomic_rename)
        self.assertIsInstance(spool.capabilities.directory_fsync, bool)
        if spool.capabilities.directory_fsync:
            self.assertEqual("file_and_directory_durable", spool.capabilities.guarantee)
        else:
            self.assertIn("directory_sync_unavailable", spool.capabilities.guarantee)

    def test_filename_is_event_identity_only_and_temp_never_enumerates(self) -> None:
        spool = ReplySpool(self.root / "spool")
        event = _event()
        record = spool.enqueue(event)
        (spool.ready / ".event-ignored.partial.tmp").write_bytes(b"partial")

        self.assertEqual("event-001.json", record.path.name)
        self.assertNotIn("Unicode", record.path.name)
        self.assertEqual((event.event_id,), spool.recover_cursor().ready_event_ids)
        self.assertEqual((event,), tuple(item.event for item in spool.pending()))

    def test_writer_crash_before_close_and_after_fsync_has_no_visible_event(self) -> None:
        for phase in ("before_temp_flush", "after_file_fsync_before_ready_rename"):
            with self.subTest(phase=phase):
                case = self.root / phase

                def crash(name: str) -> None:
                    if name == phase:
                        raise RuntimeError("synthetic crash")

                spool = ReplySpool(case, phase_hook=crash)
                with self.assertRaisesRegex(RuntimeError, "synthetic crash"):
                    spool.enqueue(_event())
                self.assertEqual((), ReplySpool(case).pending())

    def test_crash_after_ready_rename_replays_on_restart(self) -> None:
        def crash(name: str) -> None:
            if name == "after_ready_rename_before_directory_fsync":
                raise RuntimeError("synthetic crash")

        spool = ReplySpool(self.root / "spool", phase_hook=crash)
        with self.assertRaises(RuntimeError):
            spool.enqueue(_event())
        replay = ReplySpool(self.root / "spool").pending()
        self.assertEqual(1, len(replay))
        self.assertEqual("event-001", replay[0].event.event_id)

    def test_deterministic_enumeration_but_receiving_reverse_order_is_correct(self) -> None:
        spool = ReplySpool(self.root / "spool")
        for identity in ("event-c", "event-a", "event-b"):
            spool.enqueue(_event(identity))
        records = spool.pending()
        self.assertEqual(["event-a", "event-b", "event-c"], [r.event.event_id for r in records])

        receiver = ReplyReceiver(self.root / "receiver")
        for record in reversed(records):
            result = receiver.receive(record.wire_bytes)
            self.assertTrue(result.apply)
            self.assertIsNotNone(result.ack)
        self.assertEqual(3, len(tuple((receiver.canonical).glob("*.json"))))

    def test_enqueue_is_idempotent_but_same_id_different_payload_conflicts(self) -> None:
        spool = ReplySpool(self.root / "spool")
        original = _event()
        self.assertEqual(original, spool.enqueue(original).event)
        self.assertEqual(original, spool.enqueue(original).event)
        with self.assertRaises(SpoolConflictError):
            spool.enqueue(_event(answer="different"))
        self.assertEqual(original, spool.pending()[0].event)

    def test_ack_requires_exact_id_and_digest_then_is_durable_and_idempotent(self) -> None:
        spool = ReplySpool(self.root / "spool")
        event = _event()
        spool.enqueue(event)
        mismatch = ReplyAck(1, event.event_id, "0" * 64)
        rejected = spool.commit_ack(mismatch)
        self.assertEqual(AckDisposition.REJECTED, rejected.disposition)
        self.assertEqual(1, len(spool.pending()))

        ack = ReplyAck.for_event(event)
        committed = spool.commit_ack(ack)
        self.assertEqual(AckDisposition.COMMITTED, committed.disposition)
        self.assertEqual((), spool.pending())
        duplicate = ReplySpool(self.root / "spool").commit_ack(ack)
        self.assertEqual(AckDisposition.ALREADY_COMMITTED, duplicate.disposition)

    def test_crash_after_remote_ack_commit_recovers_without_ready_replay(self) -> None:
        target = "after_remote_ack_commit_before_cleanup"

        def crash(name: str) -> None:
            if name == target:
                raise RuntimeError("synthetic crash")

        event = _event()
        spool = ReplySpool(self.root / "spool")
        spool.enqueue(event)
        crashing = ReplySpool(self.root / "spool", phase_hook=crash)
        with self.assertRaises(RuntimeError):
            crashing.commit_ack(ReplyAck.for_event(event))
        restarted = ReplySpool(self.root / "spool")
        self.assertEqual((), restarted.pending())
        self.assertEqual(
            AckDisposition.ALREADY_COMMITTED,
            restarted.commit_ack(ReplyAck.for_event(event)).disposition,
        )

    def test_spool_full_is_content_free_and_ready_events_are_untouched(self) -> None:
        diagnostics: list[ReplyDiagnostic] = []
        spool = ReplySpool(
            self.root / "spool",
            max_ready_events=1,
            on_diagnostic=diagnostics.append,
        )
        first = _event("event-one", answer="SECRET-ONE")
        spool.enqueue(first)
        with self.assertRaises(SpoolFullError) as raised:
            spool.enqueue(_event("event-two", answer="SECRET-TWO"))
        self.assertEqual(DiagnosticCode.SPOOL_FULL, raised.exception.code)
        self.assertNotIn("SECRET", repr(raised.exception))
        self.assertEqual((first,), tuple(record.event for record in spool.pending()))
        self.assertIn(DiagnosticCode.SPOOL_FULL, [item.code for item in diagnostics])
        self.assertNotIn("SECRET", repr(diagnostics))

    def test_corrupt_ready_is_quarantined_without_content_diagnostic(self) -> None:
        spool = ReplySpool(self.root / "spool", id_factory=lambda: "opaque")
        (spool.ready / "event-bad.json").write_bytes(b"not json SECRET")
        self.assertEqual((), spool.pending())
        files = tuple(spool.quarantine.iterdir())
        self.assertEqual(1, len(files))
        self.assertNotIn("SECRET", files[0].name)

    def test_retention_removes_only_acked_and_quarantine_eligible_files(self) -> None:
        spool = ReplySpool(self.root / "spool", id_factory=lambda: "opaque")
        ready = _event("event-ready")
        acknowledged = _event("event-acked")
        spool.enqueue(ready)
        spool.enqueue(acknowledged)
        spool.commit_ack(ReplyAck.for_event(acknowledged))
        (spool.ready / "event-corrupt.json").write_bytes(b"broken")
        spool.pending()  # moves only the corrupt artifact to quarantine
        stale_temp = spool.ready / ".event-crashed.random.tmp"
        stale_temp.write_bytes(b"partial")
        old = time.time() - 10_000
        for directory in (spool.ready, spool.acked, spool.quarantine):
            for path in directory.iterdir():
                os.utime(path, (old, old))

        result = spool.apply_retention(max_age_seconds=1, max_count=0)
        self.assertEqual(1, result.acked_removed)
        self.assertEqual(1, result.quarantine_removed)
        self.assertEqual(1, result.stale_temps_removed)
        self.assertEqual(1, result.ready_preserved)
        self.assertFalse(stale_temp.exists())
        self.assertEqual((ready,), tuple(record.event for record in spool.pending()))

    def test_throwing_spool_observer_never_changes_replay_or_ack_commit(self) -> None:
        spool = ReplySpool(
            self.root / "spool",
            on_diagnostic=lambda _item: (_ for _ in ()).throw(
                RuntimeError("observer")
            ),
        )
        event = _event("event-observer")
        spool.enqueue(event)

        self.assertEqual((event,), tuple(record.event for record in spool.pending()))
        result = spool.commit_ack(ReplyAck.for_event(event))

        self.assertEqual(AckDisposition.COMMITTED, result.disposition)
        self.assertEqual((), spool.pending())


class ReplyReceiverTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_commit_before_ack_and_same_digest_replay_has_one_effect(self) -> None:
        receiver = ReplyReceiver(self.root / "receiver")
        event = _event()
        first = receiver.receive(event.to_bytes())
        self.assertEqual(ReceiveDisposition.COMMITTED, first.disposition)
        self.assertTrue(first.apply)
        self.assertEqual(ReplyAck.for_event(event), first.ack)
        self.assertTrue((receiver.canonical / "event-001.json").exists())
        self.assertTrue((receiver.dedupe / "event-001.json").exists())

        duplicate = ReplyReceiver(self.root / "receiver").receive(event.to_bytes())
        self.assertEqual(ReceiveDisposition.DUPLICATE, duplicate.disposition)
        self.assertFalse(duplicate.apply)
        self.assertEqual(first.ack, duplicate.ack)

    def test_throwing_receiver_observer_cannot_block_commit_or_ack(self) -> None:
        receiver = ReplyReceiver(
            self.root / "receiver",
            on_diagnostic=lambda _item: (_ for _ in ()).throw(
                RuntimeError("observer")
            ),
        )
        event = _event("event-observer")

        first = receiver.receive(event.to_bytes())
        duplicate = receiver.receive(event.to_bytes())

        self.assertEqual(ReceiveDisposition.COMMITTED, first.disposition)
        self.assertEqual(ReplyAck.for_event(event), first.ack)
        self.assertEqual(ReceiveDisposition.DUPLICATE, duplicate.disposition)
        self.assertEqual(first.ack, duplicate.ack)

    def test_pending_committed_is_deterministic_and_skips_exact_consumption(self) -> None:
        receiver = ReplyReceiver(self.root / "receiver")
        events = [_event(identity) for identity in ("event-c", "event-a", "event-b")]
        for event in events:
            receiver.receive(event.to_bytes())
        by_id = {event.event_id: event for event in events}
        self.assertEqual(
            ["event-a", "event-b", "event-c"],
            [event.event_id for event in receiver.pending_committed()],
        )
        self.assertTrue(receiver.commit_consumed(ReplyAck.for_event(by_id["event-b"])))
        self.assertEqual(
            ["event-a", "event-c"],
            [event.event_id for event in ReplyReceiver(self.root / "receiver").pending_committed()],
        )

    def test_ack_lost_before_remote_commit_replays_dedupe_then_commits(self) -> None:
        spool = ReplySpool(self.root / "remote")
        receiver = ReplyReceiver(self.root / "local")
        event = _event()
        spool.enqueue(event)

        first = receiver.receive(spool.pending()[0].wire_bytes)
        self.assertTrue(first.apply)
        self.assertIsNotNone(first.ack)
        # Simulate disconnect after the ACK was written but before the remote
        # helper committed it: ready remains the durable replay cursor.
        replay_record = ReplySpool(self.root / "remote").pending()[0]
        replay = ReplyReceiver(self.root / "local").receive(replay_record.wire_bytes)
        self.assertFalse(replay.apply)
        self.assertEqual(first.ack, replay.ack)
        assert replay.ack is not None
        committed = ReplySpool(self.root / "remote").commit_ack(replay.ack)
        self.assertEqual(AckDisposition.COMMITTED, committed.disposition)
        self.assertEqual((), ReplySpool(self.root / "remote").pending())

    def test_crash_after_canonical_before_dedupe_replay_applies_once(self) -> None:
        def crash(name: str) -> None:
            if name == "after_local_canonical_commit":
                raise RuntimeError("synthetic crash")

        event = _event()
        receiver = ReplyReceiver(self.root / "receiver", phase_hook=crash)
        with self.assertRaises(RuntimeError):
            receiver.receive(event.to_bytes())
        self.assertTrue((receiver.canonical / "event-001.json").exists())
        self.assertFalse((receiver.dedupe / "event-001.json").exists())

        replay = ReplyReceiver(self.root / "receiver").receive(event.to_bytes())
        self.assertTrue(replay.apply)
        self.assertEqual(ReplyAck.for_event(event), replay.ack)
        later = ReplyReceiver(self.root / "receiver").receive(event.to_bytes())
        self.assertFalse(later.apply)

    def test_crash_after_local_commit_before_ack_replays_without_second_effect(self) -> None:
        def crash(name: str) -> None:
            if name == "after_local_commit_before_ack_eligibility":
                raise RuntimeError("synthetic crash")

        event = _event()
        with self.assertRaises(RuntimeError):
            ReplyReceiver(self.root / "receiver", phase_hook=crash).receive(event.to_bytes())
        replay = ReplyReceiver(self.root / "receiver").receive(event.to_bytes())
        self.assertFalse(replay.apply)
        self.assertEqual(ReplyAck.for_event(event), replay.ack)
        restarted = ReplyReceiver(self.root / "receiver")
        self.assertEqual((event,), restarted.pending_committed())
        self.assertFalse(
            restarted.commit_consumed(ReplyAck(1, event.event_id, "0" * 64))
        )
        self.assertEqual((event,), restarted.pending_committed())
        self.assertTrue(restarted.commit_consumed(ReplyAck.for_event(event)))
        self.assertTrue(restarted.commit_consumed(ReplyAck.for_event(event)))
        self.assertEqual((), ReplyReceiver(self.root / "receiver").pending_committed())

    def test_send_before_local_commit_then_replay_commits_event(self) -> None:
        def crash(name: str) -> None:
            if name == "after_local_canonical_file_fsync_before_rename":
                raise RuntimeError("synthetic crash")

        event = _event()
        with self.assertRaises(RuntimeError):
            ReplyReceiver(self.root / "receiver", phase_hook=crash).receive(event.to_bytes())
        replay = ReplyReceiver(self.root / "receiver").receive(event.to_bytes())
        self.assertTrue(replay.apply)
        self.assertIsNotNone(replay.ack)

    def test_same_id_different_digest_is_quarantined_without_success_ack(self) -> None:
        receiver = ReplyReceiver(self.root / "receiver", id_factory=lambda: "opaque")
        original = _event(answer="first SECRET")
        conflict = _event(answer="second SECRET")
        self.assertTrue(receiver.receive(original.to_bytes()).apply)
        result = receiver.receive(conflict.to_bytes())
        self.assertEqual(ReceiveDisposition.QUARANTINED, result.disposition)
        self.assertIsNone(result.ack)
        self.assertFalse(result.apply)
        self.assertEqual(original, receiver.read_committed(original.event_id))
        self.assertNotIn("SECRET", repr(result))

    def test_same_answer_with_changed_session_has_distinct_bound_digest(self) -> None:
        receiver = ReplyReceiver(self.root / "receiver")
        original = _event(session="session-one", answer="same answer")
        tampered = _event(session="session-two", answer="same answer")
        self.assertNotEqual(original.digest, tampered.digest)
        self.assertTrue(receiver.receive(original.to_bytes()).apply)
        result = receiver.receive(tampered.to_bytes())
        self.assertEqual(ReceiveDisposition.QUARANTINED, result.disposition)
        self.assertIsNone(result.ack)

    def test_corrupt_invalid_digest_and_partial_utf8_never_ack(self) -> None:
        receiver = ReplyReceiver(self.root / "receiver", id_factory=lambda: "opaque")
        event = _event()
        value = json.loads(event.to_bytes())
        value["answer"] = "tampered"
        invalid_digest = json.dumps(value, separators=(",", ":"), sort_keys=True).encode()
        for wire in (b'{"answer":"partial', invalid_digest, b"\xf0\x9f"):
            with self.subTest(wire_length=len(wire)):
                result = receiver.receive(wire)
                self.assertIsNone(result.ack)
                self.assertFalse(result.apply)
                self.assertEqual(ReceiveDisposition.CORRUPT, result.disposition)

    def test_oversized_corrupt_frame_quarantine_is_bounded_metadata_only(self) -> None:
        receiver = ReplyReceiver(self.root / "receiver", id_factory=lambda: "opaque")
        wire = b"SECRET" * (8 * 1024 * 1024 // 6 + 2)
        result = receiver.receive(wire)
        self.assertEqual(ReceiveDisposition.CORRUPT, result.disposition)
        artifact = next(receiver.quarantine.iterdir())
        self.assertLess(artifact.stat().st_size, 256)
        self.assertNotIn(b"SECRET", artifact.read_bytes())

    def test_quarantine_is_idempotently_keyed_by_input(self) -> None:
        receiver = ReplyReceiver(self.root / "receiver")
        corrupt = b'not-json-SECRET'
        for _ in range(20):
            self.assertEqual(
                ReceiveDisposition.CORRUPT,
                receiver.receive(corrupt).disposition,
            )
        self.assertEqual(1, len(tuple(receiver.quarantine.glob("*.bad"))))

        original = _event(answer="original")
        conflict = _event(answer="conflict")
        receiver.receive(original.to_bytes())
        for _ in range(20):
            self.assertEqual(
                ReceiveDisposition.QUARANTINED,
                receiver.receive(conflict.to_bytes()).disposition,
            )
        self.assertEqual(2, len(tuple(receiver.quarantine.glob("*.bad"))))

    def test_receiver_retention_only_removes_governed_artifacts(self) -> None:
        receiver = ReplyReceiver(self.root / "receiver")
        event = _event()
        self.assertTrue(receiver.receive(event.to_bytes()).apply)
        self.assertEqual((event,), receiver.pending_committed())
        ready = receiver.root / "ready"
        ready.mkdir()
        ready_wire = _event("event-ready").to_bytes()
        (ready / "event-ready.json").write_bytes(ready_wire)
        receiver.receive(b"corrupt-for-retention")

        stale_temps: list[Path] = []
        for directory in (
            receiver.root,
            receiver.canonical,
            receiver.dedupe,
            receiver.consumed,
        ):
            path = directory / ".crash-evidence.tmp"
            path.write_bytes(b"partial")
            stale_temps.append(path)
        fresh_temp = receiver.quarantine / ".fresh.tmp"
        fresh_temp.write_bytes(b"fresh")
        old = time.time() - 10_000
        for path in (*receiver.quarantine.glob("*.bad"), *stale_temps):
            os.utime(path, (old, old))

        result = receiver.apply_retention(
            max_age_seconds=1,
            max_count=0,
            stale_temp_age_seconds=1,
        )
        self.assertEqual(1, result.quarantine_removed)
        self.assertEqual(4, result.stale_temps_removed)
        self.assertEqual(1, result.canonical_preserved)
        self.assertEqual(1, result.pending_preserved)
        self.assertEqual((event,), receiver.pending_committed())
        self.assertTrue((receiver.canonical / f"{event.event_id}.json").exists())
        self.assertTrue((receiver.dedupe / f"{event.event_id}.json").exists())
        self.assertEqual(ready_wire, (ready / "event-ready.json").read_bytes())
        self.assertTrue(fresh_temp.exists())

    def test_concurrent_receiver_processes_serialize_one_apply(self) -> None:
        event = _event("event-concurrent", answer="private concurrent payload")
        source = str(Path(__file__).resolve().parents[1] / "src")
        environment = dict(os.environ)
        environment["PYTHONPATH"] = source + os.pathsep + environment.get("PYTHONPATH", "")
        code = r'''
import sys, time
from pathlib import Path
from talktomeclaude.reply import ReplyEvent, ReplyReceiver
event = ReplyEvent.from_bytes(bytes.fromhex(sys.argv[2]))
def phase(name):
    if name == "after_local_canonical_commit": time.sleep(0.25)
result = ReplyReceiver(Path(sys.argv[1]), phase_hook=phase).receive(event.to_bytes())
sys.stdout.write("1" if result.apply else "0")
'''
        command = [
            sys.executable,
            "-c",
            code,
            str(self.root / "receiver"),
            event.to_bytes().hex(),
        ]
        processes = [
            subprocess.Popen(
                command,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=environment,
            )
            for _ in range(2)
        ]
        outputs: list[str] = []
        for process in processes:
            stdout, stderr = process.communicate(timeout=10)
            self.assertEqual(0, process.returncode, stderr.decode(errors="replace"))
            outputs.append(stdout.decode("ascii"))
        self.assertEqual(["0", "1"], sorted(outputs))

        receiver = ReplyReceiver(self.root / "receiver")
        self.assertEqual((event,), receiver.pending_committed())
        self.assertEqual(1, len(tuple(receiver.canonical.glob("*.json"))))
        self.assertEqual(1, len(tuple(receiver.dedupe.glob("*.json"))))
        self.assertEqual((), tuple(receiver.root.rglob(".*.tmp")))


class HardCrashWatchdogTests(unittest.TestCase):
    """Real abrupt process exits remain bounded by a parent watchdog."""

    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _run_crasher(self, code: str, root: Path) -> subprocess.CompletedProcess[bytes]:
        environment = dict(os.environ)
        source = str(Path(__file__).resolve().parents[1] / "src")
        environment["PYTHONPATH"] = source + os.pathsep + environment.get("PYTHONPATH", "")
        return subprocess.run(
            [sys.executable, "-c", code, str(root)],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            timeout=10,
            check=False,
            env=environment,
        )

    def test_abrupt_writer_exit_after_fsync_leaves_no_visible_false_event(self) -> None:
        code = r'''
import os, sys
from talktomeclaude.reply import ReplyEvent, ReplySpool
def phase(name):
    if name == "after_file_fsync_before_ready_rename": os._exit(73)
ReplySpool(sys.argv[1], phase_hook=phase).enqueue(
    ReplyEvent.create(session="session", event_id="event-hard", answer="private"))
'''
        completed = self._run_crasher(code, self.root / "spool")
        self.assertEqual(73, completed.returncode)
        self.assertEqual((), ReplySpool(self.root / "spool").pending())

    def test_abrupt_receiver_exit_after_commit_replays_as_duplicate(self) -> None:
        code = r'''
import os, sys
from talktomeclaude.reply import ReplyEvent, ReplyReceiver
def phase(name):
    if name == "after_local_commit_before_ack_eligibility": os._exit(74)
event = ReplyEvent.create(session="session", event_id="event-hard", answer="private")
ReplyReceiver(sys.argv[1], phase_hook=phase).receive(event.to_bytes())
'''
        completed = self._run_crasher(code, self.root / "receiver")
        self.assertEqual(74, completed.returncode)
        event = ReplyEvent.create(session="session", event_id="event-hard", answer="private")
        replay = ReplyReceiver(self.root / "receiver").receive(event.to_bytes())
        self.assertEqual(ReceiveDisposition.DUPLICATE, replay.disposition)
        self.assertFalse(replay.apply)
        self.assertEqual(ReplyAck.for_event(event), replay.ack)
        restarted = ReplyReceiver(self.root / "receiver")
        self.assertEqual((event,), restarted.pending_committed())
        self.assertTrue(restarted.commit_consumed(ReplyAck.for_event(event)))
        self.assertEqual((), ReplyReceiver(self.root / "receiver").pending_committed())


if __name__ == "__main__":
    unittest.main()
