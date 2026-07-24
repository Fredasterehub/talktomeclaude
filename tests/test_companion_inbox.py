from __future__ import annotations

import tempfile
import threading
import time
import unittest
from pathlib import Path

from talktomeclaude.companion.inbox import (
    DurableReplyInbox,
    InboxStatus,
    InboxStatusCode,
    SSHTransportOwner,
)
from talktomeclaude.reply import ReplyEvent, ReplyReceiver, ReplySpool
from talktomeclaude.reply.ssh import TransportResult


def _event(identity: str = "event-one", answer: str = "answer") -> ReplyEvent:
    return ReplyEvent.create(
        session="session-one",
        event_id=identity,
        answer=answer,
    )


class _SpoolSpy:
    def __init__(self, spool: ReplySpool, calls: list[str]) -> None:
        self._spool = spool
        self._calls = calls
        self.fail_next_ack = False

    def pending(self):
        self._calls.append("spool.pending")
        return self._spool.pending()

    def commit_ack(self, ack):
        self._calls.append("spool.commit_ack")
        if self.fail_next_ack:
            self.fail_next_ack = False
            raise OSError("synthetic ACK crash")
        return self._spool.commit_ack(ack)


class _ReceiverSpy:
    def __init__(self, receiver: ReplyReceiver, calls: list[str]) -> None:
        self._receiver = receiver
        self._calls = calls

    def receive(self, wire_bytes):
        self._calls.append("receiver.receive")
        return self._receiver.receive(wire_bytes)

    def pending_committed(self):
        self._calls.append("receiver.pending_committed")
        return self._receiver.pending_committed()

    def commit_consumed(self, ack):
        self._calls.append("receiver.commit_consumed")
        return self._receiver.commit_consumed(ack)


class DurableReplyInboxTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        root = Path(self.temporary.name)
        self.spool = ReplySpool(root / "spool")
        self.receiver = ReplyReceiver(root / "receiver")

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_local_effect_consumption_and_ack_have_strict_durable_order(self) -> None:
        calls: list[str] = []
        event = _event(answer="private answer")
        self.spool.enqueue(event)

        def accept(received: ReplyEvent) -> bool:
            calls.append("on_reply")
            return received == event

        inbox = DurableReplyInbox(
            _SpoolSpy(self.spool, calls),
            _ReceiverSpy(self.receiver, calls),
            accept,
        )

        result = inbox.drain_once()

        self.assertEqual(
            calls,
            [
                "receiver.pending_committed",
                "spool.pending",
                "receiver.receive",
                "on_reply",
                "receiver.commit_consumed",
                "spool.commit_ack",
            ],
        )
        self.assertEqual(result.received, 1)
        self.assertEqual(result.accepted, 1)
        self.assertEqual(result.consumed, 1)
        self.assertEqual(result.acknowledged, 1)
        self.assertEqual(self.spool.pending(), ())
        self.assertEqual(self.receiver.pending_committed(), ())

    def test_false_callback_leaves_pending_then_restart_applies_exactly_once(
        self,
    ) -> None:
        event = _event()
        self.spool.enqueue(event)
        replies: list[str] = []

        def defer(received: ReplyEvent) -> bool:
            replies.append(received.event_id)
            return False

        first = DurableReplyInbox(
            self.spool,
            self.receiver,
            defer,
        )

        deferred = first.drain_once()

        self.assertEqual(deferred.deferred, 1)
        self.assertEqual(len(self.spool.pending()), 1)
        self.assertEqual(self.receiver.pending_committed(), (event,))

        def accept(received: ReplyEvent) -> bool:
            replies.append(received.event_id)
            return True

        restarted = DurableReplyInbox(
            self.spool,
            ReplyReceiver(self.receiver.root),
            accept,
        )
        recovered = restarted.drain_once()
        empty = restarted.drain_once()

        self.assertEqual(replies, [event.event_id, event.event_id])
        self.assertEqual(recovered.recovered, 1)
        self.assertEqual(recovered.accepted, 1)
        self.assertEqual(recovered.acknowledged, 1)
        self.assertEqual(empty.accepted, 0)
        self.assertEqual(self.spool.pending(), ())
        self.assertEqual(self.receiver.pending_committed(), ())

    def test_restart_after_consumed_before_ack_never_repeats_effect(self) -> None:
        calls: list[str] = []
        event = _event()
        self.spool.enqueue(event)
        spool = _SpoolSpy(self.spool, calls)
        spool.fail_next_ack = True
        replies: list[str] = []

        def accept(received: ReplyEvent) -> bool:
            replies.append(received.event_id)
            return True

        first = DurableReplyInbox(
            spool,
            self.receiver,
            accept,
        )

        interrupted = first.drain_once()

        self.assertEqual(interrupted.consumed, 1)
        self.assertEqual(interrupted.acknowledged, 0)
        self.assertEqual(interrupted.faults, 1)
        self.assertEqual(self.receiver.pending_committed(), ())
        self.assertEqual(len(self.spool.pending()), 1)

        restarted = DurableReplyInbox(spool, self.receiver, lambda _event: False)
        completed = restarted.drain_once()

        self.assertEqual(replies, [event.event_id])
        self.assertEqual(completed.accepted, 0)
        self.assertEqual(completed.acknowledged, 1)
        self.assertEqual(self.spool.pending(), ())

    def test_receiver_only_pending_commit_is_recovered_without_spool_replay(
        self,
    ) -> None:
        event = _event()
        self.receiver.receive(event.to_bytes())
        replies: list[ReplyEvent] = []

        def accept(received: ReplyEvent) -> bool:
            replies.append(received)
            return True

        inbox = DurableReplyInbox(
            self.spool,
            self.receiver,
            accept,
        )

        result = inbox.drain_once()

        self.assertEqual(replies, [event])
        self.assertEqual(result.recovered, 1)
        self.assertEqual(result.received, 0)
        self.assertEqual(result.consumed, 1)
        self.assertEqual(result.acknowledged, 0)
        self.assertEqual(self.receiver.pending_committed(), ())

    def test_status_is_content_free_and_throwing_observer_is_isolated(self) -> None:
        secret = "SECRET-answer-مرحبا"
        self.spool.enqueue(_event(answer=secret))
        statuses: list[InboxStatus] = []

        def observe(status: InboxStatus) -> None:
            statuses.append(status)
            if status.code is InboxStatusCode.REPLY_RECEIVED:
                raise RuntimeError("observer failure")

        result = DurableReplyInbox(
            self.spool,
            self.receiver,
            lambda _event: True,
            on_status=observe,
        ).drain_once()

        self.assertEqual(result.acknowledged, 1)
        self.assertNotIn(secret, repr(statuses))
        self.assertTrue(
            {InboxStatusCode.REPLY_ACCEPTED, InboxStatusCode.ACK_COMMITTED}
            <= {status.code for status in statuses}
        )

    def test_background_owner_has_injectable_thread_and_sleep_and_stops_idempotently(
        self,
    ) -> None:
        created: list[str] = []
        slept = threading.Event()

        def factory(**options):
            created.append(str(options["name"]))
            return threading.Thread(**options)

        def sleep(_delay: float) -> None:
            slept.set()
            time.sleep(0.001)

        inbox = DurableReplyInbox(
            self.spool,
            self.receiver,
            lambda _event: True,
            thread_factory=factory,
            sleep=sleep,
            shutdown_timeout_seconds=0.2,
        )

        self.assertTrue(inbox.start())
        self.assertTrue(slept.wait(0.2))
        self.assertFalse(inbox.start())
        self.assertTrue(inbox.stop().stopped)
        self.assertEqual(inbox.stop(), inbox.stop())
        self.assertEqual(created, ["talktomeclaude-reply-inbox"])


class _CooperativeTransport:
    def __init__(self, *, boundary_fault: bool = False) -> None:
        self.entered = threading.Event()
        self.calls = 0
        self.boundary_fault = boundary_fault

    def run(self, stop: threading.Event) -> TransportResult:
        self.calls += 1
        self.entered.set()
        stop.wait()
        return TransportResult(0, 0, 0, True, self.boundary_fault)


class _BlockingTransport:
    def __init__(self) -> None:
        self.entered = threading.Event()
        self.release = threading.Event()

    def run(self, _stop: threading.Event) -> TransportResult:
        self.entered.set()
        self.release.wait()
        return TransportResult(0, 0, 0, True, False)


class SSHTransportOwnerTests(unittest.TestCase):
    def test_cooperative_transport_start_stop_is_bounded_and_idempotent(self) -> None:
        transport = _CooperativeTransport()
        owner = SSHTransportOwner(transport, shutdown_timeout_seconds=0.2)

        self.assertTrue(owner.start())
        self.assertTrue(transport.entered.wait(0.2))
        self.assertFalse(owner.start())
        self.assertEqual(owner.stop().boundary_replacement_required, False)
        self.assertTrue(owner.stop().stopped)
        self.assertEqual(transport.calls, 1)
        self.assertIsNotNone(owner.result)

    def test_uncooperative_transport_times_out_without_blocking_owner(self) -> None:
        transport = _BlockingTransport()
        statuses: list[InboxStatus] = []
        owner = SSHTransportOwner(
            transport,
            shutdown_timeout_seconds=0.01,
            on_status=statuses.append,
        )
        owner.start()
        self.assertTrue(transport.entered.wait(0.2))

        started = time.monotonic()
        result = owner.stop()

        self.assertLess(time.monotonic() - started, 0.2)
        self.assertFalse(result.stopped)
        self.assertTrue(result.boundary_replacement_required)
        self.assertIn(
            InboxStatusCode.SHUTDOWN_TIMEOUT,
            [status.code for status in statuses],
        )
        transport.release.set()
        deadline = time.monotonic() + 0.2
        while not owner.stop(0.01).stopped and time.monotonic() < deadline:
            pass
        self.assertTrue(owner.stop().stopped)

    def test_transport_boundary_fault_is_preserved_in_stop_result(self) -> None:
        transport = _CooperativeTransport(boundary_fault=True)
        owner = SSHTransportOwner(transport, shutdown_timeout_seconds=0.2)
        owner.start()
        self.assertTrue(transport.entered.wait(0.2))

        result = owner.stop()

        self.assertTrue(result.stopped)
        self.assertTrue(result.boundary_replacement_required)


if __name__ == "__main__":
    unittest.main()
