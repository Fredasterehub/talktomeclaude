"""Bounded worker shutdown tests, including non-cooperative callbacks."""

from __future__ import annotations

import threading
import time
import unittest

from talktomeclaude.core import BoundedWorker


class WorkerShutdownTests(unittest.TestCase):
    def test_cooperative_worker_stops_without_boundary_replacement(self) -> None:
        entered = threading.Event()

        def cooperative(stop: threading.Event) -> None:
            entered.set()
            stop.wait()

        worker = BoundedWorker(cooperative)
        worker.start()
        self.assertTrue(entered.wait(1))
        result = worker.shutdown(0.5)
        self.assertTrue(result.stopped)
        self.assertFalse(result.boundary_replacement_required)
        self.assertFalse(worker.alive)

    def test_non_cooperative_worker_cannot_own_control_thread_past_deadline(self) -> None:
        entered = threading.Event()
        release = threading.Event()

        def non_cooperative(_stop: threading.Event) -> None:
            entered.set()
            release.wait()

        worker = BoundedWorker(non_cooperative)
        worker.start()
        self.assertTrue(entered.wait(1))
        started = time.perf_counter()
        result = worker.shutdown(0.02)
        elapsed = time.perf_counter() - started
        try:
            self.assertFalse(result.stopped)
            self.assertTrue(result.boundary_replacement_required)
            self.assertLess(elapsed, 0.25)
            self.assertTrue(worker.alive)
        finally:
            release.set()

    def test_shutdown_before_start_is_safe_and_idempotent(self) -> None:
        worker = BoundedWorker(lambda _stop: None)
        first = worker.shutdown(0)
        second = worker.shutdown(100)
        self.assertIs(first, second)
        self.assertTrue(first.stopped)
        self.assertFalse(first.boundary_replacement_required)
        with self.assertRaises(RuntimeError):
            worker.start()

    def test_worker_is_single_use_and_deadline_must_be_non_negative(self) -> None:
        worker = BoundedWorker(lambda _stop: None)
        worker.start()
        worker.shutdown(1)
        with self.assertRaises(RuntimeError):
            worker.start()
        with self.assertRaises(ValueError):
            BoundedWorker(lambda _stop: None).shutdown(-0.1)

    def test_concurrent_shutdown_timeout_taints_boundary_permanently(self) -> None:
        entered = threading.Event()
        release = threading.Event()
        callers_ready = threading.Barrier(3)
        results: dict[str, object] = {}

        def non_cooperative(_stop: threading.Event) -> None:
            entered.set()
            release.wait()

        def shutdown(label: str, deadline: float) -> None:
            callers_ready.wait()
            results[label] = worker.shutdown(deadline)

        worker = BoundedWorker(non_cooperative)
        worker.start()
        self.assertTrue(entered.wait(1))
        short = threading.Thread(target=shutdown, args=("short", 0.01))
        long = threading.Thread(target=shutdown, args=("long", 0.5))
        short.start()
        long.start()
        callers_ready.wait()
        short.join(0.2)
        self.assertFalse(short.is_alive())
        release.set()
        long.join(1)
        self.assertFalse(long.is_alive())

        short_result = results["short"]
        long_result = results["long"]
        self.assertTrue(short_result.boundary_replacement_required)
        self.assertTrue(long_result.boundary_replacement_required)
        self.assertTrue(long_result.stopped)
        cached = worker.shutdown(1)
        self.assertTrue(cached.boundary_replacement_required)


if __name__ == "__main__":
    unittest.main()
