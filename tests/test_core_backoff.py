"""Deterministic bounded-jitter backoff tests."""

from __future__ import annotations

import unittest

from talktomeclaude.core import BackoffPolicy, JitteredBackoff


class FakeRandom:
    def __init__(self, values: list[float]) -> None:
        self.values = iter(values)

    def random(self) -> float:
        return next(self.values)


class FakeClock:
    def __init__(self) -> None:
        self.sleeps: list[float] = []

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)


class BackoffTests(unittest.TestCase):
    def test_jitter_uses_injected_random_and_stays_within_global_bounds(self) -> None:
        policy = BackoffPolicy(
            floor=1.0, ceiling=8.0, multiplier=2.0, jitter_ratio=0.25
        )
        backoff = JitteredBackoff(
            policy, FakeRandom([0.0, 0.5, 1.0, 0.0, 1.0, 0.5])
        )
        delays = [backoff.delay(attempt) for attempt in range(6)]
        self.assertEqual([1.0, 2.0, 5.0, 6.0, 8.0, 8.0], delays)
        self.assertTrue(all(policy.floor <= d <= policy.ceiling for d in delays))

    def test_wait_uses_injected_clock_and_returns_the_same_delay(self) -> None:
        clock = FakeClock()
        backoff = JitteredBackoff(
            BackoffPolicy(floor=0.5, ceiling=5.0, jitter_ratio=0),
            FakeRandom([0.25]),
        )
        delay = backoff.wait(2, clock)
        self.assertEqual(2.0, delay)
        self.assertEqual([2.0], clock.sleeps)

    def test_huge_attempt_saturates_before_exponentiation(self) -> None:
        backoff = JitteredBackoff(
            BackoffPolicy(
                floor=0.25,
                ceiling=30.0,
                multiplier=2.0,
                jitter_ratio=0,
            ),
            FakeRandom([0.5]),
        )
        self.assertEqual(30.0, backoff.delay(10**9))

    def test_invalid_policy_attempt_and_random_source_fail_closed(self) -> None:
        invalid_policies = [
            dict(floor=0),
            dict(floor=2, ceiling=1),
            dict(multiplier=0.5),
            dict(jitter_ratio=-0.1),
            dict(jitter_ratio=1.1),
            dict(ceiling=float("inf")),
            dict(multiplier=float("inf")),
        ]
        for kwargs in invalid_policies:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                BackoffPolicy(**kwargs)

        backoff = JitteredBackoff(BackoffPolicy(), FakeRandom([2.0]))
        with self.assertRaises(ValueError):
            backoff.delay(0)
        with self.assertRaises(ValueError):
            JitteredBackoff(BackoffPolicy(), FakeRandom([0.5])).delay(-1)


if __name__ == "__main__":
    unittest.main()
