"""Bounded exponential reconnect backoff with deterministic injection points."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol


class RandomSource(Protocol):
    def random(self) -> float: ...


class SleepClock(Protocol):
    def sleep(self, seconds: float) -> None: ...


@dataclass(frozen=True, slots=True)
class BackoffPolicy:
    floor: float = 0.25
    ceiling: float = 30.0
    multiplier: float = 2.0
    jitter_ratio: float = 0.2

    def __post_init__(self) -> None:
        if not math.isfinite(self.floor) or self.floor <= 0:
            raise ValueError("backoff floor must be positive")
        if not math.isfinite(self.ceiling) or self.ceiling < self.floor:
            raise ValueError("backoff ceiling must be at least the floor")
        if not math.isfinite(self.multiplier) or self.multiplier < 1:
            raise ValueError("backoff multiplier must be at least one")
        if not math.isfinite(self.jitter_ratio) or not 0 <= self.jitter_ratio <= 1:
            raise ValueError("jitter ratio must be between zero and one")


class JitteredBackoff:
    def __init__(self, policy: BackoffPolicy, random_source: RandomSource) -> None:
        self.policy = policy
        self._random = random_source

    def delay(self, attempt: int) -> float:
        """Return delay for zero-based ``attempt``, clamped after jitter."""

        if attempt < 0:
            raise ValueError("attempt must be non-negative")
        if (
            self.policy.floor == self.policy.ceiling
            or self.policy.multiplier == 1
        ):
            base = self.policy.floor
        else:
            attempts_to_ceiling = math.ceil(
                math.log(self.policy.ceiling / self.policy.floor)
                / math.log(self.policy.multiplier)
            )
            if attempt >= attempts_to_ceiling:
                base = self.policy.ceiling
            else:
                base = self.policy.floor * self.policy.multiplier**attempt
        sample = self._random.random()
        if not 0.0 <= sample <= 1.0:
            raise ValueError("random source must return a value in [0, 1]")
        spread = base * self.policy.jitter_ratio
        jittered = base - spread + (2.0 * spread * sample)
        return min(self.policy.ceiling, max(self.policy.floor, jittered))

    def wait(self, attempt: int, clock: SleepClock) -> float:
        delay = self.delay(attempt)
        clock.sleep(delay)
        return delay
