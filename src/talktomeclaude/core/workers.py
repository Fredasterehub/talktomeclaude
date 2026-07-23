"""Bounded lifecycle helpers for isolated or daemon worker boundaries."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True, slots=True)
class ShutdownResult:
    stopped: bool
    boundary_replacement_required: bool
    deadline_seconds: float


class BoundedWorker:
    """Run one callback without letting it own control-thread shutdown.

    Python cannot safely terminate a thread.  Therefore a timed-out callback is
    a daemon and the returned result explicitly requires replacement of its
    process/isolation boundary.  Callers must not reuse that boundary.
    """

    def __init__(
        self,
        callback: Callable[[threading.Event], None],
        *,
        name: str = "companion-worker",
    ) -> None:
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=callback,
            args=(self._stop,),
            name=name,
            daemon=True,
        )
        self._started = False
        self._shutdown_result: ShutdownResult | None = None
        self._boundary_tainted = False
        self._lock = threading.Lock()

    @property
    def alive(self) -> bool:
        return self._started and self._thread.is_alive()

    def start(self) -> None:
        with self._lock:
            if self._started or self._shutdown_result is not None:
                raise RuntimeError("worker may only be started once")
            self._started = True
            self._thread.start()

    def shutdown(self, deadline_seconds: float) -> ShutdownResult:
        if deadline_seconds < 0:
            raise ValueError("shutdown deadline must be non-negative")
        with self._lock:
            if self._shutdown_result is not None:
                return self._shutdown_result
            if not self._started:
                self._shutdown_result = ShutdownResult(
                    stopped=True,
                    boundary_replacement_required=False,
                    deadline_seconds=deadline_seconds,
                )
                return self._shutdown_result
            self._stop.set()
        self._thread.join(deadline_seconds)
        stopped = not self._thread.is_alive()
        with self._lock:
            if not stopped:
                self._boundary_tainted = True
            previous = self._shutdown_result
            result = ShutdownResult(
                stopped=stopped or (previous.stopped if previous else False),
                boundary_replacement_required=self._boundary_tainted,
                deadline_seconds=(
                    min(deadline_seconds, previous.deadline_seconds)
                    if previous
                    else deadline_seconds
                ),
            )
            self._shutdown_result = result
        return result
