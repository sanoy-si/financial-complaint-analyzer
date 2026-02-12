"""A small fixed-window rate limiter.

In-process and dependency-free, with an injectable clock so it is fully
deterministic under test. For a multi-replica production deployment this same
interface can be backed by Redis; the public endpoint depends only on
``allow()``.
"""

from __future__ import annotations

import threading
import time
from collections import defaultdict
from typing import Callable


class InMemoryRateLimiter:
    def __init__(
        self,
        limit: int,
        window_seconds: float = 60.0,
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._limit = limit
        self._window = window_seconds
        self._clock = clock
        self._lock = threading.Lock()
        # key -> (window_start, count)
        self._buckets: dict[str, tuple[float, int]] = defaultdict(lambda: (0.0, 0))

    def allow(self, key: str) -> bool:
        now = self._clock()
        with self._lock:
            start, count = self._buckets[key]
            if now - start >= self._window:
                self._buckets[key] = (now, 1)
                return True
            if count >= self._limit:
                return False
            self._buckets[key] = (start, count + 1)
            return True
