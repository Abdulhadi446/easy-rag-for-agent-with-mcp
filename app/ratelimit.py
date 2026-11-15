"""Tiny in-process throttle for login attempts.

Good enough for a single-process deployment; swap for Redis when you run more
than one web process.
"""

from __future__ import annotations

import threading
import time
from collections import defaultdict, deque

MAX_FAILURES = 8
WINDOW_SECONDS = 10 * 60

_lock = threading.Lock()
_failures: dict[str, deque[float]] = defaultdict(deque)


def _prune(bucket: deque[float], now: float) -> None:
    while bucket and now - bucket[0] > WINDOW_SECONDS:
        bucket.popleft()


def is_limited(key: str) -> bool:
    now = time.monotonic()
    with _lock:
        bucket = _failures[key]
        _prune(bucket, now)
        return len(bucket) >= MAX_FAILURES


def record_failure(key: str) -> None:
    now = time.monotonic()
    with _lock:
        bucket = _failures[key]
        _prune(bucket, now)
        bucket.append(now)


def reset(key: str) -> None:
    with _lock:
        _failures.pop(key, None)
