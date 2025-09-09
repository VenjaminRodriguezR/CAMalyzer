"""Simple helpers for memory and time profiling."""

from __future__ import annotations

import logging
import psutil
import time


def log_memory(prefix: str = "") -> None:
    """Log the current process RSS memory with an optional *prefix*."""
    process = psutil.Process()
    rss = process.memory_info().rss / (1024 ** 2)
    logging.info(f"{prefix}RSS memory: {rss:.1f} MB")


def time_block(func):
    """Decorator that logs the execution time of *func*."""

    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        logging.info(f"{func.__name__} took {end - start:.3f}s")
        return result

    return wrapper
