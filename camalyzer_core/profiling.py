"""Simple profiling helpers."""

from contextlib import contextmanager
import time


@contextmanager
def time_block():
    """Context manager that measures elapsed time."""
    start = time.time()
    yield lambda: time.time() - start
