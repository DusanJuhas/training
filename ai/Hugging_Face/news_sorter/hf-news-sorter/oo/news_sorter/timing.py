"""A tiny stopwatch, so timing code does not clutter the classifiers."""

import time


class Timer:
    """Context manager that measures the wall-clock time of a ``with`` block.

    Example::

        with Timer() as timer:
            do_something_slow()
        print(timer.seconds)

    ``time.perf_counter()`` is used because it is the most precise clock for
    measuring durations (unlike ``time.time()`` it never jumps backwards).
    """

    def __init__(self) -> None:
        self.seconds: float = 0.0  # filled in when the ``with`` block ends
        self._start: float = 0.0

    def __enter__(self) -> "Timer":
        self._start = time.perf_counter()
        return self

    def __exit__(self, *exc_info: object) -> None:
        # Runs even if the block raised an exception; returning None lets the
        # exception propagate normally.
        self.seconds = time.perf_counter() - self._start
