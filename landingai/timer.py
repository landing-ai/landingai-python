"""Timer class for timing code execution and reporting results."""

import logging
import math
import pprint
import statistics
import time
from collections import defaultdict, deque
from contextlib import ContextDecorator
from dataclasses import dataclass, field
from enum import Enum
from functools import partial
from typing import Any, Callable, ClassVar, Dict, MutableSequence, Optional, Sequence


class TextColor(Enum):
    GRAY = "\x1b[38;21m"
    BLUE = "\x1b[38;5;39m"
    YELLOW = "\x1b[38;5;226m"
    RED = "\x1b[38;5;196m"
    BOLD_RED = "\x1b[31;1m"
    RESET = "\x1b[0m"


_MAX_SIZE = 10_000


class TimerStats:
    """Custom dictionary that stores time values of different timers.
    For each timer, it stores a sequence of time duration values. Each sequence is limited to a maximum size to avoid memory issues.

    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Add a private dictionary keeping track of all timings"""
        super().__init__(*args, **kwargs)
        self._timings: Dict[str, MutableSequence[float]] = defaultdict(
            partial(deque, maxlen=_MAX_SIZE)  # type: ignore
        )

    def add(self, name: str, value: float) -> None:
        """Add a timing value to the given timer."""
        self._timings[name].append(value)

    def clear(self) -> None:
        """Clear timers."""
        self._timings.clear()

    def apply(self, func: Callable[[Sequence[float]], float], name: str) -> float:
        """Apply a function to the results of one named timer."""
        if name in self._timings:
            return func(self._timings[name])
        raise KeyError(name)

    def count(self, name: str) -> float:
        """Number of timings."""
        return self.apply(len, name=name)

    def total(self, name: str) -> float:
        """Total time for timers."""
        return self.apply(sum, name=name)

    def min(self, name: str) -> float:
        """Minimal value of timings."""
        return self.apply(lambda values: min(values or [0]), name=name)

    def max(self, name: str) -> float:
        """Maximal value of timings."""
        return self.apply(lambda values: max(values or [0]), name=name)

    def mean(self, name: str) -> float:
        """Mean value of timings."""
        return self.apply(lambda values: statistics.mean(values or [0]), name=name)

    def median(self, name: str) -> float:
        """Median value of timings."""
        return self.apply(lambda values: statistics.median(values or [0]), name=name)

    def p95(self, name: str) -> float:
        return self.apply(
            lambda values: statistics.quantiles(
                values or [0], n=100, method="inclusive"
            )[95],
            name=name,
        )

    def stdev(self, name: str) -> float:
        """Standard deviation of timings."""
        if name in self._timings:
            value = self._timings[name]
            return statistics.stdev(value) if len(value) >= 2 else math.nan
        raise KeyError(name)

    def stats(self, name: str) -> Dict[str, float]:
        """All the stats for a given timer."""
        return {
            "count": self.count(name),
            "min": self.min(name),
            "max": self.max(name),
            "mean": self.mean(name),
            "median": self.median(name),
            "p95": self.p95(name),
            "stdev": self.stdev(name),
            "sum_total": self.total(name),
        }

    def __repr__(self) -> str:
        """Return a string representation of the TimerStats."""
        stats_repr = {k: self.stats(k) for k in self._timings.keys()}
        return f"{self.__class__.__name__}({pprint.pformat(stats_repr)})"


@dataclass
class Timer(ContextDecorator):
    """Time your code using a class, context manager, or decorator.
    See below examples for usage.
    As a class:
    ```
    t = Timer(name="class")
    t.start()
    # Do something
    t.stop()
    ```

    As a context manager:
    ```
    with Timer(name="context manager"):
        # Do something
    ```

    As a decorator:
    ```
    @Timer(name="decorator")
    def stuff():
        # Do something
    ```

    All the time values are stored in a global dictionary, accessible via the `stats` attribute.
    See the `TimerStats` class for more information.
    """

    # Global stats of all the timers
    stats: ClassVar[TimerStats] = TimerStats()
    # Instance attributes
    name: str = "default"
    text: str = (
        "Timer '{name}' finished. Elapsed time: {color}{:0.3f}{color_reset} seconds."
    )
    log_fn: Callable[[str], None] = logging.getLogger(__name__).info
    _elapsed_time: float = field(default=math.nan, init=False, repr=False)
    _start_time: Optional[float] = field(default=None, init=False, repr=False)

    def start(self) -> None:
        """Start a new timer."""
        if self._start_time is not None:
            raise ValueError("Timer is running. Use .stop() to stop it")

        self._start_time = time.perf_counter()

    def stop(self) -> float:
        """Stop the timer, and report the elapsed time."""
        if self._start_time is None:
            raise ValueError("Timer is not running. Use .start() to start it")

        # Calculate elapsed time
        self._elapsed_time = time.perf_counter() - self._start_time

        # Report elapsed time
        attributes = {
            "name": self.name,
            "milliseconds": self._elapsed_time * 1000,
            "seconds": self._elapsed_time,
            "minutes": self._elapsed_time / 60,
            "color": TextColor.BOLD_RED.value,
            "color_reset": TextColor.RESET.value,
        }
        text = self.text.format(self._elapsed_time, **attributes)
        self.log_fn(str(text))
        # Save stats
        Timer.stats.add(self.name, self._elapsed_time)

        self._start_time = None
        return self._elapsed_time

    def __enter__(self) -> "Timer":
        """Start a new timer as a context manager."""
        self.start()
        return self

    def __exit__(self, *exc_info: Any) -> None:
        """Stop the context manager timer."""
        self.stop()
