"""Environment schedules for ProSeD.

A schedule maps a dilution-cycle index to an environment index, selecting
which landscape from a sequence is in force during that cycle. This is how
the persister model switches between the antibiotic-free and antibiotic
environments.

Making the switch times an explicit, recorded part of the run configuration
replaces the previous arrangement, where the switch point was a literal
inside the simulation loop while the output filename advertised a different
value -- the two could disagree silently.
"""

from __future__ import annotations

from collections.abc import Callable

__all__ = ["Schedule", "constant", "switch_at", "periodic", "describe"]

Schedule = Callable[[int], int]


def constant() -> Schedule:
    """A single, unchanging environment."""

    def _schedule(cycle: int) -> int:
        return 0

    _schedule.__propgen_repr__ = "constant()"
    _schedule.__propgen_n_env__ = 1
    return _schedule


def switch_at(*cycles: int) -> Schedule:
    """Switch to the next environment at each of the given cycle indices.

    ``switch_at(100)`` runs environment 0 for cycles 0-99 and environment 1
    from cycle 100 onward -- the behaviour used for the published persister
    runs.
    """
    points = sorted(int(c) for c in cycles)
    if any(c < 0 for c in points):
        raise ValueError(f"switch cycles must be non-negative, got {points}")

    def _schedule(cycle: int) -> int:
        env = 0
        for point in points:
            if cycle >= point:
                env += 1
            else:
                break
        return env

    _schedule.__propgen_repr__ = f"switch_at({', '.join(map(str, points))})"
    _schedule.__propgen_n_env__ = len(points) + 1
    return _schedule


def periodic(period: int, n_env: int = 2) -> Schedule:
    """Cycle through ``n_env`` environments every ``period`` dilution cycles."""
    if period <= 0:
        raise ValueError(f"period must be positive, got {period}")
    if n_env < 1:
        raise ValueError(f"n_env must be at least 1, got {n_env}")

    def _schedule(cycle: int) -> int:
        return (cycle // period) % n_env

    _schedule.__propgen_repr__ = f"periodic(period={period}, n_env={n_env})"
    _schedule.__propgen_n_env__ = n_env
    return _schedule


def describe(schedule: Schedule) -> str:
    """Human-readable label for a schedule, recorded in run metadata."""
    return getattr(schedule, "__propgen_repr__", getattr(schedule, "__name__", repr(schedule)))
