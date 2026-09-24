"""Reduce per-trial simulation runs to committed summary files.

Figures in the paper plot the mean and standard error across independent
trials, never an individual run. That reduction is performed once here and the
result committed to ``results/``, so the figure notebooks execute from a fresh
clone without the raw per-trial output -- which runs to gigabytes for the
valley-crossing sweep.

Summaries are stored as a single ``.npz`` per experiment with this layout:

==============  =================================  =============================
``param_names`` ``(k,)`` str                        names of the swept parameters
``params``      ``(n_cond, k)`` float64             coordinate of each condition
``mean``        ``(n_cond, Ng, Np, n_t)`` float32   mean frequency over trials
``sem``         ``(n_cond, Ng, Np, n_t)`` float32   standard error over trials
``cycles``      ``(n_t,)`` int32                    dilution cycle of each column
``n_trials``    ``(n_cond,)`` int32                 trials contributing
``f_eq``        ``(n_cond, Ng*Np)`` float64         analytic equilibrium
``Xbar_theory`` ``(n_cond,)`` float64               analytic mean fitness
``meta``        JSON string                         provenance
==============  =================================  =============================
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .results import SimResult

__all__ = ["Summary", "collect_runs", "aggregate_runs", "load_summary"]

#: Filename patterns for output written by the pre-1.0 simulation scripts.
#: The trial number is dropped (it is the axis reduced over); the remaining
#: named groups become the condition coordinates.
LEGACY_PATTERNS = {
    "bridge": re.compile(
        r"sim_data_map_(?P<bridge_mapping>[\d.]+)_fit_(?P<gamma>[\d.]+)_trial_(?P<trial>\d+)\.pkl"
    ),
    "buoy": re.compile(r"sim_data_map_(?P<mapping>[\d.]+)_trial_(?P<trial>\d+)\.pkl"),
    "absfit": re.compile(
        r"sim_data_r1_(?P<r1>[\d.]+)_r2_(?P<r2>[\d.]+)_trial_(?P<trial>\d+)\.pkl"
    ),
    "meanfit": re.compile(r"sim_data_trial_(?P<trial>\d+)\.pkl"),
}


@dataclass
class Summary:
    """Aggregated results for one experiment, across all its conditions."""

    param_names: list[str]
    params: np.ndarray
    mean: np.ndarray
    sem: np.ndarray
    cycles: np.ndarray
    n_trials: np.ndarray
    f_eq: np.ndarray | None = None
    Xbar_theory: np.ndarray | None = None
    meta: dict[str, Any] | None = None

    @property
    def n_conditions(self) -> int:
        return self.mean.shape[0]

    @property
    def shape(self) -> tuple[int, int]:
        """``(n_genotypes, n_phenotypes)``."""
        return self.mean.shape[1:3]

    def index(self, **kwargs: float) -> int:
        """Index of the condition matching the given parameter values.

        Values are matched with a small tolerance, so ``mapping=0.3`` finds a
        condition stored as ``0.30000000000000004``.
        """
        mask = np.ones(self.n_conditions, dtype=bool)
        for name, value in kwargs.items():
            if name not in self.param_names:
                raise KeyError(
                    f"{name!r} is not a swept parameter; available: {self.param_names}"
                )
            col = self.param_names.index(name)
            mask &= np.isclose(self.params[:, col], value, rtol=0, atol=1e-8)

        hits = np.flatnonzero(mask)
        if hits.size == 0:
            available = sorted({tuple(np.round(row, 6)) for row in self.params})
            raise KeyError(
                f"no condition matches {kwargs}; available {self.param_names} "
                f"values: {available[:12]}{' ...' if len(available) > 12 else ''}"
            )
        if hits.size > 1:
            raise KeyError(f"{kwargs} matches {hits.size} conditions; be more specific")
        return int(hits[0])

    def select(self, **kwargs: float) -> dict[str, Any]:
        """Return ``mean``, ``sem``, ``cycles``, ``f_eq`` for one condition.

        Examples
        --------
        >>> s = load_summary("results/bridge/summary.npz")   # doctest: +SKIP
        >>> panel = s.select(bridge_mapping=0.9, gamma=0.0002)   # doctest: +SKIP
        >>> panel["mean"].shape   # doctest: +SKIP
        (3, 2, 3000)
        """
        i = self.index(**kwargs)
        return {
            "mean": self.mean[i],
            "sem": self.sem[i],
            "cycles": self.cycles,
            "n_trials": int(self.n_trials[i]),
            "f_eq": None if self.f_eq is None else self.f_eq[i],
            "Xbar_theory": None if self.Xbar_theory is None else float(self.Xbar_theory[i]),
        }

    def values(self, name: str) -> np.ndarray:
        """Sorted unique values taken by one swept parameter."""
        col = self.param_names.index(name)
        return np.unique(self.params[:, col])

    def to_npz(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays = {
            "param_names": np.array(self.param_names, dtype="U64"),
            "params": np.asarray(self.params, dtype=np.float64),
            "mean": np.asarray(self.mean, dtype=np.float32),
            "sem": np.asarray(self.sem, dtype=np.float32),
            "cycles": np.asarray(self.cycles, dtype=np.int32),
            "n_trials": np.asarray(self.n_trials, dtype=np.int32),
            "meta": np.array(json.dumps(self.meta or {}, default=str)),
        }
        if self.f_eq is not None:
            arrays["f_eq"] = np.asarray(self.f_eq, dtype=np.float64)
        if self.Xbar_theory is not None:
            arrays["Xbar_theory"] = np.asarray(self.Xbar_theory, dtype=np.float64)
        np.savez_compressed(path, **arrays)
        return path

    def __repr__(self) -> str:
        Ng, Np = self.shape
        return (
            f"Summary(n_conditions={self.n_conditions}, params={self.param_names}, "
            f"n_genotypes={Ng}, n_phenotypes={Np}, n_recorded={self.mean.shape[-1]})"
        )


def load_summary(path: str | Path) -> Summary:
    """Load a summary written by :meth:`Summary.to_npz`."""
    with np.load(path, allow_pickle=False) as z:
        return Summary(
            param_names=[str(s) for s in z["param_names"]],
            params=z["params"],
            mean=z["mean"],
            sem=z["sem"],
            cycles=z["cycles"],
            n_trials=z["n_trials"],
            f_eq=z["f_eq"] if "f_eq" in z else None,
            Xbar_theory=z["Xbar_theory"] if "Xbar_theory" in z else None,
            meta=json.loads(str(z["meta"])) if "meta" in z else {},
        )


def collect_runs(
    raw_dir: str | Path,
    *,
    pattern: str | re.Pattern | None = None,
    glob: str = "*",
) -> dict[tuple[tuple[str, float], ...], list[Path]]:
    """Group run files by condition, reducing over the trial axis.

    ``pattern`` is a regex with named groups; every group except ``trial``
    becomes a condition coordinate. Pass one of the keys of
    :data:`LEGACY_PATTERNS` to use a built-in pattern for pre-1.0 output.
    """
    raw_dir = Path(raw_dir)
    if isinstance(pattern, str) and pattern in LEGACY_PATTERNS:
        pattern = LEGACY_PATTERNS[pattern]
    elif isinstance(pattern, str):
        pattern = re.compile(pattern)

    groups: dict[tuple[tuple[str, float], ...], list[Path]] = {}
    for path in sorted(raw_dir.glob(glob)):
        if not path.is_file():
            continue
        if pattern is not None:
            match = pattern.match(path.name)
            if not match:
                continue
            coords = tuple(
                (name, float(value))
                for name, value in sorted(match.groupdict().items())
                if name != "trial"
            )
        else:
            coords = _coords_from_file(path)
            if coords is None:
                continue
        groups.setdefault(coords, []).append(path)

    if not groups:
        raise FileNotFoundError(
            f"no run files matched in {raw_dir} (glob={glob!r}, "
            f"pattern={getattr(pattern, 'pattern', pattern)!r})"
        )
    return groups


def _coords_from_file(path: Path) -> tuple[tuple[str, float], ...] | None:
    """Read condition coordinates recorded inside a ``.npz`` run."""
    if path.suffix != ".npz":
        return None
    with np.load(path, allow_pickle=False) as z:
        if "meta" not in z:
            return None
        meta = json.loads(str(z["meta"]))
    sweep = meta.get("sweep") or {}
    return tuple((name, float(value)) for name, value in sorted(sweep.items()))


def aggregate_runs(
    groups: dict[tuple[tuple[str, float], ...], list[Path]],
    *,
    thin: int = 1,
    meta: dict[str, Any] | None = None,
) -> Summary:
    """Reduce grouped runs to per-condition mean and standard error.

    ``thin`` keeps every ``k``-th recorded cycle. For the 30,000-cycle
    valley-crossing sweep this is what brings the committed summary down to a
    few megabytes; the recorded cycle indices are stored alongside, so the
    thinning is visible to anyone reading the file.
    """
    if thin < 1:
        raise ValueError(f"thin must be at least 1, got {thin}")

    conditions = sorted(groups)
    param_names = [name for name, _ in conditions[0]] if conditions[0] else []

    means, sems, n_trials, f_eqs, xbars = [], [], [], [], []
    cycles: np.ndarray | None = None

    for coords in conditions:
        if [name for name, _ in coords] != param_names:
            raise ValueError(
                f"inconsistent parameter names across conditions: "
                f"{param_names} vs {[n for n, _ in coords]}"
            )
        results = [_load_any(p) for p in groups[coords]]
        stack = np.stack([r.frequencies[..., ::thin] for r in results]).astype(np.float64)

        means.append(stack.mean(axis=0).astype(np.float32))
        n = stack.shape[0]
        sems.append(
            (stack.std(axis=0, ddof=1) / np.sqrt(n)).astype(np.float32)
            if n > 1
            else np.zeros_like(stack[0], dtype=np.float32)
        )
        n_trials.append(n)

        if cycles is None:
            cycles = results[0].cycles[::thin]
        f_eqs.append(results[0].f_eq)
        xbars.append(results[0].Xbar_theory)

    have_theory = all(f is not None for f in f_eqs)
    summary_meta = dict(meta or {})
    summary_meta.setdefault("thin", thin)
    summary_meta.setdefault("n_source_files", sum(len(v) for v in groups.values()))

    return Summary(
        param_names=param_names,
        params=np.array([[value for _, value in coords] for coords in conditions], dtype=np.float64)
        .reshape(len(conditions), len(param_names)),
        mean=np.stack(means),
        sem=np.stack(sems),
        cycles=np.asarray(cycles, dtype=np.int32),
        n_trials=np.asarray(n_trials, dtype=np.int32),
        f_eq=np.stack(f_eqs) if have_theory else None,
        Xbar_theory=np.asarray(xbars, dtype=np.float64) if have_theory else None,
        meta=summary_meta,
    )


def _load_any(path: Path) -> SimResult:
    if path.suffix == ".pkl":
        return SimResult.from_legacy_pickle(path)
    return SimResult.from_npz(path)


def summarize(paths: Iterable[str | Path], *, thin: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """Mean and standard error across a set of runs of the same condition."""
    stack = np.stack(
        [_load_any(Path(p)).frequencies[..., ::thin] for p in paths]
    ).astype(np.float64)
    n = stack.shape[0]
    sem = stack.std(axis=0, ddof=1) / np.sqrt(n) if n > 1 else np.zeros_like(stack[0])
    return stack.mean(axis=0), sem
