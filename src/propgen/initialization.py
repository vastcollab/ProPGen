"""Initial-population strategies for ProSeD.

An initializer is a callable ``(landscape, pop_size, rng) -> (genotypes,
phenotypes)`` returning two integer arrays of length ``pop_size``. The four
constructors here cover every initialization used in the paper:

==========================  ==========================================
:func:`at`                  all individuals at one ``(g, p)`` pair
:func:`uniform_over_support`  uniform over pairs with ``pheno_probs > 0``
:func:`frequencies`         explicit frequency vector over pairs
:func:`from_arrays`         an exact, caller-supplied population
==========================  ==========================================
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import ArrayLike

    from .landscape import Landscape

__all__ = ["Init", "at", "uniform_over_support", "frequencies", "from_arrays"]

Init = Callable[["Landscape", int, np.random.Generator], tuple[np.ndarray, np.ndarray]]


def at(genotype: int, phenotype: int) -> Init:
    """All individuals start at a single genotype-phenotype pair.

    Used by the bridge (valley-crossing) experiment, where the population
    starts fixed at ``(0, 0)`` on one side of the fitness valley.
    """

    def _init(landscape: Landscape, pop_size: int, rng: np.random.Generator):
        if not 0 <= genotype < landscape.n_genotypes:
            raise ValueError(f"genotype {genotype} out of range for {landscape}")
        if not 0 <= phenotype < landscape.n_phenotypes:
            raise ValueError(f"phenotype {phenotype} out of range for {landscape}")
        return (
            np.full(pop_size, genotype, dtype=np.int64),
            np.full(pop_size, phenotype, dtype=np.int64),
        )

    _init.__propgen_repr__ = f"at(genotype={genotype}, phenotype={phenotype})"
    return _init


def uniform_over_support() -> Init:
    """Uniform over all pairs ``(g, p)`` with ``pheno_probs[g, p] > 0``.

    Pairs with zero mapping probability are excluded because they cannot be
    sustained by the dynamics. Used by the absolute-fitness and buoying
    experiments.
    """

    def _init(landscape: Landscape, pop_size: int, rng: np.random.Generator):
        gs, ps = np.nonzero(landscape.pheno_probs > 0)
        if gs.size == 0:
            raise ValueError("landscape has no genotype-phenotype pair with nonzero probability")
        pick = rng.integers(0, gs.size, size=pop_size)
        return gs[pick].astype(np.int64), ps[pick].astype(np.int64)

    _init.__propgen_repr__ = "uniform_over_support()"
    return _init


def frequencies(freqs: Mapping[tuple[int, int], float] | ArrayLike) -> Init:
    """Sample the initial population from an explicit frequency distribution.

    Accepts either a mapping ``{(g, p): frequency}`` or a full ``(Ng, Np)``
    array. Frequencies are normalised to sum to 1. Used by the mean-fitness
    experiment, which starts deliberately away from equilibrium.
    """
    if isinstance(freqs, Mapping):
        spec: Mapping[tuple[int, int], float] | None = dict(freqs)
        arr = None
    else:
        spec = None
        arr = np.asarray(freqs, dtype=np.float64)

    def _init(landscape: Landscape, pop_size: int, rng: np.random.Generator):
        Ng, Np = landscape.shape
        if spec is not None:
            table = np.zeros((Ng, Np), dtype=np.float64)
            for (g, p), value in spec.items():
                table[g, p] = value
        else:
            table = arr.reshape(Ng, Np).copy()

        total = table.sum()
        if total <= 0:
            raise ValueError("initial frequencies must contain at least one positive entry")
        flat = (table / total).ravel()

        pick = rng.choice(flat.size, size=pop_size, p=flat)
        return (pick // Np).astype(np.int64), (pick % Np).astype(np.int64)

    shown = spec if spec is not None else np.asarray(arr).ravel().tolist()
    _init.__propgen_repr__ = f"frequencies({shown})"
    return _init


def from_arrays(genotypes: ArrayLike, phenotypes: ArrayLike) -> Init:
    """Start from an exact population, e.g. the final state of a previous run."""
    g0 = np.asarray(genotypes, dtype=np.int64)
    p0 = np.asarray(phenotypes, dtype=np.int64)
    if g0.shape != p0.shape:
        raise ValueError(f"genotypes {g0.shape} and phenotypes {p0.shape} must have equal length")

    def _init(landscape: Landscape, pop_size: int, rng: np.random.Generator):
        if g0.size != pop_size:
            raise ValueError(f"supplied population has {g0.size} individuals, expected {pop_size}")
        return g0.copy(), p0.copy()

    _init.__propgen_repr__ = f"from_arrays(<{g0.size} individuals>)"
    return _init


def describe(init: Init) -> str:
    """Human-readable label for an initializer, recorded in run metadata."""
    return getattr(init, "__propgen_repr__", getattr(init, "__name__", repr(init)))
