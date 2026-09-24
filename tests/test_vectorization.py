"""The vectorised loop draws from the same distribution as the original.

:mod:`propgen.prosed` replaces three per-individual Python loops with array
operations, which makes it about 21x faster on the valley-crossing
configuration. That is only a legitimate optimisation if it does not change
the model. These tests compare against
:mod:`tests.reference_loop`, a transcription of the pre-1.0 implementation.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from propgen import Landscape, initialization, simulate
from propgen.prosed import _sample_categorical
from reference_loop import reference_run


def test_categorical_sampler_matches_np_random_choice():
    """Inverse-CDF sampling reproduces ``np.random.choice(p=...)`` frequencies."""
    pheno_probs = np.array([[0.4, 0.6], [0.7, 0.3], [0.1, 0.9]])
    cdf = np.cumsum(pheno_probs, axis=1)
    cdf[:, -1] = 1.0

    n = 200_000
    rng = np.random.default_rng(0)
    rows = rng.integers(0, 3, size=n)

    fast = _sample_categorical(rng, cdf, rows)

    for g in range(3):
        mask = rows == g
        observed = np.bincount(fast[mask], minlength=2)
        expected = pheno_probs[g] * mask.sum()
        chi2, p = stats.chisquare(observed, expected)
        assert p > 0.001, f"genotype {g}: chi-square p={p:.2g}, observed {observed}"


def test_categorical_sampler_handles_many_phenotypes():
    """The chunked path is exercised and stays correct for wide landscapes."""
    rng = np.random.default_rng(0)
    probs = rng.random((5, 40))
    probs /= probs.sum(axis=1, keepdims=True)
    cdf = np.cumsum(probs, axis=1)
    cdf[:, -1] = 1.0

    rows = rng.integers(0, 5, size=50_000)
    drawn = _sample_categorical(rng, cdf, rows)
    assert drawn.min() >= 0
    assert drawn.max() < 40

    mask = rows == 0
    observed = np.bincount(drawn[mask], minlength=40)
    chi2, p = stats.chisquare(observed, probs[0] * mask.sum())
    assert p > 0.001


@pytest.mark.parametrize("mutation_rate", [0.0, 0.1])
def test_equilibrium_frequencies_agree_with_reference_implementation(mutation_rate):
    """Both implementations settle on the same frequencies within noise.

    Individual trajectories differ -- the two consume random numbers in a
    different order -- so this compares the distributions they converge to,
    averaged over independent replicates.
    """
    adjacency = np.array([[0.0, 1.0], [1.0, 0.0]])
    pheno_probs = np.array([[0.4, 0.6], [0.7, 0.3]])
    repro_probs = np.array([0.009, 0.002])
    landscape = Landscape(adjacency, pheno_probs, repro_probs)

    n_reps, n_cycles, pop_size = 5, 150, 4000

    np.random.seed(0)
    reference = np.stack(
        [
            reference_run(
                adjacency, pheno_probs, repro_probs,
                n_cycles=n_cycles, pop_size=pop_size, mutation_rate=mutation_rate,
            )
            for _ in range(n_reps)
        ]
    )[..., -50:].mean(axis=(0, 3))

    fast = np.stack(
        [
            simulate(
                landscape, n_cycles=n_cycles, pop_size=pop_size,
                mutation_rate=mutation_rate, seed=rep,
                init=initialization.at(0, 0),
                compute_theory=False,
            ).frequencies
            for rep in range(n_reps)
        ]
    )[..., -50:].mean(axis=(0, 3))

    assert np.max(np.abs(reference - fast)) < 0.02, (
        f"reference {np.round(reference, 4)} vs vectorised {np.round(fast, 4)}"
    )


def test_dilution_preserves_a_uniform_subsample():
    """argpartition-based dilution is unbiased across individuals.

    The vectorised loop selects survivors with ``argpartition`` on uniform
    random keys rather than ``np.random.choice(replace=False)``. The set of
    survivors is distributed identically; only their order differs.
    """
    rng = np.random.default_rng(0)
    pop, keep = 1000, 100
    counts = np.zeros(pop)
    for _ in range(2000):
        survivors = np.argpartition(rng.random(pop), keep)[:keep]
        counts[survivors] += 1

    expected = 2000 * keep / pop
    chi2, p = stats.chisquare(counts, np.full(pop, expected))
    assert p > 0.001, f"dilution is biased: chi-square p={p:.2g}"
