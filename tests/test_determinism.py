"""Seeded runs are reproducible, and the global NumPy random state is untouched.

The pre-1.0 scripts read a ``seed`` from their config but never applied it --
``np.random.seed(seed)`` was commented out in every one of them -- so no
published run could be reproduced, even on the same machine. These tests
prevent that from recurring.
"""

from __future__ import annotations

import numpy as np

from propgen import Landscape, simulate

LANDSCAPE = Landscape(
    adjacency=np.array([[0.0, 1.0], [1.0, 0.0]]),
    pheno_probs=np.array([[0.4, 0.6], [0.7, 0.3]]),
    repro_probs=np.array([0.009, 0.002]),
)

KWARGS = dict(n_cycles=30, pop_size=2000, mutation_rate=0.1, compute_theory=False)


def test_same_seed_is_bitwise_identical():
    a = simulate(LANDSCAPE, seed=7, **KWARGS)
    b = simulate(LANDSCAPE, seed=7, **KWARGS)
    assert np.array_equal(a.frequencies, b.frequencies)
    assert np.array_equal(a.genotypes, b.genotypes)
    assert np.array_equal(a.phenotypes, b.phenotypes)


def test_different_seeds_differ():
    a = simulate(LANDSCAPE, seed=7, **KWARGS)
    b = simulate(LANDSCAPE, seed=8, **KWARGS)
    assert not np.array_equal(a.frequencies, b.frequencies)


def test_sweep_seed_reproduces_a_single_trial_in_isolation():
    """``seed=[master, trial]`` lets one trial of a sweep be re-run alone.

    This matters when a long parameter sweep partially fails: the failed
    trials can be re-run without repeating the whole sweep, and they reproduce
    exactly what the full sweep would have produced.
    """
    a = simulate(LANDSCAPE, seed=[42, 57], **KWARGS)
    b = simulate(LANDSCAPE, seed=[42, 57], **KWARGS)
    c = simulate(LANDSCAPE, seed=[42, 58], **KWARGS)
    assert np.array_equal(a.frequencies, b.frequencies)
    assert not np.array_equal(a.frequencies, c.frequencies)


def test_global_numpy_random_state_is_not_used_or_modified():
    """Simulation must not consume draws from the legacy global RNG."""
    np.random.seed(0)
    expected = np.random.random(5)

    np.random.seed(0)
    simulate(LANDSCAPE, seed=3, **KWARGS)
    actual = np.random.random(5)

    assert np.array_equal(expected, actual)


def test_seed_is_recorded_in_the_result():
    result = simulate(LANDSCAPE, seed=11, **KWARGS)
    assert result.params["seed"] == 11

    result = simulate(LANDSCAPE, seed=[42, 7], **KWARGS)
    assert result.params["seed"] == [42, 7]
