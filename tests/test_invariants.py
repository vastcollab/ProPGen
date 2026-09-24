"""Structural invariants of the ProSeD algorithm and landscape validation."""

from __future__ import annotations

import numpy as np
import pytest

from propgen import Landscape, ProSeD, initialization, schedule, simulate

LANDSCAPE = Landscape(
    adjacency=np.array([[0.0, 1.0], [1.0, 0.0]]),
    pheno_probs=np.array([[0.4, 0.6], [0.7, 0.3]]),
    repro_probs=np.array([0.009, 0.002]),
)


def test_frequencies_sum_to_one_at_every_cycle():
    result = simulate(
        LANDSCAPE, n_cycles=40, pop_size=3000, mutation_rate=0.1, seed=0, compute_theory=False
    )
    assert np.allclose(result.frequencies.sum(axis=(0, 1)), 1.0)


def test_population_size_is_restored_by_dilution():
    sim = ProSeD(LANDSCAPE, pop_size=5000, mutation_rate=0.1, seed=0)
    for _ in range(10):
        sim.step()
        assert sim.genotypes.size == 5000
        assert sim.phenotypes.size == 5000


def test_recorded_shape_and_cycle_indices():
    result = simulate(
        LANDSCAPE, n_cycles=100, pop_size=1000, mutation_rate=0.1, seed=0,
        record_every=10, compute_theory=False,
    )
    assert result.frequencies.shape == (2, 2, 10)
    assert np.array_equal(result.cycles, np.arange(0, 100, 10))


def test_zero_mutation_preserves_the_genotype_multiset():
    """With mu=0 no genotype can appear that was not present initially."""
    landscape = Landscape(
        adjacency=np.array([[0.0, 1.0], [1.0, 0.0]]),
        pheno_probs=np.array([[0.5, 0.5], [0.5, 0.5]]),
        repro_probs=np.array([0.01, 0.01]),
    )
    result = simulate(
        landscape, n_cycles=30, pop_size=2000, mutation_rate=0.0, seed=0,
        init=initialization.at(0, 0), compute_theory=False,
    )
    assert np.all(result.genotypes == 0)
    assert np.allclose(result.frequencies[1].sum(), 0.0)


def test_offspring_per_division_increases_turnover():
    """Larger c means more offspring competing at dilution."""
    common = dict(n_cycles=20, pop_size=2000, mutation_rate=0.5, seed=0, compute_theory=False)
    one = simulate(LANDSCAPE, offspring_per_division=1, **common)
    four = simulate(LANDSCAPE, offspring_per_division=4, **common)
    assert not np.array_equal(one.frequencies, four.frequencies)


def test_environment_schedule_switches_landscape():
    """A switching schedule changes the dynamics at the declared cycle."""
    normal = Landscape(
        adjacency=np.array([[1.0]]),
        pheno_probs=np.array([[0.8, 0.2]]),
        repro_probs=np.array([0.02, 0.0002]),
    )
    antibiotic = Landscape(
        adjacency=np.array([[1.0]]),
        pheno_probs=np.array([[0.2, 0.8]]),
        repro_probs=np.array([0.0, 0.05]),
    )
    result = simulate(
        [normal, antibiotic],
        schedule=schedule.switch_at(50),
        n_cycles=120,
        pop_size=5000,
        mutation_rate=0.0,
        seed=0,
        compute_theory=False,
    )
    persister_before = result.frequencies[0, 1, 40:49].mean()
    persister_after = result.frequencies[0, 1, 100:].mean()
    assert persister_after > persister_before, (
        f"persister frequency should rise under antibiotic: "
        f"{persister_before:.3f} -> {persister_after:.3f}"
    )


def test_schedule_required_when_several_landscapes_given():
    with pytest.raises(ValueError, match="no schedule"):
        simulate(
            [LANDSCAPE, LANDSCAPE], n_cycles=5, pop_size=100, mutation_rate=0.0,
        )


@pytest.mark.parametrize(
    "kwargs, message",
    [
        (dict(adjacency=np.array([[0.0, 1.0]]), pheno_probs=np.eye(2),
              repro_probs=np.ones(2) * 0.1), "square"),
        (dict(adjacency=np.array([[0.0, 1.0], [0.0, 0.0]]), pheno_probs=np.eye(2),
              repro_probs=np.ones(2) * 0.1), "symmetric"),
        (dict(adjacency=np.zeros((2, 2)), pheno_probs=np.eye(2),
              repro_probs=np.ones(2) * 0.1), "no mutational neighbours"),
        (dict(adjacency=np.array([[0.0, 1.0], [1.0, 0.0]]),
              pheno_probs=np.array([[0.5, 0.2], [0.5, 0.5]]),
              repro_probs=np.ones(2) * 0.1), "sum to"),
        (dict(adjacency=np.array([[0.0, 1.0], [1.0, 0.0]]), pheno_probs=np.eye(2),
              repro_probs=np.array([0.1, 1.5])), r"\[0, 1\]"),
        (dict(adjacency=np.array([[0.0, 1.0], [1.0, 0.0]]), pheno_probs=np.eye(2),
              repro_probs=np.array([0.1, np.nan])), "non-finite"),
    ],
)
def test_landscape_validation_rejects_bad_input(kwargs, message):
    """Malformed landscapes fail loudly at construction, not silently later.

    The nan case is the one that used to bite: an unset command-line override
    wrote ``None`` into the reproduction rates and the run produced nans.
    """
    with pytest.raises(ValueError, match=message):
        Landscape(**kwargs)


def test_landscape_arrays_are_read_only():
    with pytest.raises(ValueError):
        LANDSCAPE.pheno_probs[0, 0] = 0.5


def test_with_pheno_probs_does_not_mutate_the_original():
    modified = LANDSCAPE.with_pheno_probs(1, [0.9, 0.1])
    assert LANDSCAPE.pheno_probs[1, 0] == 0.7
    assert modified.pheno_probs[1, 0] == 0.9
