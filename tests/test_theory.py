"""The analytic equilibrium: correctness, normalisation and failure modes."""

from __future__ import annotations

import numpy as np
import pytest

from propgen import (
    Landscape,
    NoEquilibriumError,
    calc_f_eq,
    coexistence_ordering,
    equilibrium,
    evolution_operator,
)

LANDSCAPE = Landscape(
    adjacency=np.array([[0.0, 1.0], [1.0, 0.0]]),
    pheno_probs=np.array([[0.4, 0.6], [0.7, 0.3]]),
    repro_probs=np.array([0.009, 0.002]),
)


def test_equilibrium_is_normalised_real_and_non_negative():
    f_eq, Xbar = equilibrium(LANDSCAPE, mutation_rate=0.1)
    assert f_eq.shape == (4,)
    assert f_eq.sum() == pytest.approx(1.0)
    assert np.all(f_eq >= -1e-12)
    assert np.isreal(Xbar)
    assert Xbar > 0


def test_equilibrium_is_an_eigenvector_of_the_evolution_operator():
    """f_eq and Xbar really do satisfy M f = Xbar f."""
    f_eq, Xbar = equilibrium(LANDSCAPE, mutation_rate=0.1)
    M = evolution_operator(
        LANDSCAPE.pheno_probs, LANDSCAPE.repro_probs, LANDSCAPE.adjacency, 0.1, 1.0
    )
    assert np.allclose(M @ f_eq, Xbar * f_eq, atol=1e-12)


def test_mean_fitness_is_the_dominant_eigenvalue():
    _, Xbar = equilibrium(LANDSCAPE, mutation_rate=0.1)
    M = evolution_operator(
        LANDSCAPE.pheno_probs, LANDSCAPE.repro_probs, LANDSCAPE.adjacency, 0.1, 1.0
    )
    assert Xbar == pytest.approx(np.max(np.linalg.eigvals(M).real))


def test_deterministic_map_equilibrium_is_the_fitter_genotype():
    """With an identity map and no mutation, the fitter genotype takes all."""
    landscape = Landscape(
        adjacency=np.array([[0.0, 1.0], [1.0, 0.0]]),
        pheno_probs=np.eye(2),
        repro_probs=np.array([0.002, 0.009]),
    )
    f_eq, Xbar = equilibrium(landscape, mutation_rate=0.0)
    assert f_eq[3] == pytest.approx(1.0)      # pair (1, 1)
    assert Xbar == pytest.approx(0.009)


def test_offspring_per_division_scales_mean_fitness():
    _, x1 = equilibrium(LANDSCAPE, mutation_rate=0.1, offspring_per_division=1)
    _, x3 = equilibrium(LANDSCAPE, mutation_rate=0.1, offspring_per_division=3)
    assert x3 == pytest.approx(3 * x1)


def test_isolated_genotype_raises_rather_than_dividing_by_zero():
    with pytest.raises(ValueError, match="no mutational neighbours"):
        calc_f_eq(np.eye(2), np.array([0.1, 0.2]), np.zeros((2, 2)), mu=0.1)


def test_legacy_keyword_names_still_work():
    """Code written against the pre-1.0 function keeps working."""
    f_eq, Xbar = calc_f_eq(
        pheno_probs=LANDSCAPE.pheno_probs,
        repro_probs=LANDSCAPE.repro_probs,
        A=LANDSCAPE.adjacency,
        mu=0.1,
        c=1,
    )
    expected, expected_x = equilibrium(LANDSCAPE, mutation_rate=0.1)
    assert np.allclose(f_eq, expected)
    assert Xbar == pytest.approx(expected_x)


def test_sparse_and_dense_solvers_agree():
    dense, x_dense = calc_f_eq(
        LANDSCAPE.pheno_probs, LANDSCAPE.repro_probs, LANDSCAPE.adjacency, 0.1, sparse=False
    )
    sparse, x_sparse = calc_f_eq(
        LANDSCAPE.pheno_probs, LANDSCAPE.repro_probs, LANDSCAPE.adjacency, 0.1, sparse=True
    )
    assert np.allclose(dense, sparse, atol=1e-8)
    assert x_dense == pytest.approx(x_sparse)


def test_coexistence_ordering_ranks_pairs_by_frequency():
    ordering = coexistence_ordering(np.array([0.1, 0.4, 0.2, 0.3]))
    assert ordering == (1, 3, 2, 0)


def test_no_equilibrium_error_is_importable():
    """The failure mode has a named exception rather than an IndexError."""
    assert issubclass(NoEquilibriumError, RuntimeError)


def test_phase_diagram_partitions_the_grid():
    """Every grid point is assigned to exactly one known phase."""
    from propgen import phase_diagram

    diagram = phase_diagram(
        LANDSCAPE,
        mutation_rates=np.linspace(0.01, 0.5, 8),
        pheno_prob_values=np.linspace(0.0, 1.0, 8),
        genotype=1,
    )
    assert diagram.phase.shape == (8, 8)
    assert diagram.f_eq.shape == (8, 8, 4)
    assert diagram.phase.min() >= 0
    assert diagram.phase.max() == len(diagram.orderings) - 1
    assert np.allclose(diagram.f_eq.sum(axis=-1), 1.0)


def test_phase_labels_are_consistent_with_the_frequencies():
    """A cell's phase ordering really is the argsort of its equilibrium."""
    from propgen import coexistence_ordering, phase_diagram

    diagram = phase_diagram(
        LANDSCAPE,
        mutation_rates=np.linspace(0.01, 0.4, 5),
        pheno_prob_values=np.linspace(0.1, 0.9, 5),
        genotype=1,
    )
    for i in range(5):
        for j in range(5):
            expected = coexistence_ordering(diagram.f_eq[i, j])
            assert diagram.orderings[diagram.phase[i, j]] == expected


def test_phase_diagram_rejects_more_than_two_phenotypes():
    from propgen import phase_diagram

    wide = Landscape(
        adjacency=np.array([[0.0, 1.0], [1.0, 0.0]]),
        pheno_probs=np.full((2, 3), 1 / 3),
        repro_probs=np.array([0.01, 0.005, 0.002]),
    )
    with pytest.raises(ValueError, match="2 phenotypes"):
        phase_diagram(wide, mutation_rates=[0.1], pheno_prob_values=[0.5])
