"""ProSeD simulations converge to the analytically predicted equilibrium.

This is the central check on the package: the simulation algorithm and the
analytic theory are independent derivations of the same model, so agreement
between them validates both. It is also the check that most directly supports
the paper's claims, which is why it runs in continuous integration on every
commit.
"""

from __future__ import annotations

import numpy as np
import pytest

from propgen import Landscape, equilibrium, simulate

# Equilibrium is read off a finite population, so agreement is limited by
# sampling noise of order 1/sqrt(pop_size * window). At pop_size=20000
# averaged over 100 cycles the observed error is ~0.001; 0.02 leaves ample
# headroom without being vacuous.
TOLERANCE = 0.02


def buoy_landscape(m: float) -> Landscape:
    """Two-genotype buoying landscape with genotype 1's map set by ``m``."""
    return Landscape(
        adjacency=np.array([[0.0, 1.0], [1.0, 0.0]]),
        pheno_probs=np.array([[0.4, 0.6], [m, 1.0 - m]]),
        repro_probs=np.array([0.009, 0.002]),
    )


@pytest.mark.parametrize("m", [0.0, 0.3, 0.5, 0.7, 1.0])
def test_simulation_matches_analytic_equilibrium(m):
    """Simulated equilibrium frequencies match calc_f_eq across the sweep."""
    landscape = buoy_landscape(m)
    f_eq, _ = equilibrium(landscape, mutation_rate=0.1)

    result = simulate(
        landscape,
        n_cycles=400,
        pop_size=20_000,
        mutation_rate=0.1,
        seed=1,
        compute_theory=False,
    )
    observed = result.final_frequencies(last=100).ravel()

    assert np.max(np.abs(observed - f_eq)) < TOLERANCE, (
        f"m={m}: simulated {np.round(observed, 4)} vs analytic {np.round(f_eq, 4)}"
    )


def test_simulation_matches_analytic_mean_fitness():
    """Simulated mean fitness matches the Perron eigenvalue within 5%."""
    landscape = buoy_landscape(0.5)
    _, Xbar_theory = equilibrium(landscape, mutation_rate=0.1)

    result = simulate(
        landscape,
        n_cycles=400,
        pop_size=20_000,
        mutation_rate=0.1,
        seed=2,
        compute_theory=False,
    )
    observed = result.mean_fitness()[-100:].mean()

    assert observed == pytest.approx(Xbar_theory, rel=0.05), (
        f"simulated mean fitness {observed:.6g} vs analytic {Xbar_theory:.6g}"
    )


def test_deterministic_map_reduces_to_classical_case():
    """With an identity genotype-phenotype map the fitter genotype fixes.

    This is the classical deterministic limit, and a sanity check that the
    probabilistic machinery does not disturb it.
    """
    landscape = Landscape(
        adjacency=np.array([[0.0, 1.0], [1.0, 0.0]]),
        pheno_probs=np.eye(2),
        repro_probs=np.array([0.002, 0.009]),
    )
    result = simulate(
        landscape,
        n_cycles=300,
        pop_size=10_000,
        mutation_rate=0.0,
        seed=3,
        compute_theory=False,
    )
    final = result.final_frequencies(last=20)

    # Genotype 1 expresses the higher-fitness phenotype and should dominate.
    assert final[1, 1] > 0.95
    # Off-diagonal pairs are unreachable under a deterministic map.
    assert final[0, 1] == 0.0
    assert final[1, 0] == 0.0
