"""Reference implementation of the original ProSeD inner loop.

This is a faithful transcription of the simulation loop as it appeared in the
pre-1.0 scripts (``bridge_sims.py`` and its five near-identical siblings),
using the same NumPy calls in the same order:

* ``np.random.binomial(1, r[phenotypes])`` for reproduction,
* a Python ``for`` loop with ``np.random.choice(neighbors)`` for mutation,
* a Python list comprehension of ``np.random.choice(Np, p=...)`` for
  phenotype re-expression,
* ``np.random.choice(size=N, replace=False)`` for dilution.

It exists only so the tests can show that the vectorised implementation in
:mod:`propgen.prosed` draws from the same distribution. It is deliberately not
part of the installed package.
"""

from __future__ import annotations

import numpy as np


def reference_cycle(
    genotypes: np.ndarray,
    phenotypes: np.ndarray,
    adjacency: np.ndarray,
    pheno_probs: np.ndarray,
    repro_probs: np.ndarray,
    *,
    pop_size: int,
    mutation_rate: float,
    generations_per_cycle: int = 10,
    offspring_per_division: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """One dilution cycle, written the original way."""
    G = genotypes.copy()
    P = phenotypes.copy()
    Np = pheno_probs.shape[1]

    for _ in range(generations_per_cycle):
        chosen_to_reproduce = np.where(np.random.binomial(1, repro_probs[P], size=len(P)))[0]
        offspring_genotypes = np.repeat(G[chosen_to_reproduce], offspring_per_division)

        chosen_to_mutate = np.where(
            np.random.binomial(1, mutation_rate, size=len(offspring_genotypes))
        )[0]
        for i in chosen_to_mutate:
            neighbors = np.nonzero(adjacency[offspring_genotypes[i]])[0]
            offspring_genotypes[i] = np.random.choice(neighbors)

        probs = pheno_probs[offspring_genotypes]
        offspring_phenotypes = np.array(
            [np.random.choice(Np, p=probs[i]) for i in range(len(offspring_genotypes))],
            dtype=np.int64,
        )

        G = np.concatenate([G, offspring_genotypes])
        P = np.concatenate([P, offspring_phenotypes])

    select_ids = np.random.choice(len(G), size=pop_size, replace=False)
    return G[select_ids].astype(np.int64), P[select_ids].astype(np.int64)


def reference_run(
    adjacency: np.ndarray,
    pheno_probs: np.ndarray,
    repro_probs: np.ndarray,
    *,
    n_cycles: int,
    pop_size: int,
    mutation_rate: float,
    generations_per_cycle: int = 10,
    init_genotype: int = 0,
    init_phenotype: int = 0,
) -> np.ndarray:
    """Run the reference loop, returning ``(Ng, Np, n_cycles)`` frequencies."""
    Ng, Np = pheno_probs.shape
    G = np.full(pop_size, init_genotype, dtype=np.int64)
    P = np.full(pop_size, init_phenotype, dtype=np.int64)
    freqs = np.zeros((Ng, Np, n_cycles))

    for t in range(n_cycles):
        G, P = reference_cycle(
            G, P, adjacency, pheno_probs, repro_probs,
            pop_size=pop_size,
            mutation_rate=mutation_rate,
            generations_per_cycle=generations_per_cycle,
        )
        combined = np.column_stack((G, P))
        unique, counts = np.unique(combined, axis=0, return_counts=True)
        for k in range(len(unique)):
            freqs[unique[k][0], unique[k][1], t] = counts[k] / pop_size

    return freqs
