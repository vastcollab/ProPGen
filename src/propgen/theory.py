"""Analytic equilibrium of the ProP Gen selection-mutation operator.

The central analytic result of the paper: for a landscape with genotype-to-
phenotype map :math:`\\phi`, phenotype reproduction probabilities :math:`r`,
mutation graph :math:`A` and mutation rate :math:`\\mu`, the equilibrium
frequency vector over genotype-phenotype pairs is the Perron-Frobenius
eigenvector of

.. math::
    M = \\underbrace{\\phi \\odot \\delta_{\\text{block}} \\odot X}_{\\text{selection}}
      + \\underbrace{\\phi \\odot (\\mu L) \\odot X}_{\\text{mutation}}

and the corresponding eigenvalue is the mean fitness :math:`\\bar{X}`.

The mathematics here is unchanged from the implementation used to produce the
published figures; the differences are in error handling (explicit exceptions
rather than an ``IndexError`` or a divide-by-zero) and in the optional sparse
eigensolver path for large landscapes.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import scipy.linalg

from .landscape import Landscape

if TYPE_CHECKING:
    from numpy.typing import ArrayLike

__all__ = [
    "NoEquilibriumError",
    "calc_f_eq",
    "equilibrium",
    "coexistence_ordering",
    "evolution_operator",
    "PhaseDiagram",
    "phase_diagram",
]

# Landscapes larger than this switch to a sparse eigensolver by default;
# the dense np.linalg.eig is O((Ng*Np)^3) in both time and memory.
_SPARSE_THRESHOLD = 2000


class NoEquilibriumError(RuntimeError):
    """Raised when no single-signed (Perron-Frobenius) eigenvector exists.

    In practice this means the operator is reducible -- typically because the
    landscape has a genotype-phenotype pair that cannot be reached from, or
    cannot reach, the rest of the state space.
    """


def evolution_operator(
    pheno_probs: np.ndarray,
    repro_probs: np.ndarray,
    adjacency: np.ndarray,
    mutation_rate: float,
    offspring_per_division: float = 1.0,
) -> np.ndarray:
    """Build the ``(Ng*Np, Ng*Np)`` selection-plus-mutation operator.

    Exposed separately so the operator can be inspected or reused (for
    instance when sweeping a single parameter over a grid) without repeating
    the eigendecomposition.
    """
    Ng, Np = pheno_probs.shape
    X = repro_probs * offspring_per_division

    # Fitness broadcast across the flattened (genotype, phenotype) index.
    Xk = np.outer(np.ones(Ng * Np), np.ndarray.flatten(np.outer(np.ones(Ng), X)))

    # Selection acts only within a genotype: an individual re-expresses its
    # phenotype at division but does not change genotype.
    block_dirac = scipy.linalg.block_diag(*([np.ones((Np, Np))] * Ng))
    phi_mat = np.outer(np.reshape(pheno_probs, (Ng * Np, 1)), np.ones(Ng * Np))
    selection_mat = phi_mat * block_dirac * Xk

    # Mutation: uniform over mutational neighbours, with matching outflow on
    # the diagonal so that probability is conserved.
    n_neighbors = np.sum(adjacency, axis=0)
    inverse_neighbor_mat = adjacency / n_neighbors
    mu_rate_mat = mutation_rate * inverse_neighbor_mat - np.diag(
        np.sum(mutation_rate * inverse_neighbor_mat, axis=0)
    )
    expanded_mu_mat = np.kron(mu_rate_mat, np.ones((Np, Np)))
    complete_mutation_mat = phi_mat * expanded_mu_mat * Xk

    return selection_mat + complete_mutation_mat


def calc_f_eq(
    pheno_probs: np.ndarray,
    repro_probs: np.ndarray,
    A: np.ndarray,
    mu: float,
    c: float = 1.0,
    *,
    sparse: bool | None = None,
) -> tuple[np.ndarray, float]:
    """Equilibrium frequencies and mean fitness of a ProP Gen model.

    Parameters
    ----------
    pheno_probs
        ``(Ng, Np)`` genotype-to-phenotype probability matrix, rows summing to 1.
    repro_probs
        ``(Np,)`` per-phenotype reproduction probability.
    A
        ``(Ng, Ng)`` symmetric genotype mutation graph. Every genotype must
        have at least one neighbour.
    mu
        Per-offspring mutation probability.
    c
        Offspring produced per division.
    sparse
        Use a sparse eigensolver. ``None`` (default) selects automatically
        based on landscape size.

    Returns
    -------
    f_eq
        ``(Ng*Np,)`` equilibrium frequencies, row-major in ``(g, p)`` and
        normalised to sum to 1.
    Xbar_theory
        Mean fitness at equilibrium (the Perron root).

    Raises
    ------
    ValueError
        If a genotype has no mutational neighbours.
    NoEquilibriumError
        If no single-signed eigenvector exists.

    Notes
    -----
    The argument names are those of the original implementation, so code and
    notebooks written against it keep working. :func:`equilibrium` is the
    :class:`~propgen.landscape.Landscape`-based equivalent.
    """
    pheno_probs = np.asarray(pheno_probs, dtype=np.float64)
    repro_probs = np.asarray(repro_probs, dtype=np.float64).ravel()
    A = np.asarray(A, dtype=np.float64)

    degrees = np.sum(A, axis=0)
    isolated = np.flatnonzero(degrees == 0)
    if isolated.size:
        raise ValueError(
            f"genotypes {isolated.tolist()} have no mutational neighbours, so the "
            "mutation term is undefined (divide by zero)"
        )

    Ng, Np = pheno_probs.shape
    dim = Ng * Np
    rhs_mat = evolution_operator(pheno_probs, repro_probs, A, mu, c)

    if sparse is None:
        sparse = dim > _SPARSE_THRESHOLD

    if sparse:
        from scipy.sparse.linalg import eigs

        eigvals, eigvecs = eigs(rhs_mat, k=1, which="LR")
    else:
        eigvals, eigvecs = np.linalg.eig(rhs_mat)

    # The equilibrium is the eigenvector whose entries all share one sign
    # (Perron-Frobenius). Prefer the one with the largest real eigenvalue,
    # which is the Perron root.
    single_signed = np.flatnonzero(
        np.all(eigvecs.real >= -1e-12, axis=0) | np.all(eigvecs.real <= 1e-12, axis=0)
    )
    if single_signed.size == 0:
        raise NoEquilibriumError(
            "no single-signed eigenvector found; the evolution operator is "
            f"likely reducible. Eigenvalue spectrum: "
            f"{np.array2string(np.sort_complex(eigvals)[::-1][:6], precision=4)}"
        )

    idx = single_signed[np.argmax(eigvals[single_signed].real)]
    f_eq_unnormalized = eigvecs[:, idx]
    Xbar_theory = eigvals[idx]

    imag_mag = np.max(np.abs(np.imag(f_eq_unnormalized)))
    if imag_mag > 1e-9:
        warnings.warn(
            f"equilibrium eigenvector has imaginary components up to {imag_mag:.3e}; "
            "taking the real part",
            RuntimeWarning,
            stacklevel=2,
        )
    if abs(np.imag(Xbar_theory)) > 1e-9:
        warnings.warn(
            f"mean fitness has an imaginary component of {np.imag(Xbar_theory):.3e}; "
            "taking the real part",
            RuntimeWarning,
            stacklevel=2,
        )

    f_eq = np.real(f_eq_unnormalized)
    f_eq = f_eq / np.sum(f_eq)
    return f_eq, float(np.real(Xbar_theory))


def equilibrium(
    landscape: Landscape,
    mutation_rate: float,
    offspring_per_division: float = 1.0,
    *,
    sparse: bool | None = None,
) -> tuple[np.ndarray, float]:
    """:func:`calc_f_eq` taking a :class:`~propgen.landscape.Landscape`.

    Examples
    --------
    >>> import numpy as np
    >>> from propgen import Landscape, equilibrium
    >>> ls = Landscape(
    ...     adjacency=np.array([[0.0, 1.0], [1.0, 0.0]]),
    ...     pheno_probs=np.array([[0.4, 0.6], [0.7, 0.3]]),
    ...     repro_probs=np.array([0.009, 0.002]),
    ... )
    >>> f_eq, Xbar = equilibrium(ls, mutation_rate=0.1)
    >>> float(np.round(f_eq.sum(), 12))
    1.0
    """
    return calc_f_eq(
        landscape.pheno_probs,
        landscape.repro_probs,
        landscape.adjacency,
        mutation_rate,
        offspring_per_division,
        sparse=sparse,
    )


def coexistence_ordering(f_eq: np.ndarray) -> tuple[int, ...]:
    """Rank ordering of genotype-phenotype pairs by equilibrium frequency.

    Returns the flattened ``(g, p)`` indices sorted by decreasing frequency.
    This tuple is the categorical label that indexes the coexistence phase
    diagram: each distinct ordering is one phase.
    """
    return tuple(int(i) for i in np.argsort(-np.asarray(f_eq)))


@dataclass
class PhaseDiagram:
    """Coexistence phases over a two-parameter grid.

    Attributes
    ----------
    mutation_rates, pheno_prob_values
        The grid coordinates, of length ``n_mu`` and ``n_phi``.
    phase
        ``(n_mu, n_phi)`` integer labels indexing :attr:`orderings`.
    orderings
        The distinct rank orderings found, each a tuple of flattened ``(g, p)``
        indices in decreasing equilibrium frequency.
    f_eq
        ``(n_mu, n_phi, Ng*Np)`` equilibrium frequencies at every grid point.
    """

    mutation_rates: np.ndarray
    pheno_prob_values: np.ndarray
    phase: np.ndarray
    orderings: list[tuple[int, ...]]
    f_eq: np.ndarray

    def label(self, index: int, n_phenotypes: int) -> str:
        """LaTeX label for one phase, e.g. ``$f_0^{(1)} > f_0^{(0)} > ...$``."""
        terms = [
            f"$f_{{{i // n_phenotypes}}}^{{({i % n_phenotypes})}}$"
            for i in self.orderings[index]
        ]
        return " > ".join(terms)

    def __repr__(self) -> str:
        return (
            f"PhaseDiagram(grid={self.phase.shape}, "
            f"n_phases={len(self.orderings)})"
        )


def phase_diagram(
    landscape: Landscape,
    *,
    mutation_rates: ArrayLike,
    pheno_prob_values: ArrayLike,
    genotype: int = 1,
    offspring_per_division: float = 1.0,
) -> PhaseDiagram:
    """Map the coexistence phases of a landscape over (mutation rate, phi).

    At each grid point the genotype-to-phenotype distribution of ``genotype``
    is set to ``[v, 1 - v]`` and the analytic equilibrium is computed. Grid
    points are grouped by the *rank ordering* of the resulting
    genotype-phenotype frequencies; each distinct ordering is a phase of
    simultaneous coexistence.

    This is exact and requires no simulation: a 50x50 grid takes under a
    second. Currently limited to two-phenotype landscapes, since the sweep
    parametrises a single row as ``[v, 1 - v]``.

    Examples
    --------
    >>> import numpy as np
    >>> from propgen import Landscape, phase_diagram
    >>> ls = Landscape(
    ...     adjacency=np.array([[0.0, 1.0], [1.0, 0.0]]),
    ...     pheno_probs=np.array([[0.4, 0.6], [0.5, 0.5]]),
    ...     repro_probs=np.array([0.009, 0.002]),
    ... )
    >>> diagram = phase_diagram(
    ...     ls, mutation_rates=np.linspace(0.01, 0.5, 5),
    ...     pheno_prob_values=np.linspace(0.0, 1.0, 5),
    ... )
    >>> diagram.phase.shape
    (5, 5)
    """
    if landscape.n_phenotypes != 2:
        raise ValueError(
            f"phase_diagram sweeps a row as [v, 1 - v] and so needs 2 phenotypes, "
            f"got {landscape.n_phenotypes}"
        )

    mutation_rates = np.asarray(mutation_rates, dtype=np.float64)
    pheno_prob_values = np.asarray(pheno_prob_values, dtype=np.float64)

    dim = landscape.n_genotypes * landscape.n_phenotypes
    phase = np.empty((mutation_rates.size, pheno_prob_values.size), dtype=np.int64)
    all_f_eq = np.empty((mutation_rates.size, pheno_prob_values.size, dim))

    seen: dict[tuple[int, ...], int] = {}
    orderings: list[tuple[int, ...]] = []

    for i, mu in enumerate(mutation_rates):
        for j, v in enumerate(pheno_prob_values):
            local = landscape.with_pheno_probs(genotype, [v, 1.0 - v])
            f_eq, _ = equilibrium(local, mu, offspring_per_division)
            all_f_eq[i, j] = f_eq

            ordering = coexistence_ordering(f_eq)
            if ordering not in seen:
                seen[ordering] = len(orderings)
                orderings.append(ordering)
            phase[i, j] = seen[ordering]

    return PhaseDiagram(
        mutation_rates=mutation_rates,
        pheno_prob_values=pheno_prob_values,
        phase=phase,
        orderings=orderings,
        f_eq=all_f_eq,
    )
