"""Genotype-phenotype landscapes for ProP Gen models.

A :class:`Landscape` bundles the three objects that define a ProP Gen model:

* ``adjacency``    -- the genotype mutation graph, :math:`A`, shape ``(Ng, Ng)``
* ``pheno_probs``  -- the probabilistic genotype-to-phenotype map,
  :math:`\\phi_g^{(p)}`, shape ``(Ng, Np)``, rows summing to 1
* ``repro_probs``  -- the per-phenotype reproduction probability,
  :math:`r_p`, shape ``(Np,)``

Index conventions used throughout the package: genotypes are rows, phenotypes
are columns, and flattened ``(g, p)`` vectors (such as the equilibrium
frequency ``f_eq``) are row-major, so entry ``g * Np + p`` is the pair
``(g, p)``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import ArrayLike

__all__ = ["Landscape", "read_matrix", "random_erdos_renyi"]

# Tolerance for the row-stochasticity check on pheno_probs.
_ROW_SUM_TOL = 1e-9


def read_matrix(path: str | Path) -> np.ndarray:
    """Read a whitespace-delimited numeric matrix.

    Tolerates a UTF-8 byte-order mark, CRLF line endings, trailing whitespace
    and ``#`` comment lines. Always returns a 2-D array, so a single-row file
    (as used by the one-genotype persister model) does not collapse to 1-D.
    """
    arr = np.loadtxt(path, dtype=np.float64, encoding="utf-8-sig", comments="#", ndmin=2)
    return arr


@dataclass(eq=False)
class Landscape:
    """A ProP Gen genotype-phenotype landscape.

    Instances are treated as immutable: the underlying arrays are marked
    read-only, and the ``with_*`` methods return new instances rather than
    mutating in place. This is deliberate -- the parameter sweeps in the paper
    vary single entries of ``pheno_probs`` and ``repro_probs``, and doing that
    by assignment makes it impossible to tell from a call site what model was
    actually simulated.

    Examples
    --------
    >>> import numpy as np
    >>> ls = Landscape(
    ...     adjacency=np.array([[0.0, 1.0], [1.0, 0.0]]),
    ...     pheno_probs=np.array([[0.4, 0.6], [0.7, 0.3]]),
    ...     repro_probs=np.array([0.009, 0.002]),
    ... )
    >>> ls.n_genotypes, ls.n_phenotypes
    (2, 2)
    """

    adjacency: np.ndarray
    pheno_probs: np.ndarray
    repro_probs: np.ndarray

    # Derived structures, precomputed once and reused by the simulation loop.
    _pheno_cdf: np.ndarray = field(init=False, repr=False)
    _degrees: np.ndarray = field(init=False, repr=False)
    _neighbors: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.adjacency = np.ascontiguousarray(self.adjacency, dtype=np.float64)
        self.pheno_probs = np.ascontiguousarray(self.pheno_probs, dtype=np.float64)
        self.repro_probs = np.ascontiguousarray(self.repro_probs, dtype=np.float64).ravel()
        self.validate()

        self._pheno_cdf = np.cumsum(self.pheno_probs, axis=1)
        # Guard against floating-point cumsum ending just below 1.0, which
        # would let inverse-CDF sampling fall off the end of a row.
        self._pheno_cdf[:, -1] = 1.0
        self._degrees, self._neighbors = self._build_neighbor_table()

        for arr in (self.adjacency, self.pheno_probs, self.repro_probs,
                    self._pheno_cdf, self._degrees, self._neighbors):
            arr.flags.writeable = False

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(self) -> None:
        """Raise :class:`ValueError` if the landscape is not well formed.

        Checks performed here are the ones that would otherwise surface far
        from their cause: an isolated genotype becomes a divide-by-zero inside
        the analytic equilibrium calculation, and a ``None`` left in
        ``repro_probs`` by an unset command-line override becomes a silent
        ``nan`` in the trajectory.
        """
        A, pi, r = self.adjacency, self.pheno_probs, self.repro_probs

        if A.ndim != 2 or A.shape[0] != A.shape[1]:
            raise ValueError(f"adjacency must be square 2-D, got shape {A.shape}")
        if pi.ndim != 2:
            raise ValueError(f"pheno_probs must be 2-D (Ng, Np), got shape {pi.shape}")
        if pi.shape[0] != A.shape[0]:
            raise ValueError(
                f"pheno_probs has {pi.shape[0]} genotypes but adjacency has {A.shape[0]}"
            )
        if r.shape[0] != pi.shape[1]:
            raise ValueError(
                f"repro_probs has {r.shape[0]} entries but pheno_probs has "
                f"{pi.shape[1]} phenotypes"
            )

        for name, arr in (("adjacency", A), ("pheno_probs", pi), ("repro_probs", r)):
            if not np.all(np.isfinite(arr)):
                raise ValueError(f"{name} contains non-finite values (nan or inf)")

        if not np.allclose(A, A.T):
            raise ValueError("adjacency must be symmetric")

        degrees = A.sum(axis=1)
        isolated = np.flatnonzero(degrees == 0)
        if isolated.size:
            raise ValueError(
                f"genotypes {isolated.tolist()} have no mutational neighbours; "
                "every genotype needs at least one (use a self-loop for a "
                "single-genotype model)"
            )

        row_sums = pi.sum(axis=1)
        bad = np.flatnonzero(np.abs(row_sums - 1.0) > _ROW_SUM_TOL)
        if bad.size:
            raise ValueError(
                f"pheno_probs rows {bad.tolist()} sum to {row_sums[bad].tolist()}, "
                "expected 1.0 -- each genotype's phenotype distribution must be normalised"
            )
        if np.any(pi < 0):
            raise ValueError("pheno_probs contains negative probabilities")

        if np.any(r < 0) or np.any(r > 1):
            raise ValueError(f"repro_probs must lie in [0, 1], got {r.tolist()}")

    # ------------------------------------------------------------------
    # Shape accessors
    # ------------------------------------------------------------------

    @property
    def n_genotypes(self) -> int:
        return self.pheno_probs.shape[0]

    @property
    def n_phenotypes(self) -> int:
        return self.pheno_probs.shape[1]

    @property
    def shape(self) -> tuple[int, int]:
        """``(n_genotypes, n_phenotypes)``."""
        return self.pheno_probs.shape

    def _build_neighbor_table(self) -> tuple[np.ndarray, np.ndarray]:
        """Padded neighbour lookup enabling branch-free vectorised mutation.

        Returns ``(degrees, neighbors)`` where ``neighbors`` has shape
        ``(Ng, max_degree)``. Row ``g`` holds that genotype's neighbours in its
        first ``degrees[g]`` slots; the remaining slots repeat the first
        neighbour, so an out-of-range index can never select an invalid
        genotype even under floating-point edge cases.
        """
        adj = self.adjacency > 0
        degrees = adj.sum(axis=1).astype(np.int64)
        max_degree = int(degrees.max())
        neighbors = np.zeros((self.n_genotypes, max_degree), dtype=np.int64)
        for g in range(self.n_genotypes):
            nbrs = np.flatnonzero(adj[g])
            neighbors[g, : nbrs.size] = nbrs
            neighbors[g, nbrs.size:] = nbrs[0]
        return degrees, neighbors

    # ------------------------------------------------------------------
    # Functional updates (replace in-place `pi[1][0] = ...` style overrides)
    # ------------------------------------------------------------------

    def with_pheno_probs(self, genotype: int, probs: ArrayLike) -> Landscape:
        """Return a copy with ``pheno_probs[genotype]`` replaced by ``probs``."""
        pi = self.pheno_probs.copy()
        pi[genotype] = np.asarray(probs, dtype=np.float64)
        return Landscape(self.adjacency.copy(), pi, self.repro_probs.copy())

    def with_repro_prob(self, phenotype: int, value: float) -> Landscape:
        """Return a copy with ``repro_probs[phenotype]`` replaced by ``value``."""
        r = self.repro_probs.copy()
        r[phenotype] = float(value)
        return Landscape(self.adjacency.copy(), self.pheno_probs.copy(), r)

    def with_repro_probs(self, values: ArrayLike) -> Landscape:
        """Return a copy with ``repro_probs`` replaced wholesale."""
        return Landscape(
            self.adjacency.copy(),
            self.pheno_probs.copy(),
            np.asarray(values, dtype=np.float64),
        )

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_files(
        cls,
        adjacency: str | Path,
        pheno_probs: str | Path,
        repro_probs: str | Path,
    ) -> Landscape:
        """Build a landscape from three whitespace-delimited text files."""
        return cls(
            adjacency=read_matrix(adjacency),
            pheno_probs=read_matrix(pheno_probs),
            repro_probs=read_matrix(repro_probs).ravel(),
        )

    def __repr__(self) -> str:
        return (
            f"Landscape(n_genotypes={self.n_genotypes}, "
            f"n_phenotypes={self.n_phenotypes}, "
            f"repro_probs={np.array2string(self.repro_probs, precision=5)})"
        )


def random_erdos_renyi(
    n_genotypes: int,
    edge_prob: float = 0.3,
    n_phenotypes: int = 2,
    *,
    rng: np.random.Generator | int | None = None,
) -> Landscape:
    """Random connected Erdos-Renyi genotype graph with random phenotype maps.

    Provided for exploration and testing. The published experiments all use
    explicit landscapes read from ``data/``.
    """
    rng = np.random.default_rng(rng)
    for _ in range(1000):
        upper = rng.random((n_genotypes, n_genotypes)) < edge_prob
        adj = np.triu(upper, k=1).astype(np.float64)
        adj = adj + adj.T
        if _is_connected(adj):
            break
    else:
        raise RuntimeError(
            f"could not sample a connected graph with n={n_genotypes}, p={edge_prob}"
        )

    pi = rng.random((n_genotypes, n_phenotypes))
    pi /= pi.sum(axis=1, keepdims=True)
    r = rng.uniform(1e-4, 1e-2, size=n_phenotypes)
    return Landscape(adj, pi, r)


def _is_connected(adj: np.ndarray) -> bool:
    """Breadth-first reachability check; avoids a networkx dependency."""
    n = adj.shape[0]
    seen = np.zeros(n, dtype=bool)
    seen[0] = True
    frontier = [0]
    while frontier:
        node = frontier.pop()
        for nbr in np.flatnonzero(adj[node] > 0):
            if not seen[nbr]:
                seen[nbr] = True
                frontier.append(int(nbr))
    return bool(seen.all())
