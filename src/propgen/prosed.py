"""ProSeD -- Probabilistic Serial Dilution.

The discrete-time simulation algorithm introduced in the paper. Each dilution
cycle runs ``generations_per_cycle`` rounds of

1. **reproduce** -- each individual divides with probability :math:`r_p` set by
   its *current phenotype*, producing ``offspring_per_division`` offspring;
2. **mutate** -- each offspring moves to a uniformly chosen mutational
   neighbour with probability :math:`\\mu`;
3. **re-express** -- each offspring independently draws its phenotype from
   :math:`\\phi_g` given its genotype,

with offspring accumulating alongside their parents, so generations overlap.
The cycle ends with a **serial dilution**: the population is subsampled
uniformly without replacement back to ``pop_size``, mirroring a laboratory
passaging experiment.

Phenotype is re-drawn at every division rather than inherited, which is what
distinguishes this from a Wright-Fisher branching process and what makes it
able to represent phenotypic uncertainty.

Examples
--------
>>> import numpy as np
>>> from propgen import Landscape, simulate
>>> ls = Landscape(
...     adjacency=np.array([[0.0, 1.0], [1.0, 0.0]]),
...     pheno_probs=np.array([[0.4, 0.6], [0.7, 0.3]]),
...     repro_probs=np.array([0.009, 0.002]),
... )
>>> res = simulate(ls, n_cycles=50, pop_size=1000, mutation_rate=0.1, seed=0)
>>> res.frequencies.shape
(2, 2, 50)
"""

from __future__ import annotations

import platform
import warnings
from collections.abc import Sequence
from datetime import datetime, timezone
from typing import Literal

import numpy as np

from . import initialization
from . import schedule as schedule_mod
from .landscape import Landscape
from .results import SimResult
from .theory import NoEquilibriumError, equilibrium

__all__ = ["ProSeD", "simulate"]

# Cap on the number of elements materialised by the vectorised categorical
# sampler, so a landscape with many phenotypes cannot exhaust memory.
_CHUNK_ELEMS = 10_000_000


def _sample_categorical(
    rng: np.random.Generator, cdf: np.ndarray, rows: np.ndarray
) -> np.ndarray:
    """Draw one categorical sample per entry of ``rows`` by inverse CDF.

    ``cdf`` is the row-wise cumulative sum of the genotype-to-phenotype matrix;
    ``rows`` holds the genotype of each offspring. Equivalent to calling
    ``rng.choice(Np, p=pheno_probs[g])`` once per offspring, but as a single
    broadcast comparison.
    """
    m = rows.size
    Np = cdf.shape[1]
    if m == 0:
        return np.empty(0, dtype=np.int64)
    if m * Np <= _CHUNK_ELEMS:
        return (rng.random(m)[:, None] < cdf[rows]).argmax(axis=1).astype(np.int64)

    out = np.empty(m, dtype=np.int64)
    step = max(1, _CHUNK_ELEMS // Np)
    for lo in range(0, m, step):
        hi = min(m, lo + step)
        out[lo:hi] = (rng.random(hi - lo)[:, None] < cdf[rows[lo:hi]]).argmax(axis=1)
    return out


class ProSeD:
    """Stateful ProSeD simulator.

    Use this when you want to inspect or intervene on the population between
    cycles. For a single run to completion, :func:`simulate` is simpler.

    Parameters
    ----------
    landscape
        The genotype-phenotype landscape to simulate.
    pop_size
        Population size, restored by dilution at the end of every cycle.
    mutation_rate
        Per-offspring probability of moving to a mutational neighbour.
    generations_per_cycle
        Rounds of reproduction between dilutions. Generations overlap: parents
        are not removed when they divide.
    offspring_per_division
        Offspring produced per division event.
    init
        Initialization strategy; see :mod:`propgen.initialization`.
    seed
        Seed for the internal :class:`numpy.random.Generator`. The global
        NumPy random state is never used or modified.
    """

    def __init__(
        self,
        landscape: Landscape,
        *,
        pop_size: int,
        mutation_rate: float,
        generations_per_cycle: int = 10,
        offspring_per_division: int = 1,
        init: initialization.Init | None = None,
        seed: int | Sequence[int] | np.random.Generator | None = None,
    ) -> None:
        if pop_size < 1:
            raise ValueError(f"pop_size must be positive, got {pop_size}")
        if not 0.0 <= mutation_rate <= 1.0:
            raise ValueError(f"mutation_rate must lie in [0, 1], got {mutation_rate}")
        if generations_per_cycle < 1:
            raise ValueError(
                f"generations_per_cycle must be positive, got {generations_per_cycle}"
            )
        if offspring_per_division < 1:
            raise ValueError(
                f"offspring_per_division must be positive, got {offspring_per_division}"
            )

        self._landscape = landscape
        self.pop_size = int(pop_size)
        self.mutation_rate = float(mutation_rate)
        self.generations_per_cycle = int(generations_per_cycle)
        self.offspring_per_division = int(offspring_per_division)
        self.cycle = 0

        self._seed = seed
        self._rng = seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)

        self._init = init if init is not None else initialization.uniform_over_support()
        self.genotypes, self.phenotypes = self._init(landscape, self.pop_size, self._rng)
        self.genotypes = self.genotypes.astype(np.int64, copy=False)
        self.phenotypes = self.phenotypes.astype(np.int64, copy=False)

    # ------------------------------------------------------------------
    # State
    # ------------------------------------------------------------------

    @property
    def landscape(self) -> Landscape:
        return self._landscape

    def set_landscape(self, landscape: Landscape) -> None:
        """Swap the landscape in force, e.g. on an environment switch.

        The population carries over unchanged. The new landscape must have the
        same number of genotypes and phenotypes as the old one.
        """
        if landscape.shape != self._landscape.shape:
            raise ValueError(
                f"cannot switch from a {self._landscape.shape} landscape to a "
                f"{landscape.shape} one; the population indices would not carry over"
            )
        self._landscape = landscape

    @property
    def frequencies(self) -> np.ndarray:
        """Current ``(Ng, Np)`` frequency matrix, summing to 1."""
        Ng, Np = self._landscape.shape
        counts = np.bincount(self.genotypes * Np + self.phenotypes, minlength=Ng * Np)
        return counts.reshape(Ng, Np) / self.genotypes.size

    # ------------------------------------------------------------------
    # Dynamics
    # ------------------------------------------------------------------

    def step(self, n: int = 1) -> None:
        """Advance the population by ``n`` dilution cycles."""
        for _ in range(n):
            self._one_cycle()

    def _one_cycle(self) -> None:
        rng = self._rng
        ls = self._landscape
        cdf = ls._pheno_cdf
        degrees, neighbors = ls._degrees, ls._neighbors
        r = ls.repro_probs
        mu = self.mutation_rate
        c = self.offspring_per_division

        G, P = self.genotypes, self.phenotypes

        for _ in range(self.generations_per_cycle):
            # Reproduce: one Bernoulli trial per individual, at the rate set
            # by its current phenotype.
            parents = np.flatnonzero(rng.random(G.size) < r[P])
            if parents.size == 0:
                continue
            offspring_g = np.repeat(G[parents], c)

            # Mutate: one uniform draw per mutant, indexed into the padded
            # neighbour table.
            mutants = np.flatnonzero(rng.random(offspring_g.size) < mu)
            if mutants.size:
                gm = offspring_g[mutants]
                choice = (rng.random(mutants.size) * degrees[gm]).astype(np.int64)
                offspring_g[mutants] = neighbors[gm, choice]

            # Re-express: each offspring draws its phenotype afresh.
            offspring_p = _sample_categorical(rng, cdf, offspring_g)

            G = np.concatenate([G, offspring_g])
            P = np.concatenate([P, offspring_p])

        # Serial dilution back to pop_size. argpartition selects a uniform
        # random subset without materialising a full permutation; the set of
        # survivors is distributed identically to sampling without
        # replacement, though their order differs.
        if G.size > self.pop_size:
            keep = np.argpartition(rng.random(G.size), self.pop_size)[: self.pop_size]
            G, P = G[keep], P[keep]

        self.genotypes, self.phenotypes = G, P
        self.cycle += 1

    # ------------------------------------------------------------------
    # Running to completion
    # ------------------------------------------------------------------

    def run(
        self,
        n_cycles: int,
        *,
        record_every: int = 1,
        progress: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Run ``n_cycles`` cycles, returning ``(frequencies, cycle_indices)``."""
        Ng, Np = self._landscape.shape
        recorded = range(0, n_cycles, record_every)
        freqs = np.zeros((Ng, Np, len(recorded)), dtype=np.float32)
        cycle_index = np.fromiter(recorded, dtype=np.int32, count=len(recorded))

        iterator = range(n_cycles)
        if progress:
            from tqdm import tqdm

            iterator = tqdm(iterator, desc="ProSeD", unit="cycle")

        k = 0
        for t in iterator:
            self._one_cycle()
            if t % record_every == 0:
                freqs[:, :, k] = self.frequencies
                k += 1

        return freqs, cycle_index


def simulate(
    landscape: Landscape | Sequence[Landscape],
    *,
    n_cycles: int,
    pop_size: int,
    mutation_rate: float,
    generations_per_cycle: int = 10,
    offspring_per_division: int = 1,
    init: initialization.Init | None = None,
    schedule: schedule_mod.Schedule | None = None,
    seed: int | Sequence[int] | np.random.Generator | None = None,
    record: Literal["frequencies", "frequencies+state"] = "frequencies+state",
    record_every: int = 1,
    compute_theory: bool = True,
    progress: bool = False,
) -> SimResult:
    """Run a ProSeD simulation and return a :class:`~propgen.results.SimResult`.

    Parameters
    ----------
    landscape
        A single :class:`~propgen.landscape.Landscape`, or a sequence of them
        together with ``schedule`` to switch environments during the run.
    n_cycles
        Number of dilution cycles. Note this is *not* the number of
        generations: the run spans ``n_cycles * generations_per_cycle``
        rounds of reproduction.
    pop_size
        Population size, restored by dilution at the end of each cycle.
    mutation_rate
        Per-offspring mutation probability.
    schedule
        Maps cycle index to environment index. Required when ``landscape`` is
        a sequence; see :mod:`propgen.schedule`.
    seed
        Seed for this run. For a parameter sweep, pass ``[master_seed, trial]``
        so that any single trial can be re-run in isolation and reproduce
        exactly.
    record
        ``"frequencies+state"`` also stores the final population arrays.
    record_every
        Record frequencies every ``k`` cycles. The recorded cycle indices are
        stored alongside, so thinned output stays self-describing.
    compute_theory
        Attach the analytic equilibrium and mean fitness. Skipped
        automatically, with a warning, if the operator has no equilibrium.

    Returns
    -------
    SimResult
    """
    landscapes = [landscape] if isinstance(landscape, Landscape) else list(landscape)
    if not landscapes:
        raise ValueError("at least one landscape is required")

    if len(landscapes) > 1 and schedule is None:
        raise ValueError(
            f"{len(landscapes)} landscapes were given but no schedule; pass e.g. "
            "schedule=propgen.schedule.switch_at(100) to say when to switch"
        )
    if schedule is None:
        schedule = schedule_mod.constant()

    sim = ProSeD(
        landscapes[0],
        pop_size=pop_size,
        mutation_rate=mutation_rate,
        generations_per_cycle=generations_per_cycle,
        offspring_per_division=offspring_per_division,
        init=init,
        seed=seed,
    )

    Ng, Np = landscapes[0].shape
    recorded = range(0, n_cycles, record_every)
    freqs = np.zeros((Ng, Np, len(recorded)), dtype=np.float32)
    cycle_index = np.fromiter(recorded, dtype=np.int32, count=len(recorded))

    iterator = range(n_cycles)
    if progress:
        from tqdm import tqdm

        iterator = tqdm(iterator, desc="ProSeD", unit="cycle")

    current_env = 0
    k = 0
    for t in iterator:
        env = schedule(t)
        if env != current_env:
            if env >= len(landscapes):
                raise ValueError(
                    f"schedule requested environment {env} at cycle {t}, but only "
                    f"{len(landscapes)} landscape(s) were provided"
                )
            sim.set_landscape(landscapes[env])
            current_env = env

        sim._one_cycle()
        if t % record_every == 0:
            freqs[:, :, k] = sim.frequencies
            k += 1

    f_eq: np.ndarray | None = None
    Xbar: float | None = None
    if compute_theory:
        try:
            f_eq, Xbar = equilibrium(landscapes[0], mutation_rate, offspring_per_division)
        except (NoEquilibriumError, ValueError) as exc:
            warnings.warn(
                f"analytic equilibrium unavailable for this landscape: {exc}",
                RuntimeWarning,
                stacklevel=2,
            )

    params = {
        "pop_size": pop_size,
        "n_cycles": n_cycles,
        "mutation_rate": mutation_rate,
        "generations_per_cycle": generations_per_cycle,
        "offspring_per_division": offspring_per_division,
        "record_every": record_every,
        "seed": list(seed) if isinstance(seed, (list, tuple)) else seed,
        "init": initialization.describe(sim._init),
        "schedule": schedule_mod.describe(schedule),
        "propgen_version": _version(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }

    keep_state = record == "frequencies+state"
    return SimResult(
        frequencies=freqs,
        cycles=cycle_index,
        landscapes=landscapes,
        genotypes=sim.genotypes.astype(np.int32) if keep_state else None,
        phenotypes=sim.phenotypes.astype(np.int32) if keep_state else None,
        f_eq=f_eq,
        Xbar_theory=Xbar,
        params=params,
    )


def _version() -> str:
    from . import __version__

    return __version__
