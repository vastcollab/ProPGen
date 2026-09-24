"""ProPGen -- Probabilistic Phenotype Genetics.

A framework for evolutionary dynamics under phenotypic uncertainty, where the
map from genotype to phenotype is probabilistic rather than deterministic.

The package provides two things:

``ProSeD`` (Probabilistic Serial Dilution)
    A discrete-time simulation algorithm with overlapping generations,
    phenotypic noise and stochastic phenotype switching. See
    :func:`propgen.simulate`.

The analytic theory
    Exact equilibrium frequencies and mean fitness, obtained from the
    Perron-Frobenius eigenvector of the selection-mutation operator. See
    :func:`propgen.equilibrium`.

Examples
--------
>>> import numpy as np
>>> from propgen import Landscape, simulate, equilibrium
>>> ls = Landscape(
...     adjacency=np.array([[0.0, 1.0], [1.0, 0.0]]),
...     pheno_probs=np.array([[0.4, 0.6], [0.7, 0.3]]),
...     repro_probs=np.array([0.009, 0.002]),
... )
>>> result = simulate(ls, n_cycles=200, pop_size=5000, mutation_rate=0.1, seed=0)
>>> f_eq, mean_fitness = equilibrium(ls, mutation_rate=0.1)

See the repository README for the commands that reproduce each figure in the
accompanying paper.
"""

from __future__ import annotations

__version__ = "1.0.0"

from . import initialization, schedule
from .aggregate import Summary, aggregate_runs, collect_runs, load_summary
from .config import load_config, resolve_config
from .landscape import Landscape, random_erdos_renyi, read_matrix
from .prosed import ProSeD, simulate
from .results import SimResult, load
from .theory import (
    NoEquilibriumError,
    PhaseDiagram,
    calc_f_eq,
    coexistence_ordering,
    equilibrium,
    evolution_operator,
    phase_diagram,
)

__all__ = [
    "__version__",
    # Model
    "Landscape",
    "read_matrix",
    "random_erdos_renyi",
    # Simulation
    "ProSeD",
    "simulate",
    "initialization",
    "schedule",
    # Theory
    "equilibrium",
    "calc_f_eq",
    "evolution_operator",
    "coexistence_ordering",
    "phase_diagram",
    "PhaseDiagram",
    "NoEquilibriumError",
    # Results
    "SimResult",
    "load",
    "Summary",
    "load_summary",
    "collect_runs",
    "aggregate_runs",
    # Config
    "load_config",
    "resolve_config",
]
