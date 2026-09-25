"""Simulation results and their on-disk format.

The canonical format is compressed NumPy ``.npz``. Pickle is supported for
reading only, so that output produced by the original scripts can still be
aggregated.

``.npz`` is the public format because unpickling executes arbitrary code,
which is not a reasonable thing to ask of anyone reproducing a paper; because
NumPy pickles are not guaranteed to survive a major-version change; and
because ``.npz`` is readable from MATLAB, R and Julia. Metadata travels inside
the archive as a JSON string, so results always load with
``allow_pickle=False``.
"""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .landscape import Landscape

__all__ = ["SimResult", "load"]

_FORMAT_VERSION = 1


@dataclass
class SimResult:
    """The output of a ProSeD run.

    Attributes
    ----------
    frequencies
        ``(Ng, Np, n_recorded)`` frequency of each genotype-phenotype pair,
        recorded after each dilution.
    cycles
        ``(n_recorded,)`` the dilution-cycle index each column corresponds to.
        Stored explicitly so that thinned output is self-describing.
    genotypes, phenotypes
        The final population, or ``None`` if it was not recorded.
    landscapes
        The landscape(s) simulated. More than one entry means the run used an
        environment schedule.
    f_eq, Xbar_theory
        Analytic equilibrium and mean fitness, when computable.
    params
        Run parameters and provenance (seed, population size, package version,
        git commit, timestamp).
    """

    frequencies: np.ndarray
    cycles: np.ndarray
    landscapes: list[Landscape]
    genotypes: np.ndarray | None = None
    phenotypes: np.ndarray | None = None
    f_eq: np.ndarray | None = None
    Xbar_theory: float | None = None
    params: dict = field(default_factory=dict)

    @property
    def landscape(self) -> Landscape:
        """The initial (or only) landscape."""
        return self.landscapes[0]

    @property
    def shape(self) -> tuple[int, int]:
        """``(n_genotypes, n_phenotypes)``."""
        return self.frequencies.shape[:2]

    def final_frequencies(self, last: int = 100) -> np.ndarray:
        """Mean frequencies over the final ``last`` recorded cycles.

        Averaging over a window is how the paper reads off equilibrium
        frequencies from a finite population, where the trajectory fluctuates
        around the analytic value rather than settling on it.
        """
        window = min(last, self.frequencies.shape[-1])
        return self.frequencies[..., -window:].mean(axis=-1)

    def mean_fitness(self, environment: int = 0) -> np.ndarray:
        """Population mean fitness at each recorded cycle."""
        ls = self.landscapes[environment]
        c = self.params.get("offspring_per_division", 1)
        X = ls.repro_probs * c
        return np.einsum("gpt,p->t", self.frequencies, X)

    # ------------------------------------------------------------------
    # I/O
    # ------------------------------------------------------------------

    def to_npz(self, path: str | Path) -> Path:
        """Write to compressed ``.npz``. Returns the path written."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        arrays: dict[str, np.ndarray] = {
            "frequencies": np.asarray(self.frequencies, dtype=np.float32),
            "cycles": np.asarray(self.cycles, dtype=np.int32),
            "adjacency": self.landscapes[0].adjacency,
            "pheno_probs": np.stack([ls.pheno_probs for ls in self.landscapes]),
            "repro_probs": np.stack([ls.repro_probs for ls in self.landscapes]),
        }
        if self.genotypes is not None:
            arrays["genotypes"] = np.asarray(self.genotypes, dtype=np.int32)
        if self.phenotypes is not None:
            arrays["phenotypes"] = np.asarray(self.phenotypes, dtype=np.int32)
        if self.f_eq is not None:
            arrays["f_eq"] = np.asarray(self.f_eq, dtype=np.float64)

        meta = dict(self.params)
        meta["format_version"] = _FORMAT_VERSION
        if self.Xbar_theory is not None:
            meta["Xbar_theory"] = float(self.Xbar_theory)
        arrays["meta"] = np.array(json.dumps(meta, default=str))

        np.savez_compressed(path, **arrays)
        return path

    @classmethod
    def from_npz(cls, path: str | Path) -> SimResult:
        """Read a result written by :meth:`to_npz`."""
        with np.load(path, allow_pickle=False) as z:
            meta = json.loads(str(z["meta"]))
            adjacency = z["adjacency"]
            pheno = z["pheno_probs"]
            repro = z["repro_probs"]
            landscapes = [
                Landscape(adjacency, pheno[i], repro[i]) for i in range(pheno.shape[0])
            ]
            return cls(
                frequencies=z["frequencies"],
                cycles=z["cycles"],
                landscapes=landscapes,
                genotypes=z["genotypes"] if "genotypes" in z else None,
                phenotypes=z["phenotypes"] if "phenotypes" in z else None,
                f_eq=z["f_eq"] if "f_eq" in z else None,
                Xbar_theory=meta.pop("Xbar_theory", None),
                params=meta,
            )

    @classmethod
    def from_legacy_pickle(cls, path: str | Path) -> SimResult:
        """Read output produced by the original ``*_sims.py`` scripts.

        Those scripts pickled a plain dict. This reader maps it onto
        :class:`SimResult` so the simulation output behind the published
        figures can be aggregated without re-running anything. It handles both
        the standard layout and the persister variant, which carries
        ``*_probs_1`` / ``*_probs_2`` for its two environments and omits the
        analytic equilibrium.
        """
        with open(path, "rb") as fh:
            d = pickle.load(fh)

        adjacency = np.atleast_2d(np.asarray(d["A"], dtype=np.float64))
        freq = np.asarray(d["freq_timeseries"], dtype=np.float32)

        pheno_list, repro_list = [], []
        if "pheno_probs" in d:
            pheno_list.append(np.atleast_2d(np.asarray(d["pheno_probs"], dtype=np.float64)))
        for key in ("pheno_probs_1", "pheno_probs_2"):
            if key in d:
                pheno_list.append(np.atleast_2d(np.asarray(d[key], dtype=np.float64)))
        if "repro_probs" in d:
            repro_list.append(np.asarray(d["repro_probs"], dtype=np.float64).ravel())
        for key in ("repro_probs_1", "repro_probs_2"):
            if key in d:
                repro_list.append(np.asarray(d[key], dtype=np.float64).ravel())

        # Broadcast whichever of the two varies across environments.
        n_env = max(len(pheno_list), len(repro_list), 1)
        pheno_list = pheno_list or [np.eye(freq.shape[0], freq.shape[1])]
        repro_list = repro_list or [np.ones(freq.shape[1])]
        if len(pheno_list) < n_env:
            pheno_list = [pheno_list[0]] * n_env
        if len(repro_list) < n_env:
            repro_list = [repro_list[0]] * n_env

        landscapes = [Landscape(adjacency, pheno_list[i], repro_list[i]) for i in range(n_env)]

        params = {
            "pop_size": int(d["N"]),
            "n_cycles": int(d["T"]),
            "mutation_rate": float(d["mu"]),
            "offspring_per_division": int(d["c"]),
            "trial": d.get("trial"),
            "source": "legacy_pickle",
            "source_path": str(path),
        }

        return cls(
            frequencies=freq,
            cycles=np.arange(freq.shape[-1], dtype=np.int32),
            landscapes=landscapes,
            genotypes=np.asarray(d["Gamma_geno"], dtype=np.int32)
            if "Gamma_geno" in d
            else None,
            phenotypes=np.asarray(d["Gamma_pheno"], dtype=np.int32)
            if "Gamma_pheno" in d
            else None,
            f_eq=np.asarray(d["f_eq"], dtype=np.float64) if d.get("f_eq") is not None else None,
            Xbar_theory=float(d["Xbar_theory"]) if d.get("Xbar_theory") is not None else None,
            params=params,
        )

    def __repr__(self) -> str:
        Ng, Np = self.shape
        return (
            f"SimResult(n_genotypes={Ng}, n_phenotypes={Np}, "
            f"n_recorded={self.frequencies.shape[-1]}, "
            f"pop_size={self.params.get('pop_size')}, seed={self.params.get('seed')})"
        )


def load(path: str | Path) -> SimResult:
    """Load a result, dispatching on file extension (``.npz`` or ``.pkl``)."""
    path = Path(path)
    if path.suffix == ".pkl":
        return SimResult.from_legacy_pickle(path)
    return SimResult.from_npz(path)
