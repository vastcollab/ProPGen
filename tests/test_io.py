"""Result and summary I/O, including the reader for pre-1.0 pickle output."""

from __future__ import annotations

import pickle

import numpy as np
import pytest

from propgen import Landscape, SimResult, simulate
from propgen.aggregate import aggregate_runs, collect_runs, load_summary

LANDSCAPE = Landscape(
    adjacency=np.array([[0.0, 1.0], [1.0, 0.0]]),
    pheno_probs=np.array([[0.4, 0.6], [0.7, 0.3]]),
    repro_probs=np.array([0.009, 0.002]),
)


@pytest.fixture
def result():
    return simulate(
        LANDSCAPE, n_cycles=20, pop_size=1000, mutation_rate=0.1, seed=0
    )


def test_npz_roundtrip(tmp_path, result):
    path = result.to_npz(tmp_path / "run.npz")
    loaded = SimResult.from_npz(path)

    assert np.array_equal(result.frequencies, loaded.frequencies)
    assert np.array_equal(result.cycles, loaded.cycles)
    assert np.array_equal(result.genotypes, loaded.genotypes)
    assert np.allclose(result.f_eq, loaded.f_eq)
    assert loaded.Xbar_theory == pytest.approx(result.Xbar_theory)
    assert loaded.params["seed"] == 0
    assert np.allclose(loaded.landscape.pheno_probs, LANDSCAPE.pheno_probs)


def test_npz_loads_without_allowing_pickle(tmp_path, result):
    """Results must be readable with allow_pickle=False.

    Anyone reproducing the paper downloads and opens these files; they must
    not be able to execute code on load.
    """
    path = result.to_npz(tmp_path / "run.npz")
    with np.load(path, allow_pickle=False) as z:
        assert "frequencies" in z
        assert "meta" in z


def test_environment_switching_result_roundtrips(tmp_path):
    from propgen import schedule

    a = Landscape(np.array([[1.0]]), np.array([[0.8, 0.2]]), np.array([0.02, 0.0002]))
    b = Landscape(np.array([[1.0]]), np.array([[0.2, 0.8]]), np.array([0.0, 0.05]))
    res = simulate(
        [a, b], schedule=schedule.switch_at(10), n_cycles=20, pop_size=500,
        mutation_rate=0.0, seed=0, compute_theory=False,
    )
    loaded = SimResult.from_npz(res.to_npz(tmp_path / "env.npz"))
    assert len(loaded.landscapes) == 2
    assert np.allclose(loaded.landscapes[1].repro_probs, [0.0, 0.05])


def _write_legacy_pickle(path, *, trial, freq):
    """Write a dict in the exact shape the pre-1.0 scripts produced."""
    with open(path, "wb") as fh:
        pickle.dump(
            {
                "A": LANDSCAPE.adjacency,
                "N": 1000,
                "T": freq.shape[-1],
                "mu": 0.1,
                "c": 1,
                "pheno_probs": LANDSCAPE.pheno_probs,
                "repro_probs": LANDSCAPE.repro_probs,
                "trial": trial,
                "freq_timeseries": freq,
                "Gamma_geno": np.zeros(1000, dtype=int),
                "Gamma_pheno": np.zeros(1000, dtype=int),
                "f_eq": np.array([0.25, 0.25, 0.25, 0.25]),
                "Xbar_theory": 0.005,
            },
            fh,
        )


def test_legacy_pickle_reader(tmp_path):
    freq = np.random.default_rng(0).random((2, 2, 50))
    freq /= freq.sum(axis=(0, 1), keepdims=True)
    path = tmp_path / "sim_data_trial_1.pkl"
    _write_legacy_pickle(path, trial=1, freq=freq)

    result = SimResult.from_legacy_pickle(path)
    assert result.frequencies.shape == (2, 2, 50)
    assert result.params["pop_size"] == 1000
    assert result.params["mutation_rate"] == 0.1
    assert result.params["trial"] == 1
    assert np.allclose(result.landscape.pheno_probs, LANDSCAPE.pheno_probs)
    assert np.allclose(result.f_eq, 0.25)


def test_legacy_persister_variant_with_two_environments(tmp_path):
    """The persister scripts stored two reproduction vectors and no f_eq."""
    path = tmp_path / "sim_data_env.pkl"
    with open(path, "wb") as fh:
        pickle.dump(
            {
                "A": np.array([[1.0]]),
                "N": 1000, "T": 10, "mu": 0.0, "c": 1,
                "pheno_probs": np.array([[0.8, 0.2]]),
                "repro_probs_1": np.array([0.02, 0.0002]),
                "repro_probs_2": np.array([0.0, 0.05]),
                "trial": 1,
                "freq_timeseries": np.zeros((1, 2, 10)),
            },
            fh,
        )
    result = SimResult.from_legacy_pickle(path)
    assert len(result.landscapes) == 2
    assert result.f_eq is None
    assert np.allclose(result.landscapes[1].repro_probs, [0.0, 0.05])


def test_aggregate_computes_mean_and_sem(tmp_path):
    rng = np.random.default_rng(0)
    for trial in range(1, 6):
        freq = rng.random((2, 2, 30))
        freq /= freq.sum(axis=(0, 1), keepdims=True)
        _write_legacy_pickle(tmp_path / f"sim_data_trial_{trial}.pkl", trial=trial, freq=freq)

    groups = collect_runs(tmp_path, pattern="meanfit", glob="*.pkl")
    assert len(groups) == 1
    summary = aggregate_runs(groups)

    assert summary.n_conditions == 1
    assert summary.mean.shape == (1, 2, 2, 30)
    assert summary.n_trials[0] == 5
    assert np.allclose(summary.mean[0].sum(axis=(0, 1)), 1.0, atol=1e-5)
    assert np.all(summary.sem >= 0)


def test_summary_roundtrip_and_selection(tmp_path):
    rng = np.random.default_rng(0)
    for m in (0.0, 0.5):
        for trial in (1, 2):
            freq = rng.random((2, 2, 30))
            freq /= freq.sum(axis=(0, 1), keepdims=True)
            _write_legacy_pickle(
                tmp_path / f"sim_data_map_{m:.2f}_trial_{trial}.pkl", trial=trial, freq=freq
            )

    summary = aggregate_runs(collect_runs(tmp_path, pattern="buoy", glob="*.pkl"))
    path = summary.to_npz(tmp_path / "summary.npz")
    loaded = load_summary(path)

    assert loaded.param_names == ["mapping"]
    assert loaded.n_conditions == 2
    assert np.allclose(loaded.values("mapping"), [0.0, 0.5])

    panel = loaded.select(mapping=0.5)
    assert panel["mean"].shape == (2, 2, 30)
    assert panel["n_trials"] == 2

    with pytest.raises(KeyError, match="no condition matches"):
        loaded.select(mapping=0.25)


def test_thinning_records_the_cycle_indices(tmp_path):
    freq = np.zeros((2, 2, 100))
    freq[0, 0] = 1.0
    _write_legacy_pickle(tmp_path / "sim_data_trial_1.pkl", trial=1, freq=freq)

    summary = aggregate_runs(collect_runs(tmp_path, pattern="meanfit", glob="*.pkl"), thin=10)
    assert summary.mean.shape[-1] == 10
    assert np.array_equal(summary.cycles, np.arange(0, 100, 10))
    assert summary.meta["thin"] == 10
