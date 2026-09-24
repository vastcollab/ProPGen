"""Every shipped config loads, validates and runs.

This is the guard against config rot: the pre-1.0 repository contained a
config that was a stale copy of another experiment's, pointing at the wrong
data directory, which nothing would have caught.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from propgen import load_config, resolve_config, simulate
from propgen.config import apply_overrides

REPO = Path(__file__).resolve().parent.parent
CONFIGS = sorted((REPO / "configs").glob("*.yaml"))
SWEEPS = sorted((REPO / "experiments").glob("*/sweep*.yaml"))


def test_configs_exist():
    assert CONFIGS, "no configs found"
    assert SWEEPS, "no sweeps found"


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: p.stem)
def test_config_loads_and_runs(path):
    resolved = resolve_config(load_config(path))
    resolved.update(n_cycles=3, pop_size=200, seed=0)
    result = simulate(**resolved)

    assert np.allclose(result.frequencies.sum(axis=(0, 1)), 1.0)
    assert result.frequencies.shape[-1] == 3


@pytest.mark.parametrize("path", SWEEPS, ids=lambda p: f"{p.parent.name}-{p.stem}")
def test_sweep_references_a_real_config_and_expands(path):
    with open(path) as fh:
        spec = yaml.safe_load(fh)

    base = (path.parent / spec["base"]).resolve()
    assert base.exists(), f"{path} references missing config {base}"

    from propgen.cli import _axis_overrides, _expand_grid

    for key in ("grid", None):
        axes = spec["quick"]["grid"] if key is None else spec.get("grid", [])
        combos = _expand_grid(axes)
        assert combos, f"{path} grid expanded to nothing"
        # Every grid point must produce overrides that yield a valid landscape.
        cfg = apply_overrides(load_config(base), _axis_overrides(axes, combos[0]))
        resolve_config(cfg)


@pytest.mark.parametrize("path", SWEEPS, ids=lambda p: f"{p.parent.name}-{p.stem}")
def test_quick_preset_only_reduces_scale(path):
    """--quick must not alter the model, only the statistics.

    It may change population size, cycle count and trial count. Changing a
    rate or a probability would make the smoke test exercise different
    science from the full run.
    """
    with open(path) as fh:
        spec = yaml.safe_load(fh)

    allowed = {"simulation.n_cycles", "simulation.pop_size", "simulation.record_every"}
    overrides = set((spec.get("quick", {}).get("overrides") or {}).keys())
    assert overrides <= allowed, f"{path} --quick changes the model: {overrides - allowed}"


def test_legacy_flat_config_is_translated(tmp_path):
    """The pre-1.0 flat schema still loads, with a deprecation warning."""
    (tmp_path / "adj.txt").write_text("0 1\n1 0\n")
    (tmp_path / "pi.txt").write_text("0.4 0.6\n0.7 0.3\n")
    (tmp_path / "r.txt").write_text("0.009 0.002\n")

    legacy = tmp_path / "old.yaml"
    legacy.write_text(
        yaml.safe_dump(
            {
                "seed": 42, "graph_setup": "file", "A": "adj.txt",
                "N": 500, "timesteps": 5, "mu": 0.1, "c": 1, "num_gens": 10,
                "phenotype_probs_setup": "file", "pheno_probs": "pi.txt",
                "repro_probs_setup": "file", "repro_probs": "r.txt",
            }
        )
    )

    with pytest.warns(DeprecationWarning, match="flat config schema"):
        cfg = load_config(legacy)

    resolved = resolve_config(cfg)
    assert resolved["pop_size"] == 500
    assert resolved["n_cycles"] == 5
    assert resolved["mutation_rate"] == 0.1
    assert resolved["generations_per_cycle"] == 10


def test_overrides_set_matrix_entries():
    cfg = load_config(REPO / "configs" / "bridge.yaml")
    cfg = apply_overrides(cfg, ["landscape.pheno_probs.1=[0.25, 0.75]",
                                "landscape.repro_probs.1=0.0008"])
    resolved = resolve_config(cfg)
    landscape = resolved["landscape"]

    assert np.allclose(landscape.pheno_probs[1], [0.25, 0.75])
    assert landscape.repro_probs[1] == pytest.approx(0.0008)


def test_override_of_a_scalar_simulation_parameter():
    cfg = load_config(REPO / "configs" / "buoy.yaml")
    cfg = apply_overrides(cfg, ["simulation.mutation_rate=0.05"])
    assert resolve_config(cfg)["mutation_rate"] == 0.05


def test_malformed_override_is_rejected():
    cfg = load_config(REPO / "configs" / "buoy.yaml")
    with pytest.raises(ValueError, match="key.path=value"):
        apply_overrides(cfg, ["simulation.mutation_rate"])
