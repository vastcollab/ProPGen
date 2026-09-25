"""YAML configuration for ProSeD runs.

A config declares a landscape and the simulation parameters to run it under::

    experiment: bridge
    seed: 42
    landscape:
      adjacency:   data/bridge/adjacency.txt
      pheno_probs: data/bridge/pheno_probs.txt
      repro_probs: data/bridge/repro_probs.txt
    simulation:
      pop_size: 10000
      n_cycles: 30000              # dilution cycles, not generations
      generations_per_cycle: 10
      offspring_per_division: 1
      mutation_rate: 0.1
      init: {kind: at, genotype: 0, phenotype: 0}

Paths are resolved relative to the config file, so a config works from any
working directory.

Models with a switching environment declare several environments and a
schedule::

    landscape:
      adjacency: data/persister/adjacency.txt
      environments:
        - {pheno_probs: ..., repro_probs: ...}
        - {pheno_probs: ..., repro_probs: ...}
    simulation:
      schedule: {kind: switch_at, cycles: [100]}

The flat schema used by the original scripts (``N``, ``timesteps``, ``mu``,
``c``, ``num_gens``, ``A``, ``pheno_probs``, ``repro_probs``) is still
accepted, with a :class:`DeprecationWarning`.
"""

from __future__ import annotations

import ast
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from . import initialization
from . import schedule as schedule_mod
from .landscape import Landscape, read_matrix

__all__ = ["load_config", "resolve_config", "apply_overrides", "build_landscapes"]

# Old flat key -> new nested location.
_LEGACY_KEYS = {
    "N": ("simulation", "pop_size"),
    "timesteps": ("simulation", "n_cycles"),
    "mu": ("simulation", "mutation_rate"),
    "c": ("simulation", "offspring_per_division"),
    "num_gens": ("simulation", "generations_per_cycle"),
    "A": ("landscape", "adjacency"),
    "pheno_probs": ("landscape", "pheno_probs"),
    "repro_probs": ("landscape", "repro_probs"),
}


def load_config(path: str | Path) -> dict[str, Any]:
    """Read a config file, normalising the legacy flat schema if needed.

    The returned dict carries ``_base_dir``, the directory the config was read
    from, which :func:`resolve_config` uses to resolve data paths.
    """
    path = Path(path)
    with open(path) as fh:
        raw = yaml.safe_load(fh) or {}

    if "simulation" not in raw and any(k in raw for k in _LEGACY_KEYS):
        warnings.warn(
            f"{path} uses the pre-1.0 flat config schema; translating "
            "automatically. See configs/ for the current format.",
            DeprecationWarning,
            stacklevel=2,
        )
        raw = _translate_legacy(raw)

    raw["_base_dir"] = str(path.parent)
    return raw


def _translate_legacy(raw: dict[str, Any]) -> dict[str, Any]:
    cfg: dict[str, Any] = {"landscape": {}, "simulation": {}}
    if "seed" in raw:
        cfg["seed"] = raw["seed"]

    for old, (section, new) in _LEGACY_KEYS.items():
        if old in raw:
            cfg[section][new] = raw[old]

    # Environment-switching variants of the old persister configs.
    envs = []
    for suffix in ("_1", "_2"):
        pheno = raw.get(f"pheno_probs{suffix}", raw.get("pheno_probs"))
        repro = raw.get(f"repro_probs{suffix}")
        if repro is not None:
            envs.append({"pheno_probs": pheno, "repro_probs": repro})
    if envs:
        cfg["landscape"].pop("pheno_probs", None)
        cfg["landscape"].pop("repro_probs", None)
        cfg["landscape"]["environments"] = envs

    return cfg


def _resolve(base: Path, value: Any) -> Any:
    """Resolve a path-like config value against the config's directory."""
    if isinstance(value, str):
        candidate = Path(value)
        return str(candidate if candidate.is_absolute() else base / candidate)
    return value


def build_landscapes(cfg: dict[str, Any]) -> list[Landscape]:
    """Construct the landscape(s) declared by a config."""
    base = Path(cfg.get("_base_dir", "."))
    spec = cfg.get("landscape")
    if not spec:
        raise ValueError("config has no 'landscape' section")

    if "adjacency" not in spec:
        raise ValueError("config's landscape section has no 'adjacency' entry")
    adjacency = read_matrix(_resolve(base, spec["adjacency"]))

    def _matrix(value: Any, *, ravel: bool = False) -> np.ndarray:
        if isinstance(value, str):
            arr = read_matrix(_resolve(base, value))
        else:
            arr = np.atleast_2d(np.asarray(value, dtype=np.float64))
        return arr.ravel() if ravel else arr

    if "environments" in spec:
        return [
            Landscape(
                adjacency,
                _matrix(env["pheno_probs"]),
                _matrix(env["repro_probs"], ravel=True),
            )
            for env in spec["environments"]
        ]

    return [
        Landscape(
            adjacency,
            _matrix(spec["pheno_probs"]),
            _matrix(spec["repro_probs"], ravel=True),
        )
    ]


def build_init(spec: Any) -> initialization.Init:
    """Construct an initializer from its config declaration."""
    if spec is None:
        return initialization.uniform_over_support()
    if isinstance(spec, str):
        spec = {"kind": spec}

    kind = spec.get("kind", "uniform_over_support")
    if kind in ("uniform_over_support", "uniform"):
        return initialization.uniform_over_support()
    if kind == "at":
        return initialization.at(int(spec["genotype"]), int(spec["phenotype"]))
    if kind == "frequencies":
        values = spec["values"]
        if isinstance(values, dict):
            values = {tuple(int(i) for i in k.split(",")): v for k, v in values.items()} \
                if all(isinstance(k, str) for k in values) else values
        return initialization.frequencies(values)
    raise ValueError(f"unknown init kind {kind!r}")


def build_schedule(spec: Any) -> schedule_mod.Schedule:
    """Construct an environment schedule from its config declaration."""
    if spec is None:
        return schedule_mod.constant()
    if isinstance(spec, str):
        spec = {"kind": spec}

    kind = spec.get("kind", "constant")
    if kind == "constant":
        return schedule_mod.constant()
    if kind == "switch_at":
        return schedule_mod.switch_at(*spec["cycles"])
    if kind == "periodic":
        return schedule_mod.periodic(int(spec["period"]), int(spec.get("n_env", 2)))
    raise ValueError(f"unknown schedule kind {kind!r}")


def _parse_value(text: str) -> Any:
    """Parse an override value: a Python literal if possible, else a string."""
    try:
        return ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return text


def apply_overrides(cfg: dict[str, Any], overrides: list[str] | None) -> dict[str, Any]:
    """Apply ``key.path=value`` overrides to a config.

    Two forms are supported::

        simulation.mutation_rate=0.05          # set a config value
        landscape.repro_probs.1=0.0002         # set one entry of a matrix
        landscape.pheno_probs.1=[0.9, 0.1]     # set one row of a matrix

    The indexed forms are recorded and applied after the landscape is built;
    they replace the bespoke ``--mapping_prob`` / ``--repro_prob`` flags the
    original scripts used, without the in-place mutation those relied on.
    """
    if not overrides:
        return cfg

    cfg = dict(cfg)
    pending: list[tuple[str, int, Any]] = list(cfg.get("_landscape_overrides", []))

    for item in overrides:
        if "=" not in item:
            raise ValueError(f"override {item!r} is not of the form key.path=value")
        key, _, raw_value = item.partition("=")
        key = key.strip()
        value = _parse_value(raw_value.strip())
        parts = key.split(".")

        if len(parts) == 3 and parts[0] == "landscape" and parts[1] in (
            "pheno_probs",
            "repro_probs",
        ):
            pending.append((parts[1], int(parts[2]), value))
            continue

        node: Any = cfg
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value

    cfg["_landscape_overrides"] = pending
    return cfg


def _apply_landscape_overrides(
    landscapes: list[Landscape], overrides: list[tuple[str, int, Any]]
) -> list[Landscape]:
    out = []
    for ls in landscapes:
        for what, index, value in overrides:
            if what == "pheno_probs":
                ls = ls.with_pheno_probs(index, value)
            else:
                ls = ls.with_repro_prob(index, float(value))
        out.append(ls)
    return out


def resolve_config(cfg: dict[str, Any]) -> dict[str, Any]:
    """Turn a loaded config into keyword arguments for :func:`propgen.simulate`.

    Returns a dict with ``landscape``, ``schedule``, ``init`` and every
    simulation parameter, ready to splat into ``simulate(**resolved)``.
    """
    sim = dict(cfg.get("simulation") or {})
    landscapes = build_landscapes(cfg)
    landscapes = _apply_landscape_overrides(landscapes, cfg.get("_landscape_overrides", []))

    for required in ("pop_size", "n_cycles", "mutation_rate"):
        if required not in sim:
            raise ValueError(f"config's simulation section is missing '{required}'")

    resolved: dict[str, Any] = {
        "landscape": landscapes if len(landscapes) > 1 else landscapes[0],
        "n_cycles": int(sim["n_cycles"]),
        "pop_size": int(sim["pop_size"]),
        "mutation_rate": float(sim["mutation_rate"]),
        "generations_per_cycle": int(sim.get("generations_per_cycle", 10)),
        "offspring_per_division": int(sim.get("offspring_per_division", 1)),
        "record_every": int(sim.get("record_every", 1)),
        "init": build_init(sim.get("init")),
        "schedule": build_schedule(sim.get("schedule")),
    }
    return resolved
