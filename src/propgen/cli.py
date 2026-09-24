"""Command-line interface.

::

    propgen run       --config CONFIG [--set KEY=VALUE]... [--out PATH]
    propgen sweep     --sweep SWEEP.yaml [--jobs N] [--quick] [--resume]
    propgen aggregate --raw DIR --out FILE.npz [--thin K] [--pattern NAME]
    propgen theory    --config CONFIG [--set KEY=VALUE]...
    propgen convert-legacy --in DIR --out DIR
    propgen info
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from . import __version__
from .aggregate import aggregate_runs, collect_runs
from .config import apply_overrides, load_config, resolve_config
from .prosed import simulate
from .results import SimResult
from .theory import equilibrium

__all__ = ["main"]


def format_param(value: float) -> str:
    """Format a parameter value for a filename, consistently.

    Uses ``%g`` with six significant figures, so a value reaches the filename
    the same way regardless of how floating-point arithmetic produced it.
    """
    return f"{float(value):.6g}"


def run_name(experiment: str, sweep: dict[str, float], trial: int | None) -> str:
    parts = [experiment]
    parts += [f"{k}={format_param(v)}" for k, v in sorted(sweep.items())]
    if trial is not None:
        parts.append(f"trial={trial}")
    return "__".join(parts) + ".npz"


def git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            cwd=Path(__file__).resolve().parent,
        )
        return out.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


# ----------------------------------------------------------------------
# run
# ----------------------------------------------------------------------


def _do_run(
    config_path: str,
    overrides: list[str] | None,
    out: str | None,
    seed: Any,
    trial: int | None,
    sweep: dict[str, float] | None,
    experiment: str | None,
    progress: bool = False,
) -> Path | None:
    cfg = apply_overrides(load_config(config_path), overrides)
    resolved = resolve_config(cfg)
    resolved["seed"] = seed
    resolved["progress"] = progress

    result = simulate(**resolved)
    result.params["experiment"] = experiment or cfg.get("experiment", Path(config_path).stem)
    result.params["config"] = str(config_path)
    result.params["trial"] = trial
    result.params["git_commit"] = git_commit()
    if sweep:
        result.params["sweep"] = sweep

    if out is None:
        return None
    out_path = Path(out)
    if out_path.is_dir() or out_path.suffix == "":
        out_path = out_path / run_name(result.params["experiment"], sweep or {}, trial)
    result.to_npz(out_path)
    return out_path


def cmd_run(args: argparse.Namespace) -> int:
    seed = [args.seed, args.trial] if args.trial is not None else args.seed
    path = _do_run(
        args.config,
        args.set,
        args.out,
        seed,
        args.trial,
        None,
        None,
        progress=args.progress,
    )
    print(f"wrote {path}" if path else "run complete (no --out given)")
    return 0


# ----------------------------------------------------------------------
# sweep
# ----------------------------------------------------------------------


def _grid_values(spec: Any) -> list[float]:
    """Expand a grid axis specification into explicit values."""
    if isinstance(spec, list):
        return [float(v) for v in spec]
    if isinstance(spec, dict):
        if "values" in spec:
            return [float(v) for v in spec["values"]]
        if "start" in spec:
            start, stop, step = float(spec["start"]), float(spec["stop"]), float(spec["step"])
            n = int(round((stop - start) / step)) + 1
            return [round(start + i * step, 12) for i in range(n)]
    raise ValueError(f"cannot interpret grid values {spec!r}")


def _expand_grid(axes: list[dict]) -> list[dict[str, float]]:
    """Cartesian product of the sweep axes."""
    combos: list[dict[str, float]] = [{}]
    for axis in axes:
        name = axis["name"]
        values = _grid_values(axis.get("values", axis.get("range")))
        combos = [{**c, name: v} for c in combos for v in values]
    return combos


def _axis_overrides(axes: list[dict], combo: dict[str, float]) -> list[str]:
    """Translate a grid point into ``--set`` style override strings.

    Each axis may declare ``set: {path: expression}``, where the expression is
    evaluated with the axis value bound to ``v``. This is what lets the bridge
    sweep write ``landscape.pheno_probs.1: "[1 - v, v]"``.
    """
    overrides: list[str] = []
    for axis in axes:
        value = combo[axis["name"]]
        for path, expr in (axis.get("set") or {}).items():
            if isinstance(expr, str):
                rendered = eval(expr, {"__builtins__": {}}, {"v": value})  # noqa: S307
            else:
                rendered = expr
            overrides.append(f"{path}={rendered!r}")
    return overrides


def cmd_sweep(args: argparse.Namespace) -> int:
    sweep_path = Path(args.sweep)
    with open(sweep_path) as fh:
        spec = yaml.safe_load(fh)

    base_dir = sweep_path.parent
    config_path = Path(spec["base"])
    if not config_path.is_absolute():
        config_path = (base_dir / config_path).resolve()
        if not config_path.exists():
            config_path = (Path.cwd() / spec["base"]).resolve()

    quick = spec.get("quick", {}) if args.quick else {}
    axes = quick.get("grid", spec.get("grid", []))
    n_trials = int(quick.get("trials", spec.get("trials", 1)))
    base_overrides = [f"{k}={v}" for k, v in (quick.get("overrides") or {}).items()]

    out_dir = Path(args.out) if args.out else Path(spec["out"])
    if args.quick:
        out_dir = out_dir.parent / f"{out_dir.name}_quick"
    out_dir.mkdir(parents=True, exist_ok=True)

    experiment = spec.get("experiment", config_path.stem)
    master_seed = int(spec.get("seed", 0))

    jobs: list[tuple] = []
    for combo in _expand_grid(axes):
        overrides = base_overrides + _axis_overrides(axes, combo)
        for trial in range(1, n_trials + 1):
            target = out_dir / run_name(experiment, combo, trial)
            if args.resume and target.exists():
                continue
            jobs.append(
                (str(config_path), overrides, str(target), [master_seed, trial], trial,
                 combo, experiment)
            )

    if args.dry_run:
        for job in jobs:
            sets = " ".join(f"--set {o}" for o in job[1])
            print(f"propgen run --config {job[0]} {sets} --trial {job[4]} --out {job[2]}")
        print(f"# {len(jobs)} runs", file=sys.stderr)
        return 0

    if not jobs:
        print("nothing to do (all outputs exist; drop --resume to re-run)")
        return 0

    print(f"{len(jobs)} runs -> {out_dir} ({args.jobs} parallel)")
    failures = 0
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        futures = {pool.submit(_do_run, *job): job for job in jobs}
        for i, fut in enumerate(as_completed(futures), 1):
            try:
                fut.result()
            except Exception as exc:  # noqa: BLE001
                failures += 1
                print(f"  FAILED {Path(futures[fut][2]).name}: {exc}", file=sys.stderr)
            if i % max(1, len(jobs) // 20) == 0 or i == len(jobs):
                print(f"  {i}/{len(jobs)}", flush=True)

    print(f"done: {len(jobs) - failures} succeeded, {failures} failed")
    return 1 if failures else 0


# ----------------------------------------------------------------------
# aggregate
# ----------------------------------------------------------------------


def cmd_aggregate(args: argparse.Namespace) -> int:
    groups = collect_runs(
        args.raw,
        pattern=args.pattern,
        glob=args.glob or ("*.pkl" if args.pattern in {"bridge", "buoy", "absfit", "meanfit"}
                           else "*.npz"),
    )
    print(f"{len(groups)} conditions, {sum(len(v) for v in groups.values())} runs")
    summary = aggregate_runs(
        groups,
        thin=args.thin,
        meta={
            "raw_dir": str(args.raw),
            "propgen_version": __version__,
            "git_commit": git_commit(),
            "pattern": args.pattern,
        },
    )
    path = summary.to_npz(args.out)
    print(f"wrote {path} ({path.stat().st_size / 1e6:.2f} MB)")
    print(f"  {summary}")
    return 0


# ----------------------------------------------------------------------
# theory / convert-legacy / info
# ----------------------------------------------------------------------


def cmd_theory(args: argparse.Namespace) -> int:
    cfg = apply_overrides(load_config(args.config), args.set)
    resolved = resolve_config(cfg)
    landscape = resolved["landscape"]
    if isinstance(landscape, list):
        landscape = landscape[0]

    f_eq, Xbar = equilibrium(
        landscape, resolved["mutation_rate"], resolved["offspring_per_division"]
    )
    Ng, Np = landscape.shape
    print(f"landscape: {Ng} genotypes x {Np} phenotypes, mu={resolved['mutation_rate']}")
    print("equilibrium frequencies f_eq[g, p]:")
    for g in range(Ng):
        for p in range(Np):
            print(f"  ({g}, {p})  {f_eq[g * Np + p]:.6f}")
    print(f"mean fitness Xbar = {Xbar:.8g}")
    return 0


def cmd_convert_legacy(args: argparse.Namespace) -> int:
    src, dst = Path(args.input), Path(args.out)
    dst.mkdir(parents=True, exist_ok=True)
    files = sorted(src.glob("*.pkl"))
    if not files:
        print(f"no .pkl files in {src}", file=sys.stderr)
        return 1
    for i, path in enumerate(files, 1):
        SimResult.from_legacy_pickle(path).to_npz(dst / f"{path.stem}.npz")
        if i % 200 == 0 or i == len(files):
            print(f"  {i}/{len(files)}", flush=True)
    print(f"converted {len(files)} files -> {dst}")
    return 0


def cmd_info(args: argparse.Namespace) -> int:
    import platform

    import scipy

    info = {
        "propgen": __version__,
        "git_commit": git_commit(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
    }
    try:
        import matplotlib

        info["matplotlib"] = matplotlib.__version__
    except ImportError:
        info["matplotlib"] = None
    print(json.dumps(info, indent=2))
    return 0


# ----------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="propgen",
        description="Probabilistic Phenotype Genetics: ProSeD simulation and ProP Gen theory.",
    )
    parser.add_argument("--version", action="version", version=f"propgen {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("run", help="run a single ProSeD simulation")
    p.add_argument("--config", required=True)
    p.add_argument("--set", action="append", metavar="KEY=VALUE",
                   help="override a config value, e.g. simulation.mutation_rate=0.05")
    p.add_argument("--out", help="output .npz file or directory")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--trial", type=int, default=None)
    p.add_argument("--progress", action="store_true")
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("sweep", help="run a parameter sweep in parallel")
    p.add_argument("--sweep", required=True, help="sweep specification YAML")
    p.add_argument("--out", help="override the sweep's output directory")
    p.add_argument("--jobs", type=int, default=0, help="parallel workers (default: all cores)")
    p.add_argument("--quick", action="store_true", help="use the sweep's reduced-scale preset")
    p.add_argument("--resume", action="store_true", help="skip runs whose output already exists")
    p.add_argument("--dry-run", action="store_true", help="print commands instead of running")
    p.set_defaults(func=cmd_sweep)

    p = sub.add_parser("aggregate", help="reduce runs to a committed summary")
    p.add_argument("--raw", required=True, help="directory of run files")
    p.add_argument("--out", required=True, help="output summary .npz")
    p.add_argument("--thin", type=int, default=1, help="keep every k-th recorded cycle")
    p.add_argument("--pattern", default=None,
                   help="filename regex, or one of: bridge, buoy, absfit, meanfit")
    p.add_argument("--glob", default=None)
    p.set_defaults(func=cmd_aggregate)

    p = sub.add_parser("theory", help="print the analytic equilibrium for a config")
    p.add_argument("--config", required=True)
    p.add_argument("--set", action="append", metavar="KEY=VALUE")
    p.set_defaults(func=cmd_theory)

    p = sub.add_parser("convert-legacy", help="convert pre-1.0 pickle output to .npz")
    p.add_argument("--in", dest="input", required=True)
    p.add_argument("--out", required=True)
    p.set_defaults(func=cmd_convert_legacy)

    p = sub.add_parser("info", help="print version and environment information")
    p.set_defaults(func=cmd_info)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if getattr(args, "jobs", None) == 0:
        import os

        args.jobs = os.cpu_count() or 1
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
