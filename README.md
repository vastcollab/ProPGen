# ProPGen

Implementation of [Evolutionary dynamics under phenotypic uncertainty](https://www.biorxiv.org/content/10.64898/2026.03.15.711953v1) by Vaibhav Mohanty, Anna Sappington, Eugene I. Shakhnovich and Bonnie Berger.

Classical population genetics has relied on the same stochastic differential equations for over 60 years, and those equations assume that a genotype determines a phenotype. Real biological systems, from bacteria to cancers, are full of phenotypic heterogeneity and noise. **Probabilistic Phenotype Genetics (ProP Gen)** replaces the deterministic genotype-phenotype map with a probabilistic one, and doing so breaks several central tenets of the classical theory — including the invariance of evolutionary dynamics to a global shift in absolute fitness at fixed population size. The theory predicts that low-probability, high-fitness *phenotypic bridges* can accelerate fitness-valley crossing by more than an order of magnitude at unchanged mutation rates, and explains *phenotypic buoying*, in which a low-fitness phenotype persists at high frequency because a high-fitness phenotype carried by the same genotype acts as a source.

The standard Wright-Fisher branching process cannot represent phenotypic uncertainty correctly, because it treats phenotype as inherited rather than re-expressed. This repository implements the alternative introduced in the paper, **ProSeD** (Probabilistic Serial Dilution): a discrete-time algorithm with overlapping generations, phenotypic noise and stochastic phenotype switching, modelled on a laboratory serial-passaging experiment.

The codebase allows users to run ProSeD simulations on any genotype-phenotype map a user supplies as well as recreate simulations from our paper. 

## Installation

```
git clone https://github.com/vastcollab/ProPGen && cd ProPGen
pip install -e ".[figures]"
propgen info
```

Requires Python 3.10 or later. Dependencies are `numpy`, `scipy`, `pyyaml` and `tqdm`, plus `matplotlib` and `mpmath` for the figures. 

## Quickstart

Run ProSeD on a model defined directly in NumPy, and compare it to the analytic prediction:

```python
import numpy as np
from propgen import Landscape, simulate, equilibrium

landscape = Landscape(
    adjacency=np.array([[0.0, 1.0],            # genotype mutation graph
                        [1.0, 0.0]]),
    pheno_probs=np.array([[0.4, 0.6],          # P(phenotype | genotype), rows sum to 1
                          [0.7, 0.3]]),
    repro_probs=np.array([0.009, 0.002]),      # division probability per phenotype
)

result = simulate(
    landscape,
    n_cycles=400,          # dilution cycles
    pop_size=20_000,
    mutation_rate=0.1,
    generations_per_cycle=10,
    seed=0,
)

f_eq, mean_fitness = equilibrium(landscape, mutation_rate=0.1)

print(result.final_frequencies(last=100).ravel())   # simulated
print(f_eq)                                         # analytic
```

The two agree to about 0.001 in frequency, in under a second.

From the command line, on files instead of arrays:

```
# Run one simulation
propgen run --config configs/buoy.yaml --out results/my_run.npz --seed 0

# Analytic equilibrium only, no simulation
propgen theory --config configs/buoy.yaml

# Override any config value, including individual matrix entries
propgen run --config configs/bridge.yaml \
    --set landscape.pheno_probs.1='[0.1, 0.9]' \
    --set simulation.mutation_rate=0.05 \
    --out results/my_run.npz

# Run a parameter sweep in parallel
propgen sweep --sweep experiments/buoy/sweep.yaml --jobs 16

# Reduce per-trial runs to mean +/- SEM across trials
propgen aggregate --raw results/buoy/raw --out results/buoy/summary.npz
```

## Input files

A model is three whitespace-delimited text matrices:

```
adjacency.txt     (Ng, Ng)   genotype mutation graph, symmetric, 0/1
pheno_probs.txt   (Ng, Np)   P(phenotype | genotype); each row sums to 1
repro_probs.txt   (Np,)      division probability per phenotype, in [0, 1]
```

Genotypes are rows, phenotypes are columns, and flattened `(g, p)` vectors are row-major, so index `g * Np + p` is the pair `(g, p)`. Landscapes are validated on construction, so malformed input fails immediately rather than producing wrong dynamics. See [`data/README.md`](data/README.md) for the models used in the paper.

## Configuration

```yaml
experiment: buoy
seed: 42

landscape:
  adjacency:   ../data/buoy/adjacency.txt      # paths resolve relative to this file
  pheno_probs: ../data/buoy/pheno_probs.txt
  repro_probs: ../data/buoy/repro_probs.txt

simulation:
  pop_size: 10000
  n_cycles: 250                  # dilution cycles, NOT generations
  generations_per_cycle: 10      # reproduction rounds between dilutions
  offspring_per_division: 1
  mutation_rate: 0.1
  init: {kind: uniform_over_support}
```

`init` may be `uniform_over_support`, `{kind: at, genotype: g, phenotype: p}`, or `{kind: frequencies, values: [...]}`.

A switching environment declares several environments and a schedule — see [`configs/persister.yaml`](configs/persister.yaml):

```yaml
landscape:
  adjacency: ../data/persister/adjacency.txt
  environments:
    - {pheno_probs: ..., repro_probs: ...}     # antibiotic-free
    - {pheno_probs: ..., repro_probs: ...}     # antibiotic
simulation:
  schedule: {kind: switch_at, cycles: [100]}
```

## Reproducing the paper

Aggregated results (mean ± SEM across trials) for every experiment are committed under `results/`, totalling 4.8 MB. **Every figure therefore reproduces from a fresh clone with no downloads and no simulation.**

```
# All figures, from the committed summaries (~30 s)
bash scripts/make_figures.sh
```

To re-run the simulations themselves. All timings are measured on one CPU core at the published settings:

```
# Mean fitness from a non-equilibrium start -- 5 runs, ~1 s
propgen sweep --sweep experiments/meanfit/sweep.yaml --jobs 5

# Absolute-fitness non-invariance, deterministic vs probabilistic map -- 60 runs, ~16 s
propgen sweep --sweep experiments/absfit/sweep_dgp.yaml  --jobs 16
propgen sweep --sweep experiments/absfit/sweep_prgp.yaml --jobs 16

# Phenotypic buoying -- 110 runs, ~30 s
propgen sweep --sweep experiments/buoy/sweep.yaml --jobs 16

# Phenotypic bridges / valley crossing -- 6,300 runs, 54 core-hours
# (~1.5 h at --jobs 36, ~7 h at --jobs 8; --resume restarts an interrupted sweep)
propgen sweep --sweep experiments/bridge/sweep.yaml --jobs 36
```

then aggregate and plot:

```
propgen aggregate --raw results/buoy/raw   --out results/buoy/summary.npz
propgen aggregate --raw results/bridge/raw --out results/bridge/summary.npz --thin 10
bash scripts/make_figures.sh
```

The persister, coexistence-phase-diagram and rugged-landscape panels are analytic and need no simulation or stored data at all. The phase diagram in particular is computed exactly, by evaluating the Perron-Frobenius equilibrium over a grid in mutation rate and genotype-phenotype map:

```python
from propgen import phase_diagram

diagram = phase_diagram(
    landscape,
    mutation_rates=np.linspace(0.005, 0.5, 100),
    pheno_prob_values=np.linspace(0.0, 1.0, 100),
)
diagram.phase       # (100, 100) phase labels; ten distinct coexistence phases
```

Every experiment has a `--quick` preset that reduces population size, cycle count and trial count — and nothing else, so the model being exercised is identical. One command runs the entire pipeline end to end:

```
# Simulate, aggregate and plot everything at reduced scale (~40 s on 8 cores)
bash scripts/smoke_test.sh
```

Per-experiment commands, expected outputs and runtimes are in [`experiments/README.md`](experiments/README.md) and each experiment's own README.

### Notes on the committed results

`results/bridge/summary.npz` was aggregated from the 6,300 runs behind the published figure, keeping every 10th dilution cycle. The thinning is lossless for the analysis: the fitted relaxation time differs from the full-resolution value by under 0.1%.

Simulations are seeded through an explicit `numpy.random.Generator`, and per-trial seeds are recorded in every output file, so any single trial can be reproduced in isolation. The code used for the preprint did not set a seed; re-running these sweeps will therefore produce results that are statistically equivalent to, but not bitwise identical with, the archived output. The inner loop was also vectorised for this release — about 21× faster, verified by distributional-equivalence tests against a transcription of the original implementation in `tests/reference_loop.py`.

Certain schematic panels in the manuscript use previously published third-party data, which is not redistributed here; see the paper for those sources.

## Tests

```
pip install -e ".[dev]"
pytest
```

The suite checks, among other things, that ProSeD simulations converge to the equilibrium frequencies predicted analytically by ProP Gen theory — an independent check of the paper's central result, run in continuous integration on every commit. It also verifies seed determinism, that the global NumPy random state is never touched, that the vectorised loop is distributionally equivalent to the original, and that every shipped config loads and runs.

## Repository layout

```
src/propgen/      the package: Landscape, ProSeD, theory, aggregation, CLI
configs/          one YAML per model
data/             genotype-phenotype maps, mutation graphs, reproduction rates
experiments/      one directory per paper result: sweep, figure notebook, README
results/          committed mean +/- SEM summaries (raw per-trial runs are gitignored)
theory/           Mathematica notebooks with the symbolic derivations
tests/            test suite
scripts/          make_figures.sh, smoke_test.sh
```

The state of the code at preprint submission is preserved at tag `v0.0.1-preprint`.

## License

MIT.

## Citation

```
@article{mohanty2026propgen,
  title   = {Evolutionary dynamics under phenotypic uncertainty},
  author  = {Mohanty, Vaibhav and Sappington, Anna and Shakhnovich, Eugene I. and Berger, Bonnie},
  journal = {bioRxiv},
  year    = {2026},
  doi     = {10.64898/2026.03.15.711953}
}
```
