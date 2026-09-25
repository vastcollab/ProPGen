# Probabilistic Serial Dilution (ProSeD) and Evolutionary Dynamics under Phenotypic Uncertainty

Implementation of [Evolutionary dynamics under phenotypic uncertainty](https://www.biorxiv.org/content/10.64898/2026.03.15.711953v1) by Vaibhav Mohanty*, Anna Sappington*, Eugene I. Shakhnovich and Bonnie Berger.

<p align="center">
<img width="664" height="236" alt="Screenshot 2026-09-24 at 8 25 21 PM" src="https://github.com/user-attachments/assets/914f3aaf-ff9b-4b0c-9e51-efda785ebc96" />
</p>


Classical population genetics has relied on the same stochastic differential equations for over 60 years, and those equations assume that a genotype determines a phenotype. Real biological systems, from bacteria to cancers, are full of phenotypic heterogeneity and noise. **Probabilistic Phenotype Genetics** replaces the deterministic genotype-phenotype map with a probabilistic one, and doing so breaks several central tenets of the classical theory — including the invariance of evolutionary dynamics to a global shift in absolute fitness at fixed population size. The theory predicts that low-probability, high-fitness *phenotypic bridges* can accelerate fitness-valley crossing by more than an order of magnitude at unchanged mutation rates, and explains *phenotypic buoying*, in which a low-fitness phenotype persists at high frequency because a high-fitness phenotype carried by the same genotype acts as a source.

The standard Wright-Fisher branching process cannot represent this phenotypic uncertainty because it treats phenotype as inherited rather than re-expressed. This repository implements the alternative introduced in the paper, **ProSeD** (Probabilistic Serial Dilution): a discrete-time algorithm with overlapping generations, phenotypic noise and stochastic phenotype switching, modelled on a laboratory serial-passaging experiment.

The codebase allows users to run ProSeD simulations on any genotype-phenotype map a user supplies as well as recreate simulations from our paper. 

## Installation

```
git clone https://github.com/vastcollab/ProPGen && cd ProPGen
pip install -e ".[figures]"
propgen info
```

Requires Python 3.10 or later. Dependencies are `numpy`, `scipy`, `pyyaml` and `tqdm`, plus `matplotlib` and `mpmath` for the figures. 

## ProSeD quickstart

Example demonstrating how to run a ProSeD simulation on user inputted data and compare with the analytic prediction:

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

print(result.final_frequencies(last=100).ravel())   # ProSeD simulated
print(f_eq)                                         # analytic
```

Example of how to run ProSeD on files in the command line:

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

A ProSeD simulation requires three files:

```
adjacency.txt     (Ng, Ng)   genotype mutation graph, symmetric, 0/1
pheno_probs.txt   (Ng, Np)   P(phenotype | genotype); each row sums to 1
repro_probs.txt   (Np,)      division probability per phenotype, in [0, 1]
```

Genotypes are rows, phenotypes are columns.

## Configs

Example set up for a configuration file:

```yaml
experiment: buoy
seed: 42

landscape:
  adjacency:   ../data/buoy/adjacency.txt      # paths resolve relative to this file
  pheno_probs: ../data/buoy/pheno_probs.txt
  repro_probs: ../data/buoy/repro_probs.txt

simulation:
  pop_size: 10000
  n_cycles: 250                  # dilution cycles
  generations_per_cycle: 10      # reproduction rounds between dilutions
  offspring_per_division: 1
  mutation_rate: 0.1
  init: {kind: at, genotype: 0, phenotype: 0}
```

`init` sets where the population starts on cycle 0. Every individual has a genotype (a row) and a phenotype (a column), so `init` specifies how the population is distributed over those `(genotype, phenotype)` pairs at the start. There are three choices:

**1. Start everyone at one pair** — the whole population begins identical, at a single genotype and phenotype:

```yaml
init: {kind: at, genotype: 0, phenotype: 0}
```

**2. Spread the population evenly across all possible pairs** — every pair that a genotype can actually express (i.e. every `(g, p)` with `pheno_probs[g, p] > 0`) gets an equal share:

```yaml
init: uniform_over_support
```

**3. Start from a distribution you specify** — give the fraction of the population at each pair; the values are normalized to sum to 1. Useful for starting deliberately away from equilibrium:

```yaml
init: {kind: frequencies, values: [0.7, 0.0, 0.3, 0.0]}
```

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

## Reproducing manuscript results

Aggregated results (mean ± SEM across trials) for every experiment are located under `results/`. 
```
bash scripts/make_figures.sh
```

To re-run the simulations :

```
# Mean fitness from a non-equilibrium start 
propgen sweep --sweep experiments/meanfit/sweep.yaml --jobs 5

# Absolute-fitness non-invariance, deterministic vs probabilistic map 
propgen sweep --sweep experiments/absfit/sweep_dgp.yaml  --jobs 16
propgen sweep --sweep experiments/absfit/sweep_prgp.yaml --jobs 16

# Phenotypic buoying
propgen sweep --sweep experiments/buoy/sweep.yaml --jobs 16

# Phenotypic bridges / valley crossing 
propgen sweep --sweep experiments/bridge/sweep.yaml --jobs 36
```

then aggregate and plot:

```
propgen aggregate --raw results/buoy/raw   --out results/buoy/summary.npz
propgen aggregate --raw results/bridge/raw --out results/bridge/summary.npz --thin 10
bash scripts/make_figures.sh
```

The persister and coexistence phase diagram figues are analytic and do not require simulations. The phase diagram is computed exactly, by evaluating the equilibrium frequency orderings over a parameter grid:

```python
from propgen import phase_diagram

diagram = phase_diagram(
    landscape,
    mutation_rates=np.linspace(0.005, 0.5, 100),
    pheno_prob_values=np.linspace(0.0, 1.0, 100),
)
diagram.phase       
```

Every experiment has a `--quick` preset that reduces population size, cycle count and trial count to speed up long running simulations if desired. 

Per-experiment commands, expected outputs and runtimes are in [`experiments/README.md`](experiments/README.md) and each experiment's own README.

## Repository layout

```
src/propgen/      the package (ProSeD)
configs/          one YAML per sim
data/             genotype-phenotype maps, mutation graphs, reproduction rates
experiments/      one directory per paper result: sweep, figure notebook, README
results/          committed mean +/- SEM summaries (raw per-trial runs are gitignored)
scripts/          make_figures.sh
```


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
