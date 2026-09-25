# Phenotypic bridges and fitness-valley crossing

## What this shows

A population fixed at genotype 0 must reach genotype 2 by passing through
genotype 1, whose reproduction rate is low. Genotype 1 expresses the
*high*-fitness phenotype with probability `phi`. Even a small `phi` opens a
"phenotypic bridge" that accelerates valley crossing by more than an order of
magnitude, without any change to the mutation rate.

Sweeps `phi` over 21 values and the valley depth `gamma` over
`{0.002, 0.0008, 0.0002}`, 100 trials each.

## Quick smoke test (~30 s on 8 cores)

    propgen sweep --sweep experiments/bridge/sweep.yaml --quick --jobs 8
    propgen aggregate --raw results/bridge/raw_quick --out results/bridge/summary_quick.npz --thin 10

6 runs (3 phi x 1 gamma x 2 trials) at `pop_size=2000`, 2,000 cycles. Enough
to confirm the pipeline works end to end; far too noisy for the published
numbers.

## Full run (54 core-hours; ~1.5 h at `--jobs 36`, ~7 h at `--jobs 8`)

    propgen sweep --sweep experiments/bridge/sweep.yaml --jobs 36
    propgen aggregate --raw results/bridge/raw --out results/bridge/summary.npz --thin 10

6,300 runs at `pop_size=10000`, 30,000 dilution cycles x 10 generations each.
Produces ~1 GB under `results/bridge/raw/` (gitignored).

`--resume` skips runs whose output already exists, so an interrupted sweep can
be restarted without repeating work.

## Committed summary

`results/bridge/summary.npz` (4.6 MB) holds mean and SEM over 100 trials for
all 63 conditions, keeping every 10th cycle. Thinning is lossless for the
relaxation-time fit: tau differs by under 0.1% from the full-resolution value.

It was produced from the 6,300 runs behind the published figure. Its stored
coordinate is `bridge_mapping`; the bridge probability is
`phi = 1 - bridge_mapping`.

## Make the figure

    jupyter nbconvert --to notebook --execute --inplace experiments/bridge/figure.ipynb

Writes `figures/bridge_trajectories.pdf` and `figures/bridge_tau_vs_phi.pdf`.

`analysis.py` fits the relaxation time by orthogonal distance regression on
the approach of pair (2, 0) to its analytic equilibrium.
