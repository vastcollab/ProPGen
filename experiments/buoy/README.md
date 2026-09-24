# Phenotypic buoying

## What this shows

A low-fitness phenotype sits at a much higher frequency than its own fitness
would support, because the genotype carrying it also expresses a high-fitness
phenotype that acts as a source. Sweeps genotype 1's phenotype distribution
over 11 values, 10 trials each.

The figure overlays the analytic equilibrium from `calc_f_eq` on the simulated
trajectories; they agree to within 0.012 in frequency.

## Quick smoke test (~2 s)

    propgen sweep --sweep experiments/buoy/sweep.yaml --quick --jobs 8
    propgen aggregate --raw results/buoy/raw_quick --out results/buoy/summary_quick.npz

## Full run (~30 s on 1 core, ~3 s on 16)

    propgen sweep --sweep experiments/buoy/sweep.yaml --jobs 16
    propgen aggregate --raw results/buoy/raw --out results/buoy/summary.npz

110 runs at `pop_size=10000`, 250 dilution cycles.

## Committed summary

`results/buoy/summary.npz` (75 KB), full time resolution.

## Make the figure

    jupyter nbconvert --to notebook --execute --inplace experiments/buoy/figure.ipynb

Writes `figures/buoy_trajectories.pdf` and `figures/buoy_theory_vs_simulation.pdf`.
