# Mean fitness from a non-equilibrium start

## What this shows

The population starts at frequencies deliberately away from equilibrium
(`[0.4, 0.1, 0.05, 0.45]` over the four genotype-phenotype pairs) and relaxes
toward it. The mean fitness converges to the analytic Perron eigenvalue.

This is the smallest experiment in the repository and the fastest way to check
that an installation works.

## Full run (~1 s)

    propgen sweep --sweep experiments/meanfit/sweep.yaml --jobs 5
    propgen aggregate --raw results/meanfit/raw --out results/meanfit/summary.npz

5 trials at `pop_size=10000`, 200 dilution cycles, no mutation. There is no
swept parameter, so `--quick` differs only in scale.

## Committed summary

`results/meanfit/summary.npz` (8 KB), full time resolution.

## Make the figure

    jupyter nbconvert --to notebook --execute --inplace experiments/meanfit/figure.ipynb

Writes `figures/meanfit.pdf`.
