# Absolute fitness is not neutral under phenotypic uncertainty

## What this shows

Classical population genetics holds that at fixed population size the dynamics
depend only on fitness *differences*, not on the absolute level. This
experiment holds the difference between the two phenotypes fixed at 0.003
while shifting the absolute level across three settings:

    (0.001, 0.004)   (0.0022, 0.0052)   (0.0175, 0.0205)

and runs the same sweep under two genotype-phenotype maps:

* `dgp`  -- deterministic, each genotype expresses one phenotype with certainty
* `prgp` -- probabilistic, each genotype expresses a distribution

Under DGP the trajectories are invariant to the shift, as classical theory
predicts. Under PrGP they are not. The figure notebook prints the spread of
final frequencies across the three levels: 0.0000 for DGP, 0.0054 for PrGP.

## Quick smoke test (~2 s)

    propgen sweep --sweep experiments/absfit/sweep_dgp.yaml  --quick --jobs 8
    propgen sweep --sweep experiments/absfit/sweep_prgp.yaml --quick --jobs 8

## Full run (~16 s on 1 core)

    propgen sweep --sweep experiments/absfit/sweep_dgp.yaml  --jobs 16
    propgen sweep --sweep experiments/absfit/sweep_prgp.yaml --jobs 16
    propgen aggregate --raw results/absfit/raw_dgp  --out results/absfit/summary_dgp.npz
    propgen aggregate --raw results/absfit/raw_prgp --out results/absfit/summary_prgp.npz

30 runs each at `pop_size=10000`, 500 dilution cycles, no mutation.

## Committed summaries

`results/absfit/summary_dgp.npz` (16 KB), `summary_prgp.npz` (44 KB).

## Make the figure

    jupyter nbconvert --to notebook --execute --inplace experiments/absfit/figure.ipynb

Writes `figures/absfit.pdf`.
