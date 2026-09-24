# Persister resuscitation and partitioning

## What this shows

The diffusion limit of ProP Gen theory applied to bacterial persistence. A
dormant persister population resuscitates at an accelerating rate
`S(t) = alpha * exp(beta * t)` and partitions into failed, damaged and healthy
cells; damaged cells convert to healthy at division. The resulting linear
system has a closed-form solution in terms of lower incomplete gamma
functions, implemented in `model.py`.

**This figure requires no simulation and no stored data.** It runs in about two
seconds from a fresh clone.

The partitioning probabilities are measurements from Fang & Allison; the growth
and resuscitation rates are the estimates used in the paper. All are listed in
`model.PARAMETERS`.

## Make the figure (~2 s)

    jupyter nbconvert --to notebook --execute --inplace experiments/persister/figure.ipynb

Writes `figures/persister_fractions.pdf`.

## Related simulation

`configs/persister.yaml` sets up the corresponding discrete ProSeD model: one
genotype, two phenotypes, no mutation, with antibiotic applied at cycle 100
via an environment schedule. It is not used by any figure in the paper, but it
is the worked example of environment switching:

    propgen run --config configs/persister.yaml --out results/persister_demo.npz
