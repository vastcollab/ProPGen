# Coexistence phase diagram

## What this shows

Phenotypic buoying produces regimes in which several genotype-phenotype pairs
coexist, and the *ordering* of their equilibrium frequencies changes with the
mutation rate and the genotype-phenotype map. Each distinct ordering is a
phase. Over a 100 x 100 grid in (mu, phi) on the buoying landscape the theory
resolves ten of them.

**This is exact and analytic.** `propgen.phase_diagram` evaluates the
Perron-Frobenius equilibrium at every grid point; no simulation is involved
and no stored data is needed.

## Make the figure (~3 s)

    jupyter nbconvert --to notebook --execute --inplace experiments/phasediag/figure.ipynb

Writes `figures/phasediag.pdf` and `figures/phasediag_frequencies.pdf`.

## From Python

    from propgen import Landscape, phase_diagram

    diagram = phase_diagram(
        landscape,
        mutation_rates=np.linspace(0.005, 0.5, 100),
        pheno_prob_values=np.linspace(0.0, 1.0, 100),
        genotype=1,
    )
    diagram.phase        # (100, 100) integer phase labels
    diagram.orderings    # the distinct rank orderings
    diagram.f_eq         # (100, 100, Ng*Np) equilibrium frequencies

## Numerical verification

The paper also verifies the phase structure numerically. That sweep is a
two-dimensional grid over (mu, phi) with a full ProSeD run at each point, and
is expensive enough that it is not included here as a runnable experiment; the
analytic diagram above is the object the paper derives, and the simulation
agreement is demonstrated separately by `experiments/buoy`, where simulated
and analytic equilibria agree to within 0.011 in frequency across the sweep.
