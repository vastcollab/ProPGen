# Rugged fitness landscape schematic

## What this shows

An illustrative rugged landscape, drawn as a sum of random Gaussian bumps in
3D and as a 2D heatmap. It is a diagram, not a result: no simulation output is
involved, and it is deterministic given `SEED = 33`.

## Make the figure (~3 s)

    jupyter nbconvert --to notebook --execute --inplace experiments/landscape_diagram/figure.ipynb

Writes `figures/landscape_diagram_3d.pdf` and `figures/landscape_diagram_2d.pdf`.
