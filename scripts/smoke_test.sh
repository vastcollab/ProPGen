#!/usr/bin/env bash
# Exercise the whole pipeline end to end at reduced scale: simulate, aggregate
# and plot every experiment. Takes a few minutes on 8 cores and needs no
# pre-existing data.
#
# This verifies that the pipeline runs. It does NOT reproduce the published
# numbers -- for those, see each experiment's README.
set -euo pipefail
cd "$(dirname "$0")/.."

JOBS="${JOBS:-8}"

echo "=== propgen info ==="
propgen info

echo
echo "=== analytic equilibrium for every config ==="
for cfg in configs/*.yaml; do
    echo "--- $cfg"
    propgen theory --config "$cfg" | tail -3
done

echo
echo "=== reduced-scale sweeps ==="
for sweep in experiments/meanfit/sweep.yaml \
             experiments/buoy/sweep.yaml \
             experiments/absfit/sweep_dgp.yaml \
             experiments/absfit/sweep_prgp.yaml \
             experiments/bridge/sweep.yaml; do
    echo "--- $sweep"
    propgen sweep --sweep "$sweep" --quick --jobs "$JOBS"
done

echo
echo "=== aggregate ==="
propgen aggregate --raw results/meanfit/raw_quick     --out results/meanfit/summary_quick.npz
propgen aggregate --raw results/buoy/raw_quick        --out results/buoy/summary_quick.npz
propgen aggregate --raw results/absfit/raw_dgp_quick  --out results/absfit/summary_dgp_quick.npz
propgen aggregate --raw results/absfit/raw_prgp_quick --out results/absfit/summary_prgp_quick.npz
propgen aggregate --raw results/bridge/raw_quick      --out results/bridge/summary_quick.npz --thin 10

echo
echo "=== figures (from the committed full-scale summaries) ==="
bash scripts/make_figures.sh

echo
echo "smoke test passed"
