#!/usr/bin/env bash
# Reproduce every figure from the committed summaries. No simulation is run.
# Takes about one minute; writes PDFs to figures/.
set -euo pipefail
cd "$(dirname "$0")/.."

for nb in experiments/*/figure.ipynb; do
    echo "--- $nb"
    jupyter nbconvert --to notebook --execute --inplace \
        --ExecutePreprocessor.timeout=600 "$nb"
done

echo
echo "Figures written to figures/:"
ls -1 figures/
