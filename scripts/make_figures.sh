#!/usr/bin/env bash
# Reproduce every figure from the committed summaries. No simulation is run.
# Takes about 30 seconds; writes PDFs to figures/.
#
# The notebooks are executed into figures/executed/ rather than in place, so
# running this leaves the working tree clean.
set -euo pipefail
cd "$(dirname "$0")/.."

mkdir -p figures/executed

for nb in experiments/*/figure.ipynb examples/quickstart.ipynb; do
    name="$(basename "$(dirname "$nb")")"
    echo "--- $nb"
    jupyter nbconvert --to notebook --execute \
        --ExecutePreprocessor.timeout=600 \
        --output-dir figures/executed --output "${name}.ipynb" \
        "$nb" 2>&1 | grep -vE 'IPKernelApp|findfont' || true
done

echo
echo "Figures written to figures/:"
ls -1 figures/*.pdf
