#!/bin/bash

export PYTHONPATH=..

mkdir -p results

# number of parallel jobs
PARALLEL=32  

# sweep over mapping probability
for trial in $(seq 1 5); do
  echo "python meanfit_sim.py \
      --config=meanfit.yaml \
      --trial=$trial \
      --out=results"
    done | xargs -n 1 -P $PARALLEL -I {} bash -c "{}"
