#!/bin/bash

# number of parallel jobs
PARALLEL=32  

# sweep over mapping probability
for trial in $(seq 1 5); do
  echo "python meanfit_sim.py \
      --config=configs/meanfit/meanfit.yaml \
      --trial=$trial \
      --out=/data/cb/asapp/VAST/sim_results/meanfit"
    done | xargs -n 1 -P $PARALLEL -I {} bash -c "{}"
