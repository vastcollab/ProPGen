#!/bin/bash

# number of parallel jobs
PARALLEL=32  

# sweep over mapping probability
for map in $(seq 0 0.1 1); do
  for trial in $(seq 1 10); do
    echo "python buoy_sims.py \
         --config=configs/buoy/buoy_sims.yaml \
         --mapping_prob=$map \
         --trial=$trial \
         --out=/data/cb/asapp/VAST/sim_results/buoy"
  done
done | xargs -n 1 -P $PARALLEL -I {} bash -c "{}"
