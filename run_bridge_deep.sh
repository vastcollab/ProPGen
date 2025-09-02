#!/bin/bash

# number of parallel jobs
PARALLEL=36  

# sweep over mapping probability, repro_prob, and trials
for map in $(seq 0 0.05 1); do
  for repro in 0.0002; do
    for trial in $(seq 1 100); do
      echo "python bridge_sims.py \
           --config=configs/bridge/bridge_sims.yaml \
           --mapping_prob=$map \
           --repro_prob=$repro \
           --trial=$trial \
           --out=/data/cb/asapp/VAST/sim_results/bridge"
    done
  done
done | xargs -n 1 -P $PARALLEL -I {} bash -c "{}"
