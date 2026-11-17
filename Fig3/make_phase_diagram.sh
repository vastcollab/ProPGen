#!/bin/bash

export PYTHONPATH=..

outdir="results"
mkdir -p $outdir

# number of parallel jobs
PARALLEL=64  

# sweep over mu and phi
for mut in $(seq 0 0.01 1); do
  for map in $(seq 0 0.01 1); do
    mut_offset=$(echo "$mut + 0.001" | bc -l)
    map_offset=$(echo "$map + 0.001" | bc -l)  
    echo "python sim_phasediag.py \
         --config=config_phasediag.yaml \
         --mutation_rate=$mut \
         --mapping_prob=$map"
  done
done | xargs -n 1 -P $PARALLEL -I {} bash -c "{}"
