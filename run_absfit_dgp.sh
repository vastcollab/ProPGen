#!/bin/bash

# number of parallel jobs
PARALLEL=32  

# define pairs of reproduction probabilities
# format: "r1 r2"
repro_pairs=(
    "0.001 0.004"
    "0.0022 0.0052"
    "0.0175 0.0205"
)

# sweep over reproduction probability pairs
for pair in "${repro_pairs[@]}"; do
    r1=$(echo $pair | awk '{print $1}')
    r2=$(echo $pair | awk '{print $2}')
    
    for trial in $(seq 1 10); do
        echo "python absfit_sims.py \
             --config=configs/absfit/absfit_dgp_sim.yaml \
             --repro_prob1=$r1 \
             --repro_prob2=$r2 \
             --trial=$trial \
             --out=/data/cb/asapp/VAST/sim_results/absfit/dgp"
    done
done | xargs -n 1 -P $PARALLEL -I {} bash -c "{}"