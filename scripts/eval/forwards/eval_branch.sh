#!/bin/bash

# Total number of branch indices
# [5, 102]
# [30, 584]
TOTAL_BRANCHES=584

# Number of available GPUs
NUM_GPUS=8

# Function to run the command
run_command() {
    local gpu=$1
    local branch=$2
    CUDA_VISIBLE_DEVICES=$gpu python -m llava.eval.forwards.infer_mme \
            --branch-idx $branch \
            --mask-array ./mask_variations_30.npy \
            --save-path data/MME/ada_losses/fullset/mask_30
}

# Loop through all branch indices
for ((branch=0; branch<TOTAL_BRANCHES; branch++)); do
    # Calculate which GPU to use
    gpu=$((branch % NUM_GPUS))
    
    # Run the command in the background
    run_command $gpu $branch &
    
    # If we've started tasks on all GPUs, wait for them to finish
    if ((branch % NUM_GPUS == NUM_GPUS - 1)) || ((branch == TOTAL_BRANCHES - 1)); then
        wait
    fi
done

# Wait for any remaining background jobs to finish
wait

echo "All tasks completed."