#!/bin/bash

# Total number of branch indices
# [5, 102]
# [30, 584]

# START_BRANCH=383
# TOTAL_BRANCHES=584

START_BRANCH=0
TOTAL_BRANCHES=102

# Number of available GPUs
NUM_GPUS=8

# Function to run the command
run_command() {
    local gpu=$1
    local branch=$2
    echo "Running branch $branch on GPU $gpu"
    CUDA_VISIBLE_DEVICES=$gpu python -m llava.eval.forwards.infer_mme \
            --branch-idx $branch \
            --mask-array ./mask_variations_5.npy \
            --save-path data/MME/ada_losses/fullset/mask_5
}

# Loop through all branch indices
for ((branch=START_BRANCH; branch<TOTAL_BRANCHES; branch++)); do
    # Calculate which GPU to use
    gpu=$(((branch - START_BRANCH) % NUM_GPUS))
    
    # Run the command in the background
    run_command $gpu $branch &
    
    # If we've started tasks on all GPUs, wait for them to finish
    if (((branch - START_BRANCH) % NUM_GPUS == NUM_GPUS - 1)) || ((branch == TOTAL_BRANCHES - 1)); then
        wait
    fi
done

# Wait for any remaining background jobs to finish
wait

echo "All tasks completed."