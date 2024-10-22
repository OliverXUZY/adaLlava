#!/bin/bash

# Total number of branch indices
# [5, 102]
# [30, 584]

# START_BRANCH=383
# TOTAL_BRANCHES=584

START_BRANCH=0
TOTAL_BRANCHES=28

NUM_BRANCHES=28

# Number of available GPUs
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
NUM_GPUS=8

# model_path="checkpoints/adaft/llava-onevision-llavanext-si_alldata_0_4cuda_0.3p/checkpoint-10030"
model_path="checkpoints/adaft/llava-onevision-llavanext-si_MME_0_8cuda_0.3p"
# Function to run the command
run_command() {
    local gpu=$1
    local branch=$2
    echo "Running branch $branch on GPU $gpu"
    CUDA_VISIBLE_DEVICES=$gpu python -m llava.eval.forwards.infer_adascheduler_mme \
            --latency-idx $branch \
            --mask-latency ./latency_variations_${NUM_BRANCHES}.npy \
            --save-path data/MME/ada_losses/fullset/latency_${NUM_BRANCHES} \
            --model-path $model_path \
            2> >(tee -a "./scripts/adaeval/forwards/eval_latency_stderr.log" >&2) | tee -a "./scripts/adaeval/forwards/eval_latency_stdout.log"
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