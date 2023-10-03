#!/bin/bash

# set the input arguments
ALGO=$1
ENV=$2
MODEL=$3
SAVE_INTERVA=$4
FRAME=$5
CONDA=$6
RL_STORAGE=$7

# source conda environment
source $SOURCE
conda activate rl_env

# set storage path environment variable
export RL_STORAGE=$RL_STORAGE
echo "RL_STORAGE=$RL_STORAGE"

# print the input values
echo "Algo: $algo, Env: $env, Model: $model, Save Interval: $save_interval, Frames: $frames"

# run the training script
python3 -m train.py --algo ALGO --env ENV --model MODEL --save-interval SAVE_INTERVAL --frames FRAMES

# wait for all of the runs to complete before exiting
wait