#!/bin/bash

# set the input arguments
ALGO=$1
ENV=$2
MODEL=$3
SAVE_INTERVAL=$4
FRAMES=$5
CONDA=$6
RL_STORAGE=$7
SCRIPT=$8

# source conda environment
source $SOURCE
conda activate rl_env

# set storage path environment variable
export RL_STORAGE=$RL_STORAGE
echo "RL_STORAGE=$RL_STORAGE"

# print the input values
echo "Script: $SCRIPT, Algo: $ALGO, Env: $ENV, Model: $MODEL, Save Interval: $SAVE_INTERVAL, Frames: $FRAMES"

# run the training script
python3 $SCRIPT --algo $ALGO --env $ENV --model $MODEL --save-interval $SAVE_INTERVAL --frames $FRAMES

# wait for all of the runs to complete before exiting
wait