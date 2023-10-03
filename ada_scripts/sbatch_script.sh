#!/bin/bash

# set the input arguments
ALGO=$1
ENV=$2
MODEL=$3
CONDA=$4
RL_STORAGE=$5
SCRIPT=$6

# source conda environment
source $CONDA
conda activate rl_env

# set storage path environment variable
export RL_STORAGE=$RL_STORAGE
echo "RL_STORAGE=$RL_STORAGE"

# print the input values
echo "Script: $SCRIPT, Conda: $CONDA, RL Storage: $RL_STORAGE, Algo: $ALGO, Env: $ENV, Model: $MODEL"

# run the training script
python3 $SCRIPT --algo $ALGO --env $ENV --model $MODEL
# wait for all of the runs to complete before exiting
wait