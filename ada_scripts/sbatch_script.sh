#!/bin/bash

# set the input arguments
ARCH=$1
ALGO=$2
ENV=$3
MODEL=$4
CONDA=$5
RL_STORAGE=$6
SCRIPT=$7

# source conda environment
source $CONDA
conda activate rl_env

# set storage path environment variable
export RL_STORAGE=$RL_STORAGE
echo "RL_STORAGE=$RL_STORAGE"

# print the input values
echo "Script: $SCRIPT, Conda: $CONDA, RL Storage: $RL_STORAGE, Arch: $ARCH, Algo: $ALGO, Env: $ENV, Model: $MODEL"

# run the training script
python3 $SCRIPT --arch $ARCH --algo $ALGO --env $ENV --model $MODEL
# wait for all of the runs to complete before exiting
wait