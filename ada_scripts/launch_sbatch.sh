#!/bin/bash

# define the experiments
script='/home/aberlie1/rl-starter-files/train.py'
colnames=("algo" "env" "model" "save_interval" "frames")
experiments=(
    "ppo BabyAI-GoToObjS6-v0 BabyAI-GoToObj_PPO"
    "ppo BabyAI-GoToLocalS8N7-v0 BabyAI-GoToLocal_PPO"
    "ppo BabyAI-GoToObjMazeS7-v0 BabyAI-GoTo_PPO"
    "ppo BabyAI-PickupDist BabyAI-PickupDist_PPO"
    "ppo BabyAI-PutNextS7N4Carrying-v0 BabyAI-PutNext_PPO"
    "ppo BabyAI-UnlockPickup BabyAI-UnlockPickup_PPO"
) 
storage=/home/aberlie1/ada_user/data/rl-starter-files/
conda=/home/aberlie1/ada_user/miniconda3/etc/profile.d/conda.sh

# initialize an associative array
declare -A column_indices

# populate the associative array with column indices
for i in "${!colnames[@]}"
do
    column_indices[${colnames[$i]}]=$i
done

# loop through the eperiment data rows
for row in "${experiments[@]}"
do
    # split the row into an array
    row_array=($row)

    # access columns by header
    algo=${row_array[${column_indices["algo"]}]}
    env=${row_array[${column_indices["env"]}]}
    model=${row_array[${column_indices["model"]}]}

    # submit slurm job
    sbatch --mem=5000 --cpus-per-task=1 --gres=gpu:1 --time=24:00 sbatch_script.sh ${algo} ${env} ${model} ${conda} ${storage} ${script}
done