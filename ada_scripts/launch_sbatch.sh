#!/bin/bash

# define the experiments
script='/home/aberlie1/rl-starter-files/train.py'
colnames=("arch" "algo" "env" "model")
experiments=(
    "ac ppo BabyAI-UnlockLocal-v0 BabyAI-UnlockLocal-v0_PPO"
    "oc ppo BabyAI-UnlockLocal-v0 BabyAI-UnlockLocal-v0_PPOC"
    ) 
storage='/home/aberlie1/ada_user/data/rl-starter-files/'
conda='/home/aberlie1/ada_user/miniconda3/etc/profile.d/conda.sh'

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
    arch=${row_arrau[${column_indices["arch"]}]}
    algo=${row_array[${column_indices["algo"]}]}
    env=${row_array[${column_indices["env"]}]}
    model=${row_array[${column_indices["model"]}]}

    # submit slurm job
    sbatch --mem=5000 --cpus-per-task=1 --gres=gpu:1 --time=72:00:00 sbatch_script.sh ${arch} ${algo} ${env} ${model} ${conda} ${storage} ${script}
done