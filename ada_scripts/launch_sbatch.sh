#!/bin/bash

# define the experiments
colnames=("algo" "env" "model" "save_interval" "frames")
experiments=(
    "ppo MiniGrid-DoorKey-5x5-v0 DoorKey 100 100000000"
) 
storage='/home/aberlie1/ada_user/data/rl-starter-files/'
source='/home/aberlie1/ada_user/miniconda3/etc/profile.d/conda.sh'

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
    save_interval=${row_array[${column_indices["save_interval"]}]}
    frames=${row_array[${column_indices["frames"]}]}

    # submit slurm job
    sbatch sbatch_script.sh ${algo} ${env} ${model} ${save_interval} ${frames} ${source} ${storage}
done