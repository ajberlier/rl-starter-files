#!/bin/bash

# define the experiments
script='/home/aberlie1/rl-starter-files/train.py'
colnames=("algo" "env" "model" "save_interval" "frames")
experiments=(
    "ppo BabyAI-GoToRedBallGrey-v0 BabyAI-GoToRedBallGrey-v0_PPO”
    "ppo BabyAI-GoToRedBall-v0 BabyAI-GoToRedBall-v0_PPO”
    "ppo BabyAI-GoToRedBallNoDists-v0 BabyAI-GoToRedBallNoDists-v0_PPO”
    "ppo BabyAI-GoToObjS6-v0 BabyAI-GoToObjS6-v0_PPO”
    "ppo BabyAI-GoToLocalS8N7-v0 BabyAI-GoToLocalS8N7-v0_PPO”
    "ppo BabyAI-GoToObjMazeS7-v0 BabyAI-GoToObjMazeS7-v0_PPO”
    "ppo BabyAI-GoToImpUnlock-v0 BabyAI-GoToImpUnlock-v0_PPO”
    "ppo BabyAI-GoToSeqS5R2-v0 BabyAI-GoToSeqS5R2-v0_PPO”
    "ppo BabyAI-GoToRedBlueBall-v0 BabyAI-GoToRedBlueBall-v0_PPO”
    "ppo BabyAI-GoToDoor-v0 BabyAI-GoToDoor-v0_PPO”
    "ppo BabyAI-GoToObjDoor-v0 BabyAI-GoToObjDoor-v0_PPO”
    "ppo BabyAI-Open-v0 BabyAI-Open-v0_PPO”
    "ppo BabyAI-OpenRedDoor-v0 BabyAI-OpenRedDoor-v0_PPO”
    "ppo BabyAI-OpenDoorLoc-v0 BabyAI-OpenDoorLoc-v0_PPO”
    "ppo BabyAI-OpenRedBlueDoorsDebug-v0 BabyAI-OpenRedBlueDoorsDebug-v0_PPO”
    "ppo BabyAI-OpenDoorsOrderN4Debug-v0 BabyAI-OpenDoorsOrderN4Debug-v0_PPO”
    "ppo BabyAI-Pickup-v0 BabyAI-Pickup-v0_PPO”
    "ppo BabyAI-UnblockPickup-v0 BabyAI-UnblockPickup-v0_PPO”
    "ppo BabyAI-PickupLoc-v0 BabyAI-PickupLoc-v0_PPO”
    "ppo BabyAI-PickupDistDebug-v0 BabyAI-PickupDistDebug-v0_PPO”
    "ppo BabyAI-PickupAbove-v0 BabyAI-PickupAbove-v0_PPO”
    "ppo BabyAI-PutNextLocalS6N4-v0 BabyAI-PutNextLocalS6N4-v0_PPO”
    "ppo BabyAI-PutNextS7N4Carrying-v0 BabyAI-PutNextS7N4Carrying-v0_PPO”
    "ppo BabyAI-Unlock-v0 BabyAI-Unlock-v0_PPO”
    "ppo BabyAI-UnlockLocalDist-v0 BabyAI-UnlockLocalDist-v0_PPO”
    "ppo BabyAI-KeyInBox-v0 BabyAI-KeyInBox-v0_PPO”
    "ppo BabyAI-UnlockPickupDist-v0 BabyAI-UnlockPickupDist-v0_PPO”
    "ppo BabyAI-BlockedUnlockPickup-v0 BabyAI-BlockedUnlockPickup-v0_PPO”
    "ppo BabyAI-UnlockToUnlock-v0 BabyAI-UnlockToUnlock-v0_PPO”
    "ppo BabyAI-ActionObjDoor-v0 BabyAI-ActionObjDoor-v0_PPO”
    "ppo BabyAI-FindObjS7-v0 BabyAI-FindObjS7-v0_PPO”
    "ppo BabyAI-KeyCorridorS6R3-v0 BabyAI-KeyCorridorS6R3-v0_PPO”
    "ppo BabyAI-OneRoomS20-v0 BabyAI-OneRoomS20-v0_PPO”
    "ppo BabyAI-MoveTwoAcrossS8N9-v0 BabyAI-MoveTwoAcrossS8N9-v0_PPO”
    "ppo BabyAI-SynthS5R2-v0 BabyAI-SynthS5R2-v0_PPO”
    "ppo BabyAI-SynthLoc-v0 BabyAI-SynthLoc-v0_PPO”
    "ppo BabyAI-SynthSeq-v0 BabyAI-SynthSeq-v0_PPO”
    "ppo BabyAI-MiniBossLevel-v0 BabyAI-MiniBossLevel-v0_PPO”
    "ppo BabyAI-BossLevel-v0 BabyAI-BossLevel-v0_PPO”
    "ppo BabyAI-BossLevelNoUnlock-v0 BabyAI-BossLevelNoUnlock-v0_PPO”
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
    sbatch --mem=5000 --cpus-per-task=1 --gres=gpu:1 --time=72:00:00 sbatch_script.sh ${algo} ${env} ${model} ${conda} ${storage} ${script}
done