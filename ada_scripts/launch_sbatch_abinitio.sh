#!/bin/bash

# define the experiments
script="/home/aberlie1/rl-starter-files/train.py"
colnames=("arch" "algo" "env" "model")
experiments=(
    "ac ppo BabyAI-GoToRedBallGrey-v0 BabyAI-GoToRedBallGrey-v0_PPO"
    "oc ppo BabyAI-GoToRedBallGrey-v0 BabyAI-GoToRedBallGrey-v0_PPOC"
    "ac ppo BabyAI-GoToRedBall-v0 BabyAI-GoToRedBall-v0_PPO"
    "oc ppo BabyAI-GoToRedBall-v0 BabyAI-GoToRedBall-v0_PPOC"
    "ac ppo BabyAI-GoToRedBallNoDists-v0 BabyAI-GoToRedBallNoDists-v0_PPO"
    "oc ppo BabyAI-GoToRedBallNoDists-v0 BabyAI-GoToRedBallNoDists-v0_PPOC"
    "ac ppo BabyAI-GoToObjS6-v1 BabyAI-GoToObjS6-v1_PPO"
    "oc ppo BabyAI-GoToObjS6-v1 BabyAI-GoToObjS6-v1_PPOC"
    "ac ppo BabyAI-GoToLocalS8N7-v0 BabyAI-GoToLocalS8N7-v0_PPO"
    "oc ppo BabyAI-GoToLocalS8N7-v0 BabyAI-GoToLocalS8N7-v0_PPOC"
    "ac ppo BabyAI-GoToObjMazeS7-v0 BabyAI-GoToObjMazeS7-v0_PPO"
    "oc ppo BabyAI-GoToObjMazeS7-v0 BabyAI-GoToObjMazeS7-v0_PPOC"
    "ac ppo BabyAI-GoToImpUnlock-v0 BabyAI-GoToImpUnlock-v0_PPO"
    "oc ppo BabyAI-GoToImpUnlock-v0 BabyAI-GoToImpUnlock-v0_PPOC"
    "ac ppo BabyAI-GoToSeqS5R2-v0 BabyAI-GoToSeqS5R2-v0_PPO"
    "oc ppo BabyAI-GoToSeqS5R2-v0 BabyAI-GoToSeqS5R2-v0_PPOC"
    "ac ppo BabyAI-GoToRedBlueBall-v0 BabyAI-GoToRedBlueBall-v0_PPO"
    "oc ppo BabyAI-GoToRedBlueBall-v0 BabyAI-GoToRedBlueBall-v0_PPOC"
    "ac ppo BabyAI-GoToDoor-v0 BabyAI-GoToDoor-v0_PPO"
    "oc ppo BabyAI-GoToDoor-v0 BabyAI-GoToDoor-v0_PPOC"
    "ac ppo BabyAI-GoToObjDoor-v0 BabyAI-GoToObjDoor-v0_PPO"
    "oc ppo BabyAI-GoToObjDoor-v0 BabyAI-GoToObjDoor-v0_PPOC"
    "ac ppo BabyAI-Open-v0 BabyAI-Open-v0_PPO"
    "oc ppo BabyAI-Open-v0 BabyAI-Open-v0_PPOC"
    "ac ppo BabyAI-OpenRedDoor-v0 BabyAI-OpenRedDoor-v0_PPO"
    "oc ppo BabyAI-OpenRedDoor-v0 BabyAI-OpenRedDoor-v0_PPOC"
    "ac ppo BabyAI-OpenDoorLoc-v0 BabyAI-OpenDoorLoc-v0_PPO"
    "oc ppo BabyAI-OpenDoorLoc-v0 BabyAI-OpenDoorLoc-v0_PPOC"
    "ac ppo BabyAI-OpenRedBlueDoorsDebug-v0 BabyAI-OpenRedBlueDoorsDebug-v0_PPO"
    "oc ppo BabyAI-OpenRedBlueDoorsDebug-v0 BabyAI-OpenRedBlueDoorsDebug-v0_PPOC"
    "ac ppo BabyAI-OpenDoorsOrderN4Debug-v0 BabyAI-OpenDoorsOrderN4Debug-v0_PPO"
    "oc ppo BabyAI-OpenDoorsOrderN4Debug-v0 BabyAI-OpenDoorsOrderN4Debug-v0_PPOC"
    "ac ppo BabyAI-Pickup-v0 BabyAI-Pickup-v0_PPO"
    "oc ppo BabyAI-Pickup-v0 BabyAI-Pickup-v0_PPOC"
    "ac ppo BabyAI-UnblockPickup-v0 BabyAI-UnblockPickup-v0_PPO"
    "oc ppo BabyAI-UnblockPickup-v0 BabyAI-UnblockPickup-v0_PPOC"
    "ac ppo BabyAI-PickupLoc-v0 BabyAI-PickupLoc-v0_PPO"
    "oc ppo BabyAI-PickupLoc-v0 BabyAI-PickupLoc-v0_PPOC"
    "ac ppo BabyAI-PickupDistDebug-v0 BabyAI-PickupDistDebug-v0_PPO"
    "oc ppo BabyAI-PickupDistDebug-v0 BabyAI-PickupDistDebug-v0_PPOC"
    "ac ppo BabyAI-PickupAbove-v0 BabyAI-PickupAbove-v0_PPO"
    "oc ppo BabyAI-PickupAbove-v0 BabyAI-PickupAbove-v0_PPOC"
    "ac ppo BabyAI-PutNextLocalS6N4-v0 BabyAI-PutNextLocalS6N4-v0_PPO"
    "oc ppo BabyAI-PutNextLocalS6N4-v0 BabyAI-PutNextLocalS6N4-v0_PPOC"
    "ac ppo BabyAI-PutNextS7N4Carrying-v0 BabyAI-PutNextS7N4Carrying-v0_PPO"
    "oc ppo BabyAI-PutNextS7N4Carrying-v0 BabyAI-PutNextS7N4Carrying-v0_PPOC"
    "ac ppo BabyAI-Unlock-v0 BabyAI-Unlock-v0_PPO"
    "oc ppo BabyAI-Unlock-v0 BabyAI-Unlock-v0_PPOC"
    "ac ppo BabyAI-UnlockLocalDist-v0 BabyAI-UnlockLocalDist-v0_PPO"
    "oc ppo BabyAI-UnlockLocalDist-v0 BabyAI-UnlockLocalDist-v0_PPOC"
    "ac ppo BabyAI-KeyInBox-v0 BabyAI-KeyInBox-v0_PPO"
    "oc ppo BabyAI-KeyInBox-v0 BabyAI-KeyInBox-v0_PPOC"
    "ac ppo BabyAI-UnlockPickupDist-v0 BabyAI-UnlockPickupDist-v0_PPO"
    "oc ppo BabyAI-UnlockPickupDist-v0 BabyAI-UnlockPickupDist-v0_PPOC"
    "ac ppo BabyAI-BlockedUnlockPickup-v0 BabyAI-BlockedUnlockPickup-v0_PPO"
    "oc ppo BabyAI-BlockedUnlockPickup-v0 BabyAI-BlockedUnlockPickup-v0_PPOC"
    "ac ppo BabyAI-UnlockToUnlock-v0 BabyAI-UnlockToUnlock-v0_PPO"
    "oc ppo BabyAI-UnlockToUnlock-v0 BabyAI-UnlockToUnlock-v0_PPOC"
    "ac ppo BabyAI-ActionObjDoor-v0 BabyAI-ActionObjDoor-v0_PPO"
    "oc ppo BabyAI-ActionObjDoor-v0 BabyAI-ActionObjDoor-v0_PPOC"
    "ac ppo BabyAI-FindObjS7-v0 BabyAI-FindObjS7-v0_PPO"
    "oc ppo BabyAI-FindObjS7-v0 BabyAI-FindObjS7-v0_PPOC"
    "ac ppo BabyAI-KeyCorridorS6R3-v0 BabyAI-KeyCorridorS6R3-v0_PPO"
    "oc ppo BabyAI-KeyCorridorS6R3-v0 BabyAI-KeyCorridorS6R3-v0_PPOC"
    "ac ppo BabyAI-OneRoomS20-v0 BabyAI-OneRoomS20-v0_PPO"
    "oc ppo BabyAI-OneRoomS20-v0 BabyAI-OneRoomS20-v0_PPOC"
    "ac ppo BabyAI-MoveTwoAcrossS8N9-v0 BabyAI-MoveTwoAcrossS8N9-v0_PPO"
    "oc ppo BabyAI-MoveTwoAcrossS8N9-v0 BabyAI-MoveTwoAcrossS8N9-v0_PPOC"
    "ac ppo BabyAI-SynthS5R2-v0 BabyAI-SynthS5R2-v0_PPO"
    "oc ppo BabyAI-SynthS5R2-v0 BabyAI-SynthS5R2-v0_PPOC"
    "ac ppo BabyAI-SynthLoc-v0 BabyAI-SynthLoc-v0_PPO"
    "oc ppo BabyAI-SynthLoc-v0 BabyAI-SynthLoc-v0_PPOC"
    "ac ppo BabyAI-SynthSeq-v0 BabyAI-SynthSeq-v0_PPO"
    "oc ppo BabyAI-SynthSeq-v0 BabyAI-SynthSeq-v0_PPOC"
    "ac ppo BabyAI-MiniBossLevel-v0 BabyAI-MiniBossLevel-v0_PPO"
    "oc ppo BabyAI-MiniBossLevel-v0 BabyAI-MiniBossLevel-v0_PPOC"
    "ac ppo BabyAI-BossLevel-v0 BabyAI-BossLevel-v0_PPO"
    "oc ppo BabyAI-BossLevel-v0 BabyAI-BossLevel-v0_PPOC"
    "ac ppo BabyAI-BossLevelNoUnlock-v0 BabyAI-BossLevelNoUnlock-v0_PPO"
    "oc ppo BabyAI-BossLevelNoUnlock-v0 BabyAI-BossLevelNoUnlock-v0_PPOC"
    ) 
storage="/home/aberlie1/ada_user/data/rl-starter-files/"
conda="/home/aberlie1/ada_user/miniconda3/etc/profile.d/conda.sh"

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
    arch=${row_array[${column_indices["arch"]}]}
    algo=${row_array[${column_indices["algo"]}]}
    env=${row_array[${column_indices["env"]}]}
    model=${row_array[${column_indices["model"]}]}

    # submit slurm job
    sbatch --mem=5000 --cpus-per-task=1 --gres=gpu:1 --time=72:00:00 sbatch_script.sh ${arch} ${algo} ${env} ${model} ${conda} ${storage} ${script}
done