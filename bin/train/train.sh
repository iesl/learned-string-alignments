#!/usr/bin/env bash

set -exu

config=$1
dataset_name=$2
model=$3
gpu=${4:-"None"}

if [ $gpu != "None" ]; then
    CUDA_VISIBLE_DEVICES=$gpu $PYTHON_EXEC -m entity_align.train.TrainModel $config $dataset_name $model
else
    $PYTHON_EXEC -m entity_align.train.TrainModel $config $dataset_name $model
fi
