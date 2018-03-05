#!/usr/bin/env bash

set -exu

config=$1
model_file=$2
gpu=$3

CUDA_VISIBLE_DEVICES=$gpu $PYTHON_EXEC -m entity_align.eval.EvalModelTest $config $model_file