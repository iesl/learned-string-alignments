#!/usr/bin/env bash
set -exu

gpu=${1:-None}

sh bin/train/train.sh config/wiki/align_linear_wiki.json wiki AlignLinear $gpu
