#!/usr/bin/env bash
set -exu

gpu=${1:-None}

sh bin/train/train.sh config/wiki/align_dot_wiki.json wiki AlignDot $gpu
