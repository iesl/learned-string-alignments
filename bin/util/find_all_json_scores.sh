#!/usr/bin/env bash

set -exu

exp_dir=$1

find $exp_dir -name 'dev.scores.*.json' -exec cat {} \;
