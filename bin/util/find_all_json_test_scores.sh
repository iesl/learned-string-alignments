#!/usr/bin/env bash

set -exu

exp_dir=$1

find $exp_dir -name 'test.scores.json' -exec cat {} \;
