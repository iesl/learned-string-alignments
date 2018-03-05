#!/usr/bin/env bash

set -exu

inputfile=$1
outputfile=$2
mincount=$3

$PYTHON_EXEC -m entity_align.utils.MakeVocab $inputfile $outputfile $mincount
