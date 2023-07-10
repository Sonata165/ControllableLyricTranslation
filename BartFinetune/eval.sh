#!/usr/bin/env bash

# Load hyperparameters
if [ "$#" -ne 1 ]; then
  echo "Illegal number of parameters. Please specify hparam file."
  exit
fi
. ./$1

CUDA_VISIBLE_DEVICES=$DEVICE \
  python compute_metrics.py \
  --source_path $TEST_INPUT_PATH \
  --output_path $TEST_OUTPUT_PATH \
  --constraint_path $CONSTRAINT_PATH \
  --reference_path $REF_PATH \
  --score_path $TEST_SCORE_PATH \
