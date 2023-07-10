#!/usr/bin/env bash

# Load hyperparameters
if [ "$#" -ne 1 ]; then
  echo "Illegal number of parameters. Please specify hparam file."
  exit
fi
. ./$1

CUDA_VISIBLE_DEVICES=$DEVICE \
  python $TEST_SCRIPT \
  $MODEL_NAME_OR_PATH \
  $TEST_INPUT_PATH \
  $TEST_OUTPUT_PATH \
  --model_class_name $MODEL_CLASS \
  --reference_path $REF_PATH \
  --score_path $TEST_SCORE_PATH \
  --device cuda \
  --task translation \
  --bs $TEST_BS \
  --max_length $MAX_LEN \
  --constraint_path $CONSTRAINT_PATH \
  --tokenizer $TOKENIZER \
  --src_lang $TEST_SRC_LANG \
  --constraint_type $TEST_CONSTRAINT_TYPE \
  --dataset_class $DATASET_CLASS \
  --num_beams 5 \
  --force $FORCE \

CUDA_VISIBLE_DEVICES=$DEVICE \
  python compute_metrics.py \
  --source_path $TEST_INPUT_PATH \
  --output_path $TEST_OUTPUT_PATH \
  --constraint_path $CONSTRAINT_PATH \
  --reference_path $REF_PATH \
  --score_path $TEST_SCORE_PATH \

