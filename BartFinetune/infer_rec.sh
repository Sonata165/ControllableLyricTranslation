#!/usr/bin/env bash

if [ "$#" -ne 1 ]; then
  echo "Illegal number of parameters. Please specify hparam file."
  exit
fi
. ./$1

#. ./hparams.sh

CUDA_VISIBLE_DEVICES=$DEVICE \
  python inference_rec.py \
  $MODEL_NAME_OR_PATH \
  $REAL_TEST_INPUT_PATH \
  $REAL_TEST_OUTPUT_FILE_PATH \
  --model_class_name $MODEL_CLASS \
  --score_path $REAL_TEST_SCORE_PATH \
  --device cuda \
  --task translation \
  --bs $BS \
  --max_length $MAX_LEN \
  --constraint_path $REAL_CONSTRAINT_PATH \
  --tokenizer $TOKENIZER \
  --constraint_type source \
  --src_lang $SRC_LANG \
  --constraint_type source \
  --dataset_class $DATASET_CLASS \
  --num_beams 5 \
  --force no

#  --reference_path $REF_PATH \
#        ../results/full/baseline_doc_1/best_tfmr
#        ../Dataset/datasets/v6_doc_full/test.source
#        ../results/full/baseline_doc_1/test_output.txt
#        --model_class_name MBartForConditionalGenerationCharLevel
#        --reference_path ../Dataset/datasets/v6_doc_full/test.target
#        --score_path ../results/full/baseline_doc_1/test_score.txt
#        --device cuda
#        --task translation
#        --bs 7
#        --max_length 256
#        --constraint_path ../Dataset/datasets/v6_doc_full/constraints/reference/test.target
#        --tokenizer ./tokenizers/mbart_tokenizer_fast_ch_prefix_doc
