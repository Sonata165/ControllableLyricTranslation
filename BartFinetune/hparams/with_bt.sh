#!/usr/bin/env bash
# server: gpu2
DEVICE=2
SEED=42

# --------- For training ----------
TRAIN_SCRIPT=finetune.py
LOGGER=default # default | wandb

TOKENIZER='./tokenizers/mbart_tokenizer_fast_ch_prefix_lrs2'
MODEL_CLASS=MBartForConditionalGenerationCharLevel
MODEL="facebook/mbart-large-50"
SHORT_MODEL_NAME=boundary_encoder_prefix_rev
ATTEMP=1

DATASET_VER=data_bt
DATASET_CLASS=Seq2SeqDatasetPrefixEncoderBdr
CONSTRAINT_TYPE=reference
SRC_LANG=en_XX
NUM_WORKERS=0

WARMUP_STEPS=2500
BS=40                   # fp32: GPU2: gpu:32   |   fp16:  48  (4:15/epoch), 80 64(when)  （2:43/epoch）
VAL_CHECK_INTERVAL=0.25 # 0.25
EPOCHS=5                # 5
LR=3e-5                 # 3e-5 default
EPS=1e-06
LR_SCHEDULER=linear
DROPOUT=0.0
LABEL_SMOOTHING=0.0

# export PYTHONPATH="../":"${PYTHONPATH}"
MAX_LEN=50
DATASET_DIR="../Dataset/datasets/${DATASET_VER}"
OUTPUT_DIR="../results/${SHORT_MODEL_NAME}_${DATASET_VER}_${ATTEMP}"

# --------- For testing ----------
TEST_SCRIPT=inference.py

LENGTH_TARGET=tgt # src | tgt
TEST_BOS_TOKEN_ID=250025
TEST_SRC_LANG=en_XX # zh_CN
TEST_BS=80
FORCE=no # length | rhyme | no

TEST_DATASET_DIR="../Dataset/datasets/${DATASET_VER}"
TEST_CONSTRAINT_TYPE=source # reference | source | random
MODEL_NAME_OR_PATH=${OUTPUT_DIR}/best_tfmr
TEST_INPUT_PATH=${TEST_DATASET_DIR}/test.source
REF_PATH=${TEST_DATASET_DIR}/test.target
CONSTRAINT_PATH=${TEST_DATASET_DIR}/constraints/${TEST_CONSTRAINT_TYPE}/test.target
TEST_OUTPUT_PATH=${OUTPUT_DIR}/testset\=${TEST_DATASET_DIR}/force=${FORCE}/test_constraint\=${TEST_CONSTRAINT_TYPE}_output.txt
TEST_SCORE_PATH=${OUTPUT_DIR}/testset\=${TEST_DATASET_DIR}/force=${FORCE}/test_constraint\=${TEST_CONSTRAINT_TYPE}_scores.txt

# ---------- For testing with real English lyrics -----------
REAL_TEST_DATASET_DIR="../Dataset/datasets/test_real_lyrics"
REAL_TEST_INPUT_PATH=${REAL_TEST_DATASET_DIR}/test.source
REAL_CONSTRAINT_PATH=${REAL_TEST_DATASET_DIR}/constraints.txt
REAL_TEST_OUTPUT_PATH=${OUTPUT_DIR}/test_real_output.txt
REAL_TEST_SCORE_PATH=${OUTPUT_DIR}/test_real_scores_.txt
