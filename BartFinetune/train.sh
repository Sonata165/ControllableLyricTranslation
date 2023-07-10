#!/usr/bin/env bash

if [ "$#" -ne 1 ]; then
  echo "Illegal number of parameters. Please specify hparam file."
  exit
fi
. ./$1
#. ./hparams.sh

CUDA_VISIBLE_DEVICES=$DEVICE \
python $TRAIN_SCRIPT \
    --learning_rate=$LR \
    --val_check_interval=$VAL_CHECK_INTERVAL \
    --adam_eps $EPS \
    --num_train_epochs $EPOCHS \
    --tokenizer $TOKENIZER \
    --src_lang $SRC_LANG \
    --tgt_lang zh_CN \
    --data_dir $DATASET_DIR \
    --constraint_type $CONSTRAINT_TYPE \
    --max_source_length $MAX_LEN \
    --max_target_length $MAX_LEN \
    --val_max_target_length $MAX_LEN \
    --test_max_target_length $MAX_LEN \
    --eval_max_gen_length $MAX_LEN \
    --train_batch_size=$BS \
    --eval_batch_size=$BS \
    --task translation \
    --warmup_steps $WARMUP_STEPS \
    --freeze_embeds \
    --model_name_or_path=$MODEL \
    --output_dir=$OUTPUT_DIR \
    --do_train \
    --logger_name $LOGGER \
    --accumulate_grad_batches=1 \
    --model_class_name $MODEL_CLASS \
    --dataset_class $DATASET_CLASS \
    --lr_scheduler $LR_SCHEDULER \
    --dropout $DROPOUT \
    --label_smoothing $LABEL_SMOOTHING \
    --num_workers $NUM_WORKERS \
    --seed $SEED
#    "$@"

#    --do_predict \
# en_XX \