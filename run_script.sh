#!/bin/bash
GPUID=$1
echo "Run on GPU $GPUID"
TRAIN=$3
TEST=$4
# data
DATASET=$2
PROJECT_ROOT=$(dirname "$(readlink -f "$0")")
DATA_ROOT=$PROJECT_ROOT/dataset/

# model
TOKENIZER_TYPE=bert
# TEACHER_TYPE=bert
SPAN_TYPE=bert
TYPE_TYPE=bert
# STUDENT2_TYPE=roberta
TOKENIZER_NAME=bert-base-cased
# TEACHER_MODEL_NAME=bert-base-uncased
SPAN_MODEL_NAME=bert-base-cased
TYPE_MODEL_NAME=bert-base-cased
# SPAN_MODEL_NAME=/data/zhangxinghua/Cycle-Consistent/Teacher/ptms/music-t1/checkpoint-best-span/
# TYPE_MODEL_NAME=/data/zhangxinghua/Cycle-Consistent/Teacher/ptms/music-t1/checkpoint-best-type/
# SPAN_MODEL_NAME=/data/zhangxinghua/Meta-Cross-NER/source-domain/ptms/conll2003/checkpoint-best
# TYPE_MODEL_NAME=/data/zhangxinghua/Meta-Cross-NER/source-domain/ptms/conll2003/checkpoint-best
# SPAN_MODEL_NAME=/data/zhangxinghua/Meta-Cross-NER/ptms/politics/checkpoint-best
# TYPE_MODEL_NAME=/data/zhangxinghua/Meta-Cross-NER/ptms/politics/checkpoint-best
# STUDENT2_MODEL_NAME=roberta-base

DELTA_SPAN=$5
DELTA_TYPE=$6
# params
LR=1e-5
WEIGHT_DECAY=1e-4
EPOCH=50
SEED=0

ADAM_EPS=1e-8
ADAM_BETA1=0.9
ADAM_BETA2=0.98
WARMUP=500

TRAIN_BATCH=16
TRAIN_BATCH_META=16
TRAIN_BATCH_INTER=16
EVAL_BATCH=32

MU=1.0
ALPHA=0.9
BETA=0.1

# output
WARM_EPOCH=50
META_UPDATE_STEP=100
OUTPUT=$PROJECT_ROOT/ptms/$DATASET/

CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPUID python3 -u run_script.py --data_dir $DATA_ROOT \
  --span_model_name_or_path $SPAN_MODEL_NAME \
  --type_model_name_or_path $TYPE_MODEL_NAME \
  --output_dir $OUTPUT \
  --tokenizer_name_or_path $TOKENIZER_NAME \
  --cache_dir $PROJECT_ROOT/cached_models \
  --max_seq_length 128 \
  --learning_rate $LR \
  --weight_decay $WEIGHT_DECAY \
  --adam_epsilon $ADAM_EPS \
  --adam_beta1 $ADAM_BETA1 \
  --adam_beta2 $ADAM_BETA2 \
  --max_grad_norm 1.0 \
  --num_train_epochs $EPOCH \
  --warmup_steps $WARMUP \
  --per_gpu_train_batch_size $TRAIN_BATCH \
  --per_gpu_train_batch_size_meta $TRAIN_BATCH_META \
  --per_gpu_train_batch_size_inter $TRAIN_BATCH_INTER \
  --per_gpu_eval_batch_size $EVAL_BATCH \
  --gradient_accumulation_steps 1 \
  --logging_steps 100 \
  --save_steps 100000 \
  --do_train $TRAIN\
  --do_test $TEST \
  --evaluate_during_training \
  --seed $SEED \
  --overwrite_output_dir \
  --model_type $TOKENIZER_TYPE \
  --dataset $DATASET \
  --src_dataset conll2003 \
  --warm_num_train_epochs $WARM_EPOCH \
  --meta_update_steps $META_UPDATE_STEP \
  --delta_span $DELTA_SPAN \
  --delta_type $DELTA_TYPE \
  --mu $MU \
  --alpha $ALPHA \
  --beta $BETA \

