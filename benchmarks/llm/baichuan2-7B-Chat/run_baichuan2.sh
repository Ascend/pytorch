#!/bin/bash
export TORCHINDUCTOR_CACHE_DIR="./cache"
export ASCEND_RT_VISIBLE_DEVICES=0
export CUDA_VISIBLE_DEVICES=0
export TORCH_COMPILE_DEBUG=1

rm -rf ./cache/*
mkdir -p ./cache logs

python train_baichuan2_7B.py \
  --model_path $MODEL_PATH \
  --data_path $DATA_PATH \
  --output_dir "./baichuan2-finetuned" \
  --num_epochs 3 \
  --max_steps 200 \
  --batch_size 1 \
  --learning_rate 2e-5 \
  --max_length 1024 \
  --use_lora \
  --use_bf16 \
  --gradient_checkpointing \
  > logs/train_baichuan.log 2>&1