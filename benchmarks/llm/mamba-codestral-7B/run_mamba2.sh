#!/bin/bash
export ASCEND_RT_VISIBLE_DEVICES=0
export CUDA_VISIBLE_DEVICES=0
export TORCHINDUCTOR_CACHE_DIR="./cache"
export TORCH_COMPILE_DEBUG=1

rm -rf ./cache/*
mkdir -p ./cache logs

python train_mamba2_7B.py \
  --model_path $MODEL_PATH \
  --data_file $DATA_PATH \
  --output_dir "./mamba_codestral_lora" \
  --epochs 3 \
  --batch_size 1 \
  --gradient_accumulation_steps 1 \
  --learning_rate 1e-4 \
  --max_seq_length 256 \
  --lora_r 16 \
  --lora_alpha 32 \
  --lora_dropout 0.1 \
  --max_steps 200 \
  > logs/train_mamba2.log 2>&1