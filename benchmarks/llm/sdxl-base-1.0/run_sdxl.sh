#!/bin/bash
export ASCEND_RT_VISIBLE_DEVICES=0
export CUDA_VISIBLE_DEVICES=0
export TORCHINDUCTOR_CACHE_DIR="./cache"
export TORCH_COMPILE_DEBUG=1

rm -rf ./cache/*
mkdir -p ./cache logs

python train_sdxl_base.py \
    --model_path $MODEL_PATH \
    --data_path $DATA_PATH \
    --max_steps 200 \
    --batch_size 1 \
    --learning_rate 1e-5 \
    --lora_rank 8 \
    --resolution 1024 \
    --use_bf16 \
    --dataloader_num_workers 8 \
    > logs/train_sdxl.log 2>&1