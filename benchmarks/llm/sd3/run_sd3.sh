#!/bin/bash
export ASCEND_RT_VISIBLE_DEVICES=0
export CUDA_VISIBLE_DEVICES=0
export TORCHINDUCTOR_CACHE_DIR="./cache"
export TORCH_COMPILE_DEBUG=1
export TORCH_NPU_USE_COMPATIBLE_IMPL=1

rm -rf ./cache/*
mkdir -p ./cache logs

python train_sd3.py \
    --model_path $MODEL_PATH \
    --data_path $DATA_PATH \
    --output_dir "./sd3-finetuned" \
    --num_epochs 1 \
    --max_steps 200 \
    --batch_size 1 \
    --use_bf16  \
    --dataloader_num_workers 8 \
    > logs/train_sd3.log 2>&1