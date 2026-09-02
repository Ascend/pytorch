#!/bin/bash
export ASCEND_RT_VISIBLE_DEVICES=0
export CUDA_VISIBLE_DEVICES=0
export TORCHINDUCTOR_CACHE_DIR="./cache"
export TORCH_COMPILE_DEBUG=1
export TORCH_NPU_USE_COMPATIBLE_IMPL=1

rm -rf ./cache/*
mkdir -p ./cache logs

python train_qwen2vl.py \
  --model_name_or_path $MODEL_PATH \
  --train_data_path $DATA_PATH \
  --image_folder $IMAGE_PATH \
  --output_dir "./qwen2vl_output" \
  --num_train_epochs 3 \
  --max_steps 200 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 1 \
  --learning_rate 2e-5 \
  --max_length 2048 \
  --use_lora false \
  --freeze_vision_tower true \
  --bf16 true \
  --data_repeat 1 \
  --min_pixels 200704 \
  --max_pixels 1003520 \
  > logs/train_qwen2vl.log 2>&1