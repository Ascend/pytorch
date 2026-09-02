export TORCHINDUCTOR_CACHE_DIR='./cache'
export ASCEND_RT_VISIBLE_DEVICES=0
export TORCH_COMPILE_DEBUG=1
export TORCH_NPU_USE_COMPATIBLE_IMPL=1

rm -rf ./cache/*
mkdir -p ./cache logs

python train_qwen3_4B.py \
    --model_path "your model path" \
    --data_path "dataset path" \
    --output_dir "./qwen3-finetuned" \
    --num_epochs 3 \
    --max_steps 200 \
    --batch_size 1 \
    --learning_rate 2e-5 \
    --max_length 512 \
    --use_lora \
    --use_bf16 \
    --pad_to_max_length > logs/train_qwen.log 2>&1