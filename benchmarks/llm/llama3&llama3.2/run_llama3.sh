export TORCHINDUCTOR_CACHE_DIR='./cache'
export ASCEND_RT_VISIBLE_DEVICES=3
export TORCH_COMPILE_DEBUG=1
export TORCH_NPU_USE_COMPATIBLE_IMPL=1

rm -rf ./cache/*
mkdir -p ./cache logs

python train_llama3.py \
    --model_path "your model path" \
    --model_name "your model name" \
    --data_path "dataset path" \
    --output_dir "./llama3-finetuned" \
    --num_epochs 3 \
    --max_steps 200 \
    --batch_size 1 \
    --learning_rate 2e-5 \
    --max_length 512 \
    --use_lora \
    --use_bf16 \
    --pad_to_max_length > logs/train_llama3.log 2>&1
