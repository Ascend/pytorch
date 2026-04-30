import os
import sys
import logging
from pathlib import Path
import argparse
import torch
import time
from datasets import Dataset as HFDataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    default_data_collator,
    TrainerCallback,
)
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
)
sys.path.append(str(Path(__file__).parent.parent))
from utils.utils import TimingCallback, get_profile, detect_device_type

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_NAME = "Mamba2"
TARGET_MODULES = ["x_proj", "in_proj"]

def patch_remove_ops_from_generate_list(op_names=None):
    try:
        import torch
        from torch_npu._inductor.ascend_npu_ir.ascend_npu_ir import config as anir_config

        if not op_names:
            print("[patch] No op names provided, nothing to do.")
            return

        for name in op_names:
            parts = name.split(".")
            op = torch.ops
            for p in parts:
                op = getattr(op, p)

            if op in anir_config.GENERATE_LIST:
                anir_config.GENERATE_LIST.remove(op)
                print(f"[patch] Successfully removed {name} from GENERATE_LIST.")
            else:
                print(f"[patch] {name} not found in GENERATE_LIST (maybe already removed).")

    except Exception as e:
        print(f"[patch] Failed to modify GENERATE_LIST: {e}")

def setup_environment():
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

def parse_args():
    parser = argparse.ArgumentParser(description="Mamba Codestral LoRA finetune")

    parser.add_argument("--model_path", type=str, default="/home/zhangyican/workspace/q4_data/Mamba-Codestral-7B-v0.1", help="Base model path")
    parser.add_argument("--data_file", type=str, default="./c4_demo.jsonl", help="Training data file path")
    parser.add_argument("--output_dir", type=str, default="./mamba_codestral_lora_no_quant", help="Output directory")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size per device")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--max_seq_length", type=int, default=256, help="Maximum sequence length")
    parser.add_argument("--lora_r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.1, help="LoRA dropout")
    parser.add_argument("--max_steps", type=int, default=100, help="Maximum training steps")
    parser.add_argument("--device", type=str, default="npu:0", help="NPU device, e.g., npu:0 / npu:6")
    parser.add_argument("--enable_compile", action="store_true", help="Whether to enable torch.compile")
    parser.add_argument("--enable_profiler", action="store_true")
    parser.add_argument("--profiler_save_path", type=str, default="./profile")
    parser.add_argument("--profiler_start_step", type=int, default=5,
                       help="Start step for profiling")
    parser.add_argument("--profiler_end_step", type=int, default=6,
                    help="End step for profiling")
    parser.add_argument("--npu-backend", type=str, default="mlir")

    return parser.parse_args()

def main():
    args = parse_args()
    device = detect_device_type()
    os.environ['TORCHINDUCTOR_NPU_BACKEND']=args.npu_backend
    patch_remove_ops_from_generate_list(["aten.permute"])
    seed = 2
    torch.manual_seed(seed)

    setup_environment()
    print("Environment setup complete")

    print(f"Loading tokenizer from: {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        use_fast=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print(f"Loading base model from: {args.model_path} (no quantization)")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,  # Use bfloat16 precision
        trust_remote_code=True,
        use_cache=False,  # Disable cache to save memory
    )
    
    base_model.gradient_checkpointing_enable()
    
    print(f"Configuring LoRA, target modules: {TARGET_MODULES}")
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=TARGET_MODULES,
        bias="none",
    )
    
    print("Attaching LoRA adapters to model...")
    peft_model = get_peft_model(base_model, peft_config)
    peft_model.print_trainable_parameters()  # Print number of trainable parameters
    
    peft_model.to(device)
    
    print("Preparing dataset...")
    from datasets import load_dataset
    dataset = load_dataset(
        "json",
        data_files=args.data_file,
        split="train"
    )
    
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=args.max_seq_length,
            return_tensors=None
        )
    
    train_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])
    train_dataset = train_dataset.map(lambda x: {"labels": x["input_ids"]}, batched=True)

    if args.enable_compile:
        peft_model = torch.compile(
            peft_model,
            dynamic=False
        )
    
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=1,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        bf16=True,
        fp16=False,
        save_strategy="steps",
        save_total_limit=2,
        remove_unused_columns=False,
        report_to="none",
        dataloader_pin_memory=False,
        max_grad_norm=0.3,
        logging_dir=f"{args.output_dir}/logs",
    )
    
    mode = "compile" if args.enable_compile else "eager"
    prof=None

    if args.enable_profiler:
        profiling_save_path = args.profiler_save_path + '/' + MODEL_NAME + '/' + mode
        prof = get_profile(args.profiler_start_step, args.profiler_end_step, profiling_save_path)

    timing_callback = TimingCallback(prof, mode) 

    print("Initializing Trainer...")
    trainer = Trainer(
        model=peft_model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        data_collator=default_data_collator,
        callbacks=[timing_callback],
    )
    
    print("Starting LoRA training...")
    trainer.train()

    if args.enable_compile:
        headers, values = torch._dynamo.utils.compile_times("csv")
        for header, value in zip(headers, values):
            if header == "async_compile.wait":
                numbers = [float(num.strip()) for num in value.split(',') if num.strip()]
                op_compile_time = sum(numbers)
        print(f"op_compile_time:{op_compile_time * 1e3} ms", )
    
    print(f"Saving LoRA weights to {args.output_dir}")
    peft_model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    print("Training complete!")

if __name__ == "__main__":
    main()