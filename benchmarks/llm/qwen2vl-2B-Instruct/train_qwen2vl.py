import os
import sys
from pathlib import Path
import json
import time
import argparse
import logging
import torch
import torch_npu
from typing import Dict, List, Any, Optional
from PIL import Image
from torch.utils.data import Dataset
from torch.profiler import (
    profile,
    ProfilerActivity,
    schedule,
    tensorboard_trace_handler,
)

from transformers import (
    Qwen2VLForConditionalGeneration,
    AutoProcessor,
    TrainingArguments,
    Trainer,
    TrainerCallback,
)
from peft import LoraConfig, get_peft_model
sys.path.append(str(Path(__file__).parent.parent))
from utils.utils import TimingCallback, get_profile, detect_device_type


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_NAME = "Qwen2VL-2B-instruct"

def str2bool(v: str) -> bool:
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "1"):
        return True
    if v.lower() in ("no", "false", "f", "0"):
        return False
    raise argparse.ArgumentTypeError(f"Unable to parse '{v}' as a boolean value")

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

def parse_args():
    p = argparse.ArgumentParser(description="Qwen2-VL 2B Single GPU Training")
    p.add_argument("--model_name_or_path", type=str,
                    default="/data/zyc/Qwen2-VL-2B-Instruct")
    p.add_argument("--use_lora", type=str2bool, default=False)
    p.add_argument("--lora_r", type=int, default=64)
    p.add_argument("--lora_alpha", type=int, default=16)
    p.add_argument("--lora_dropout", type=float, default=0.1)
    p.add_argument("--freeze_vision_tower", type=str2bool, default=True)
    p.add_argument("--enable_compile", action="store_true",help="Enable torch.compile")
    p.add_argument("--train_data_path", type=str, default="./train_data.json")
    p.add_argument("--image_folder", type=str, default="./images")
    p.add_argument("--max_length", type=int, default=2048)
    p.add_argument("--data_repeat", type=int, default=1,
                    help="Number of data repetitions for small dataset augmentation")
    p.add_argument("--min_pixels", type=int, default=256 * 28 * 28,
                    help="Qwen2-VL Processor min_pixels (controls minimum image resolution)")
    p.add_argument("--max_pixels", type=int, default=1280 * 28 * 28,
                    help="Qwen2-VL Processor max_pixels (controls maximum image resolution)")
    p.add_argument("--output_dir", type=str, default="./qwen2vl_output")
    p.add_argument("--num_train_epochs", type=int, default=3)
    p.add_argument("--per_device_train_batch_size", type=int, default=1)
    p.add_argument("--gradient_accumulation_steps", type=int, default=1)
    p.add_argument("--learning_rate", type=float, default=2e-5)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--warmup_ratio", type=float, default=0.03)
    p.add_argument("--bf16", type=str2bool, default=False)
    p.add_argument("--logging_steps", type=int, default=1)
    p.add_argument("--max_steps", type=int, default=-1)
    p.add_argument("--save_steps", type=int, default=500)
    p.add_argument("--save_total_limit", type=int, default=3)
    p.add_argument("--dataloader_num_workers", type=int, default=0)
    p.add_argument("--enable_profiler", action="store_true", help="Enable profiler")
    p.add_argument("--profiler_save_path", type=str, default="./profile")
    p.add_argument("--profiler_start_step", type=int, default=5,
                       help="Start step for profiling")
    p.add_argument("--profiler_end_step", type=int, default=6,
                    help="End step for profiling")
    p.add_argument("--npu-backend", type=str, default="mlir")
    p.add_argument("--mfusion", action="store_true", help="Enable MFusion for graph fusion optimization")

    return p.parse_args()

class Qwen2VLDataset(Dataset):
    def __init__(
        self,
        data_path: str,
        processor,
        image_folder: str,
        max_length: int = 2048,
        repeat: int = 1,
    ):
        super().__init__()
        self.processor = processor
        self.image_folder = image_folder
        self.max_length = max_length
        self.repeat = repeat

        with open(data_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)

    def __len__(self) -> int:
        return len(self.data) * self.repeat

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        real_idx = idx % len(self.data)
        item = self.data[real_idx]

        image_path = os.path.join(self.image_folder, item["image"])
        image = Image.open(image_path).convert("RGB")

        conversations = item["conversations"]
        messages: List[Dict] = []
        for i, conv in enumerate(conversations):
            if conv["role"] == "user":
                if i == 0:
                    messages.append({
                        "role": "user",
                        "content": [
                            {"type": "image", "image": image},
                            {"type": "text", "text": conv["content"]},
                        ],
                    })
                else:
                    messages.append({
                        "role": "user",
                        "content": [
                            {"type": "text", "text": conv["content"]},
                        ],
                    })
            else:
                messages.append({
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": conv["content"]},
                    ],
                })

        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )

        inputs = self.processor(
            text=[text],
            images=[image],
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt",
        )

        result: Dict[str, Any] = {}
        for k, v in inputs.items():
            if k == "image_grid_thw":
                result[k] = v
            elif k == "pixel_values":
                result[k] = v.squeeze(0) if v.dim() > 4 else v
            else:
                result[k] = v.squeeze(0)

        result["labels"] = result["input_ids"].clone()
        pad_token_id = self.processor.tokenizer.pad_token_id
        result["labels"][result["labels"] == pad_token_id] = -100

        return result

class Qwen2VLDataCollator:

    def __init__(self, processor):
        self.processor = processor

    def __call__(
        self, features: List[Dict[str, Any]]
    ) -> Dict[str, torch.Tensor]:
        batch: Dict[str, torch.Tensor] = {}

        for key in ("input_ids", "attention_mask", "labels"):
            if key in features[0]:
                batch[key] = torch.stack([f[key] for f in features])

        if "pixel_values" in features[0]:
            pv_list = []
            for f in features:
                pv = f["pixel_values"]
                if pv.dim() == 3:
                    pv = pv.unsqueeze(0)
                pv_list.append(pv)
            batch["pixel_values"] = torch.cat(pv_list, dim=0)

        if "image_grid_thw" in features[0]:
            thw_list = []
            for f in features:
                thw = f["image_grid_thw"]
                if thw.dim() == 1:
                    thw = thw.unsqueeze(0)
                thw_list.append(thw)
            batch["image_grid_thw"] = torch.cat(thw_list, dim=0)

        return batch

def main():
    torch.use_deterministic_algorithms(True)
    args = parse_args()
    detect_device_type()
    os.environ['TORCHINDUCTOR_NPU_BACKEND']=args.npu_backend
    if args.npu_backend == "akg":
        os.environ['TORCHINDUCTOR_NPU_BACKEND'] = 'mlir'
        os.environ['TORCHINDUCTOR_USE_AKG'] = '1'
    if args.mfusion:
        os.environ['TORCHINDUCTOR_ENABLE_MFUSION']='1'
    patch_remove_ops_from_generate_list(["aten.clone", "aten.permute"])

    print("\nLoading Processor...")
    t0 = time.time()
    processor = AutoProcessor.from_pretrained(
        args.model_name_or_path,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
        trust_remote_code=True,
    )
    print(f"Processor loading time: {time.time() - t0:.2f}s")
    print(f"  min_pixels = {args.min_pixels}, max_pixels = {args.max_pixels}")

    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    print("\nLoading Qwen2-VL model...")
    t0 = time.time()
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    print(f"Model loading time: {time.time() - t0:.2f}s")

    model.gradient_checkpointing_enable()

    if args.freeze_vision_tower:
        print("Freezing vision encoder (parameters containing 'visual')...")
        frozen_count = 0
        for name, param in model.named_parameters():
            if "visual" in name:
                param.requires_grad = False
                frozen_count += 1
        print(f"  Frozen {frozen_count} vision parameters")

    if args.enable_compile:
        torch.autograd.set_detect_anomaly(True)
        print("Enabling torch.compile (dynamic=False) ...")
        model = torch.compile(model, dynamic=False)

    if args.use_lora:
        print("\nConfiguring LoRA...")
        t0 = time.time()
        lora_target_modules = [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ]
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=lora_target_modules,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
        print(f"LoRA configuration time: {time.time() - t0:.2f}s")

    print("\nPreparing dataset...")
    t0 = time.time()

    if not os.path.exists(args.train_data_path) or os.path.getsize(args.train_data_path) == 0:
        print("Training data not found or empty, creating dummy data...")
        create_dummy_data(args.train_data_path, args.image_folder)

    train_dataset = Qwen2VLDataset(
        data_path=args.train_data_path,
        processor=processor,
        image_folder=args.image_folder,
        max_length=args.max_length,
        repeat=args.data_repeat,
    )
    print(f"Dataset preparation time: {time.time() - t0:.2f}s")

    data_collator = Qwen2VLDataCollator(processor=processor)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type="cosine",
        logging_dir="./logs",
        max_steps=args.max_steps,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        bf16=args.bf16,
        gradient_checkpointing=True,
        dataloader_num_workers=args.dataloader_num_workers,
        remove_unused_columns=False,
        report_to="none",
        ddp_find_unused_parameters=False,
    )

    mode = "compile" if args.enable_compile else "eager"
    prof=None

    if args.enable_profiler:
        profiling_save_path = args.profiler_save_path + '/' + MODEL_NAME + '/' + mode
        prof = get_profile(args.profiler_start_step, args.profiler_end_step, profiling_save_path)

    timing_callback = TimingCallback(prof, mode)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
        callbacks=[timing_callback],
    )
    trainer.train()
    if args.enable_compile:
        headers, values = torch._dynamo.utils.compile_times("csv")
        for header, value in zip(headers, values):
            if header == "PyCodeCache.load_by_key_path":
                numbers = [float(num.strip()) for num in value.split(',') if num.strip()]
                op_compile_time = sum(numbers)
        print(f"op_compile_time:{op_compile_time * 1e3} ms", )
    final_output_dir = os.path.join(args.output_dir, "final")
    trainer.save_model(final_output_dir)
    processor.save_pretrained(final_output_dir)

if __name__ == "__main__":
    main()