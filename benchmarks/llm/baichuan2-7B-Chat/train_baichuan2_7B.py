import os
import json
import time
import sys
from pathlib import Path
import argparse
import logging
from typing import Dict, List, Optional

import torch
from datasets import Dataset, load_dataset

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    TrainerCallback,
    set_seed,
)
from peft import LoraConfig, get_peft_model, TaskType
sys.path.append(str(Path(__file__).parent.parent))
from utils.utils import TimingCallback, get_profile, detect_device_type


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_NAME = "Baichuan2-7B-Chat"

class BaichuanTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        return super().compute_loss(model, inputs, return_outputs=return_outputs)


class Baichuan2Trainer:
    def __init__(self, args):
        self.args = args
        self.model = None
        self.tokenizer = None
        self.setup_training()

    def setup_training(self):
        logger.info(f"Loading model/tokenizer from {self.args.model_path}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.args.model_path,
            trust_remote_code=True,
            padding_side="right",
            model_max_length=self.args.max_length,
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            self.args.model_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16 if self.args.use_bf16 else torch.float16,
            use_cache=not self.args.gradient_checkpointing,
        )

        if self.args.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
            self.model.config.use_cache = False

    def apply_lora(self):
        if not self.args.use_lora:
            return

        logger.info("Applying LoRA...")

        target_modules = self.args.lora_target_modules.split(",") if self.args.lora_target_modules else ["W_pack"]

        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=self.args.lora_r,
            lora_alpha=self.args.lora_alpha,
            lora_dropout=self.args.lora_dropout,
            target_modules=target_modules,
            bias="none",
        )
        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()

    def format_conversation(self, conversations: List[Dict]) -> str:
        text = ""
        for turn in conversations:
            role = turn.get("from", "")
            content = turn.get("value", "")
            if role == "human":
                text += f"<reserved_106>{content}"
            elif role == "assistant":
                text += f"<reserved_107>{content}"
            elif role == "system":
                text += f"<reserved_106>{content}"
        text += self.tokenizer.eos_token
        return text

    def load_and_preprocess_data(self) -> Dataset:
        logger.info(f"Loading dataset from {self.args.data_path}")

        if self.args.data_path.endswith(".json") or self.args.data_path.endswith(".jsonl"):
            with open(self.args.data_path, "r", encoding="utf-8") as f:
                if self.args.data_path.endswith(".jsonl"):
                    raw = [json.loads(line) for line in f]
                else:
                    raw = json.load(f)

            formatted = []
            for item in raw:
                if "conversations" in item:
                    text = self.format_conversation(item["conversations"])
                else:
                    text = item.get("text", "")
                formatted.append({"text": text})

            dataset = Dataset.from_list(formatted)
        else:
            dataset = load_dataset(self.args.data_path, split=self.args.split)

        def preprocess_function(examples):
            tokenized = self.tokenizer(
                examples["text"],
                truncation=True,
                padding="max_length",
                max_length=self.args.max_length,
                return_attention_mask=True,
            )
            tokenized["labels"] = tokenized["input_ids"].copy()
            return tokenized

        num_proc = self.args.num_proc if self.args.num_proc > 0 else None
        processed = dataset.map(
            preprocess_function,
            batched=True,
            remove_columns=dataset.column_names,
            num_proc=num_proc,
            load_from_cache_file=not self.args.overwrite_cache,
        )
        logger.info(f"Dataset size: {len(processed)}")
        return processed

    def create_trainer(self, train_dataset, eval_dataset=None):
        has_eval = eval_dataset is not None
        eval_strategy = "steps" if has_eval else "no"

        training_args = TrainingArguments(
            output_dir=self.args.output_dir,
            overwrite_output_dir=True,
            num_train_epochs=self.args.num_epochs,
            max_steps=self.args.max_steps,
            per_device_train_batch_size=self.args.batch_size,
            per_device_eval_batch_size=self.args.batch_size,
            gradient_accumulation_steps=self.args.gradient_accumulation_steps,
            learning_rate=self.args.learning_rate,
            weight_decay=self.args.weight_decay,
            warmup_ratio=self.args.warmup_ratio,
            lr_scheduler_type=self.args.lr_scheduler_type,
            logging_dir=os.path.join(self.args.output_dir, "logs"),
            logging_steps=self.args.logging_steps,
            save_steps=self.args.save_steps,
            save_total_limit=self.args.save_total_limit,
            eval_strategy=eval_strategy,
            eval_steps=self.args.eval_steps if has_eval else None,
            bf16=self.args.use_bf16,
            fp16=self.args.use_fp16,
            gradient_checkpointing=self.args.gradient_checkpointing,
            remove_unused_columns=False,
            report_to="none",
            dataloader_pin_memory=False,
            optim="adamw_torch",
        )

        data_collator = DataCollatorForSeq2Seq(
            tokenizer=self.tokenizer,
            model=self.model,
            padding="max_length",
            max_length=self.args.max_length,
            label_pad_token_id=-100,
        )

        mode = "compile" if self.args.enable_compile else "eager"
        prof=None

        if self.args.enable_profiler:
            profiling_save_path = self.args.profiler_save_path + '/' + MODEL_NAME + '/' + mode
            prof = get_profile(self.args.profiler_start_step, self.args.profiler_end_step, profiling_save_path)

        timing_callback = TimingCallback(prof, mode) 

        trainer = BaichuanTrainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            callbacks=[timing_callback],
        )
        return trainer

    def train(self):
        logger.info("Starting training...")

        if self.args.use_lora:
            self.apply_lora()

        if self.args.enable_compile:
            logger.warning("Enable torch.compile (experimental on NPU).")
            try:
                self.model = torch.compile(self.model, dynamic=False)
            except Exception as e:
                logger.error(f"torch.compile failed: {e}. Continue without compile.")

        dataset = self.load_and_preprocess_data()

        trainer = self.create_trainer(dataset)
        train_result = trainer.train()

        trainer.save_state()
        trainer.save_model(self.args.output_dir)
        self.tokenizer.save_pretrained(self.args.output_dir)

        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)

        logger.info(f"Training completed! Model saved to {self.args.output_dir}")
        return metrics


def build_argparser():
    parser = argparse.ArgumentParser(description="Train Baichuan2-7B-Chat (LoRA) on NPU")
    
    parser.add_argument("--npu-backend", type=str, default="mlir")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./baichuan2-finetuned")
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine")
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use_bf16", action="store_true")
    parser.add_argument("--use_fp16", action="store_true")
    parser.add_argument("--use_lora", action="store_true")
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.1)
    parser.add_argument("--lora_target_modules", type=str, default="W_pack")
    parser.add_argument("--validation_split", type=float, default=0.0)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--num_proc", type=int, default=0)
    parser.add_argument("--overwrite_cache", action="store_true")
    parser.add_argument("--logging_steps", type=int, default=1)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--save_total_limit", type=int, default=3)
    parser.add_argument("--eval_steps", type=int, default=500)
    parser.add_argument("--enable_compile", action="store_true")
    parser.add_argument("--enable_profiler", action="store_true")
    parser.add_argument("--profiler_save_path", type=str, default="./profile")
    parser.add_argument("--profiler_start_step", type=int, default=5,
                       help="Output directory for trained model")
    parser.add_argument("--profiler_end_step", type=int, default=7,
                    help="Output directory for trained model")

    return parser


def main():
    args = build_argparser().parse_args()
    detect_device_type()
    os.environ['TORCHINDUCTOR_NPU_BACKEND']=args.npu_backend
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "logs"), exist_ok=True)

    set_seed(args.seed)
    torch.manual_seed(args.seed)

    logger.info(f"npu train {MODEL_NAME}")
    trainer = Baichuan2Trainer(args)
    metrics = trainer.train()

    if args.enable_compile:
        headers, values = torch._dynamo.utils.compile_times("csv")
        for header, value in zip(headers, values):
            if header == "async_compile.wait":
                numbers = [float(num.strip()) for num in value.split(',') if num.strip()]
                op_compile_time = sum(numbers)
        print(f"op_compile_time:{op_compile_time * 1e3} ms", )

if __name__ == "__main__":
    main()