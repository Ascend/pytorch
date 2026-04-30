import torch
import os
import sys
from pathlib import Path
import torch.nn as nn
from datasets import Dataset, load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType
)
import json
from typing import Dict, List, Optional
import argparse
import logging

sys.path.append(str(Path(__file__).parent.parent))   
from utils.utils import (
    TimingCallback, 
    get_profile, 
    detect_device_type
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

model_name = 'GPT-OSS-20B'

class GPT_OSS_20BTrainer:
    def __init__(self, args):
        self.args = args
        self.setup_training()
        
    def setup_training(self):
        logger.info(f"Loading model from {self.args.model_path}")
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.args.model_path,
            torch_dtype=torch.bfloat16 if self.args.use_bf16 else torch.float16,
            trust_remote_code=True
        )

        logger.info(f"Moving model to {self.args.device_type}...")
        self.model = self.model.to(self.args.device_type)
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.args.model_path,
            trust_remote_code=True
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model.config.pad_token_id = self.tokenizer.eos_token_id
            
        if self.args.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
            self.model.config.use_cache = False

            
    def apply_lora(self):
        if not self.args.use_lora:
            return
            
        logger.info("Applying LoRA configuration...")
        
        if self.args.use_4bit:
            self.model = prepare_model_for_kbit_training(self.model)
        
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=self.args.lora_r,
            lora_alpha=self.args.lora_alpha,
            lora_dropout=self.args.lora_dropout,
            target_modules=self.get_lora_target_modules(),
            bias="none",
        )
        
        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()
        
        if self.args.enable_compile:
            self.model = torch.compile(self.model, dynamic=False)
        
    def get_lora_target_modules(self):
        target_modules = [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ]
        
        model_modules = set([name for name, _ in self.model.named_modules()])
        available_modules = [m for m in target_modules if any(m in name for name in model_modules)]
        
        if not available_modules:
            available_modules = ["qkv_proj", "dense", "fc1", "fc2"]
            
        logger.info(f"Using LoRA target modules: {available_modules}")
        return available_modules
        
    def load_and_preprocess_data(self) -> Dataset:
        logger.info(f"Loading dataset from {self.args.data_path}")
        
        if self.args.data_path.endswith('.json') or self.args.data_path.endswith('.jsonl'):
            with open(self.args.data_path, 'r', encoding='utf-8') as f:
                if self.args.data_path.endswith('.jsonl'):
                    data = [json.loads(line) for line in f]
                else:
                    data = json.load(f)
                    
            formatted_data = []
            for item in data:
                if "conversations" in item:
                    conversations = item["conversations"]
                    text = self.format_conversation(conversations)
                elif "messages" in item:
                    messages = item["messages"]
                    text = self.tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=False
                    )
                else:
                    text = item.get("text", "")
                    
                formatted_data.append({"text": text})
                
            dataset = Dataset.from_list(formatted_data)
            
        else:
            try:
                dataset = load_dataset(
                    self.args.data_path,
                    split=self.args.split
                )
            except:
                dataset = load_dataset(
                    "json",
                    data_files=self.args.data_path,
                    split="train"
                )
        
        def preprocess_function(examples, tokenizer=self.tokenizer, max_length=self.args.max_length):
            tokenized = tokenizer(
                examples["text"],
                truncation=True,
                padding="max_length",
                max_length=max_length,
                return_tensors=None,
                return_attention_mask=True,
            )
            
            import copy
            tokenized["labels"] = copy.deepcopy(tokenized["input_ids"])
            
            return tokenized
        
        num_proc = self.args.num_proc if self.args.num_proc > 0 else None
        processed_dataset = dataset.map(
            preprocess_function,
            batched=True,
            remove_columns=dataset.column_names,
            num_proc=num_proc, 
            load_from_cache_file=not self.args.overwrite_cache
        )
        
        logger.info(f"Dataset size: {len(processed_dataset)}")
        return processed_dataset
        
    def format_conversation(self, conversations: List[Dict]) -> str:
        formatted_text = ""
        
        for turn in conversations:
            role = turn.get("from", "").lower()
            content = turn.get("value", "")
            
            if role == "human" or role == "user":
                formatted_text += f"<|im_start|>user\n{content}<|im_end|>\n"
            elif role == "gpt" or role == "assistant":
                formatted_text += f"<|im_start|>assistant\n{content}<|im_end|>\n"
            elif role == "system":
                formatted_text += f"<|im_start|>system\n{content}<|im_end|>\n"
                
        return formatted_text
        
    def create_trainer(self, train_dataset, eval_dataset=None):
        has_eval = eval_dataset is not None
        eval_strategy = "steps" if has_eval else "no"
        save_strategy = "steps" if has_eval else "steps"  
        
        training_args = TrainingArguments(
            output_dir=self.args.output_dir,
            overwrite_output_dir=True,
            num_train_epochs=self.args.num_epochs,
            max_steps=self.args.max_steps,
            per_device_train_batch_size=self.args.batch_size,
            per_device_eval_batch_size=self.args.batch_size,
            gradient_accumulation_steps=self.args.gradient_accumulation_steps,
            weight_decay=self.args.weight_decay,
            logging_dir=f"{self.args.output_dir}/logs",
            logging_steps=self.args.logging_steps,
            save_steps=self.args.save_steps,
            save_total_limit=self.args.save_total_limit,
            eval_strategy=eval_strategy,  
            eval_steps=self.args.eval_steps if has_eval else None,
            save_strategy=save_strategy, 
            load_best_model_at_end=has_eval, 
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            learning_rate=self.args.learning_rate,
            lr_scheduler_type="cosine",
            fp16=self.args.use_fp16,
            bf16=self.args.use_bf16,
            gradient_checkpointing=self.args.gradient_checkpointing,
            report_to="tensorboard",
            ddp_find_unused_parameters=False if torch.cuda.device_count() > 1 else None,
            remove_unused_columns=False,
        )
        
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
        )

        mod='compile' if self.args.enable_compile else 'eager'
        prof=None

        if self.args.enable_profiler:
            profiling_save_path = self.args.profiler_save_path + '/' + model_name + '/' + mod
            prof = get_profile(self.args.profiler_start_step, self.args.profiler_end_step, profiling_save_path)

        timing_callback = TimingCallback(prof, mod) 
        
        trainer = Trainer(
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
        
        dataset = self.load_and_preprocess_data()
        
        if self.args.validation_split > 0:
            split_dataset = dataset.train_test_split(
                test_size=self.args.validation_split,
                seed=self.args.seed
            )
            train_dataset = split_dataset["train"]
            eval_dataset = split_dataset["test"]
        else:
            train_dataset = dataset
            eval_dataset = None
            
        trainer = self.create_trainer(train_dataset, eval_dataset)
        
        train_result = trainer.train()
        
        if self.args.enable_compile:
            headers, values = torch._dynamo.utils.compile_times("csv")
            for header, value in zip(headers, values):
                if header == "async_compile.wait":
                    numbers = [float(num.strip()) for num in value.split(',') if num.strip()]
                    op_compile_time = sum(numbers)
            print(f"op_compile_time:{op_compile_time * 1e3} ms", )

        trainer.save_model()
        self.tokenizer.save_pretrained(self.args.output_dir)
        
        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()
        
        logger.info(f"Training completed! Model saved to {self.args.output_dir}")
        
        return metrics

def main():
    parser = argparse.ArgumentParser(description="Train GPT-OSS-20B model")
    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to the pretrained model")
    parser.add_argument("--data_path", type=str, required=True,
                       help="Path to training data (json/jsonl file or dataset name)")
    parser.add_argument("--output_dir", type=str, default="./GPT-OSS-finetuned",
                       help="Output directory for trained model")
    parser.add_argument("--num_epochs", type=int, default=3,
                       help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4,
                       help="Batch size per device")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2,
                       help="Gradient accumulation steps")
    parser.add_argument("--learning_rate", type=float, default=2e-4,
                       help="Learning rate")
    parser.add_argument("--warmup_steps", type=int, default=5,
                       help="Warmup steps")
    parser.add_argument("--max_steps", type=int, default=1, 
                       help="Total training steps")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                       help="Weight decay")
    parser.add_argument("--max_length", type=int, default=2048,
                       help="Maximum sequence length")
    parser.add_argument("--use_lora", action="store_true",
                       help="Use LoRA for parameter-efficient fine-tuning")
    parser.add_argument("--lora_r", type=int, default=8,
                       help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32,
                       help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.1,
                       help="LoRA dropout")
    parser.add_argument("--use_4bit", action="store_true",
                       help="Use 4-bit quantization")
    parser.add_argument("--use_fp16", action="store_true",
                       help="Use FP16 precision")
    parser.add_argument("--use_bf16", action="store_true",
                       help="Use BF16 precision")
    parser.add_argument("--gradient_checkpointing", action="store_true",
                       help="Enable gradient checkpointing")
    parser.add_argument("--validation_split", type=float, default=0.1,
                       help="Validation split ratio")
    parser.add_argument("--split", type=str, default="train",
                       help="Dataset split to use")
    parser.add_argument("--num_proc", type=int, default=0,
                       help="Number of processes for data preprocessing (0 = single process, avoid CUDA conflict)")
    parser.add_argument("--pad_to_max_length", action="store_true",
                       help="Pad sequences to max_length")
    parser.add_argument("--overwrite_cache", action="store_true",
                       help="Overwrite cached features")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    parser.add_argument("--logging_steps", type=int, default=1,
                       help="Log every X updates steps")
    parser.add_argument("--save_steps", type=int, default=500,
                       help="Save checkpoint every X updates steps")
    parser.add_argument("--eval_steps", type=int, default=500,
                       help="Evaluate every X updates steps")
    parser.add_argument("--save_total_limit", type=int, default=3,
                       help="Limit the total amount of checkpoints")
    parser.add_argument("--report_to_tensorboard", action="store_true",
                       help="Report metrics to TensorBoard")
    parser.add_argument("--enable_compile", action="store_true",
                       help="Enable torch.compile and Inductor backend")
    parser.add_argument("--enable_profiler", action="store_true",
                       help="Enable profiler for performance analysis")
    parser.add_argument("--profiler_start_step", type=int, default=5,
                       help="Output directory for trained model")
    parser.add_argument("--profiler_end_step", type=int, default=8,
                    help="Output directory for trained model")
    parser.add_argument("--profiler_save_path", type=str, default="./profile",
                    help="Output directory for trained model")
    parser.add_argument("--npu-backend", type=str, default="mlir")
    
    args = parser.parse_args()
    torch.manual_seed(args.seed)

    args.device_type = detect_device_type()
    os.environ['TORCHINDUCTOR_NPU_BACKEND']=args.npu_backend
    print(f"{args.device_type}  train  {model_name}")

    trainer = GPT_OSS_20BTrainer(args)
    metrics = trainer.train()
    
    print("\n" + "="*50)
    print("Training completed successfully!")
    print(f"Model saved to: {args.output_dir}")
    print(f"Final training loss: {metrics.get('train_loss', 'N/A')}")
    print("="*50)

if __name__ == "__main__":
    main()