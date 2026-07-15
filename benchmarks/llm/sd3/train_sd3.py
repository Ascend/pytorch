import os
import sys
import torch
import torch.nn.functional as F
from pathlib import Path
from diffusers import StableDiffusion3Pipeline
from peft import LoraConfig, get_peft_model_state_dict
from datasets import load_dataset
from torchvision import transforms
import random
import numpy as np
import argparse
import logging
import torch_npu
from transformers import Trainer, TrainingArguments
sys.path.append(str(Path(__file__).parent.parent))
from utils.utils import (
    TimingCallback,
    get_profile,
    detect_device_type
)

model_name = 'sd3-medium'

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.npu.manual_seed_all(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def custom_collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    texts = [example["text"] for example in examples]
    return {"pixel_values": pixel_values, "text": texts}


class SD3Trainer(Trainer):
    def __init__(self, pipe, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pipe = pipe

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None, **kwargs):
        device = model.device
        model_dtype = model.dtype

        pixel_values = inputs["pixel_values"].to(device, dtype=model_dtype)
        prompts = inputs["text"]

        with torch.no_grad():
            prompt_embeds, _, pooled_prompt_embeds, _ = self.pipe.encode_prompt(
                prompt=prompts,
                prompt_2=prompts,
                prompt_3=prompts,
                device=device,
                num_images_per_prompt=1,
                do_classifier_free_guidance=False
            )

        with torch.no_grad():
            latents = self.pipe.vae.encode(pixel_values).latent_dist.mode()
            latents = (latents - self.pipe.vae.config.shift_factor) * self.pipe.vae.config.scaling_factor

        noise = torch.randn_like(latents)
        bsz = latents.shape[0]

        sigmas = torch.rand((bsz,), device=device, dtype=model_dtype)
        sigmas_reshaped = sigmas.view(-1, 1, 1, 1)

        noisy_latents = (1.0 - sigmas_reshaped) * latents + sigmas_reshaped * noise
        target = noise - latents

        timesteps = sigmas * 1000

        model_pred = model(
            hidden_states=noisy_latents,
            timestep=timesteps,
            encoder_hidden_states=prompt_embeds,
            pooled_projections=pooled_prompt_embeds,
            return_dict=False
        )[0]

        loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

        return (loss, None) if return_outputs else loss


class SD3LoRAFineTuner:
    def __init__(self, args):
        self.args = args
        self.device = detect_device_type() if hasattr(sys.modules[__name__], 'detect_device_type') else "cuda"
        self.dtype = torch.bfloat16 if self.args.use_bf16 else torch.float16

        set_seed(self.args.seed)
        self.setup_model()

    def setup_model(self):
        logger.info(f"Loading SD3 model from {self.args.model_path}")

        self.pipe = StableDiffusion3Pipeline.from_pretrained(
            self.args.model_path,
            torch_dtype=self.dtype,
            local_files_only=True
        ).to(self.device)

        self.transformer = self.pipe.transformer
        self.vae = self.pipe.vae
        self.text_encoder = self.pipe.text_encoder
        self.text_encoder_2 = self.pipe.text_encoder_2
        self.text_encoder_3 = self.pipe.text_encoder_3

        self.vae.requires_grad_(False)
        self.text_encoder.requires_grad_(False)
        self.text_encoder_2.requires_grad_(False)
        self.text_encoder_3.requires_grad_(False)
        self.transformer.requires_grad_(False)

        logger.info("Injecting LoRA adapters...")
        transformer_lora_config = LoraConfig(
            r=self.args.lora_rank,
            lora_alpha=self.args.lora_rank,
            init_lora_weights="gaussian",
            target_modules=["to_k", "to_q", "to_v", "to_out.0"],
        )

        self.transformer.add_adapter(transformer_lora_config)

        params_to_optimize = list(filter(lambda p: p.requires_grad, self.transformer.parameters()))
        logger.info(f"Trainable params: {sum(p.numel() for p in params_to_optimize)}")

        if self.args.enable_compile:
            logger.info("Compiling transformer model with torch.compile...")
            self.transformer = torch.compile(self.transformer)

    def load_and_preprocess_data(self):
        logger.info(f"Loading dataset from {self.args.data_path}")
        dataset = load_dataset("parquet", data_files=self.args.data_path)["train"]

        transform = transforms.Compose([
            transforms.Resize((self.args.resolution, self.args.resolution)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

        def preprocess_transforms(examples):
            pixel_values = []
            for image in examples["image"]:
                if image.mode != "RGB":
                    image = image.convert("RGB")
                pixel_values.append(transform(image))

            examples["pixel_values"] = pixel_values
            return examples

        dataset.set_transform(preprocess_transforms)
        return dataset

    def train(self):
        logger.info("Starting SD3 LoRA training...")
        dataset = self.load_and_preprocess_data()

        callbacks = []
        prof=None
        mod = 'compile' if self.args.enable_compile else 'eager'
        if self.args.enable_profiler:
            if not os.path.exists(self.args.profiler_save_path):
                os.makedirs(self.args.profiler_save_path)

            profiling_save_path = os.path.join(self.args.profiler_save_path, 'sd3', mod)

            prof = get_profile(
                profiler_start_step=self.args.profiler_start_step,
                profiler_end_step=self.args.profiler_end_step,
                profiling_save_path=profiling_save_path
            )

        callbacks.append(TimingCallback(profiler=prof, mod=mod))

        training_args = TrainingArguments(
            output_dir=self.args.output_dir,
            max_steps=self.args.max_steps,
            num_train_epochs=self.args.num_epochs,
            per_device_train_batch_size=self.args.batch_size,
            learning_rate=self.args.learning_rate,
            logging_steps=1,
            save_strategy="no",
            remove_unused_columns=False,
            report_to="none",
            dataloader_num_workers=self.args.dataloader_num_workers,
            bf16=self.args.use_bf16,
            fp16=not self.args.use_bf16
        )

        trainer = SD3Trainer(
            pipe=self.pipe,
            model=self.transformer,
            args=training_args,
            train_dataset=dataset,
            data_collator=custom_collate_fn,
            callbacks=callbacks
        )

        train_result = trainer.train()

        logger.info(f"Saving LoRA weights to {self.args.output_dir}")
        transformer_lora_state_dict = get_peft_model_state_dict(self.transformer)
        self.pipe.save_lora_weights(
            save_directory=self.args.output_dir,
            transformer_lora_layers=transformer_lora_state_dict,
            safe_serialization=True
        )

        return train_result.metrics


def main():
    parser = argparse.ArgumentParser(description="Train Stable Diffusion 3 LoRA")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the pretrained SD3 model")
    parser.add_argument("--data_path", type=str, required=True, help="Path to training data (parquet file)")
    parser.add_argument("--output_dir", type=str, default="./sd3_lora_weights", help="Output directory for LoRA weights")
    parser.add_argument("--num_epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--max_steps", type=int, default=10, help="Total training steps (-1 means use num_epochs)")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size per device")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--lora_rank", type=int, default=8, help="LoRA rank")
    parser.add_argument("--resolution", type=int, default=1024, help="Image resolution for training")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--dataloader_num_workers", type=int, default=0, help="Dataloader workers")
    parser.add_argument("--use_bf16", action="store_true", help="Use BF16 precision (recommended for SD3)")
    parser.add_argument("--enable_compile", action="store_true", help="Enable torch.compile")
    parser.add_argument("--enable_profiler", action="store_true", help="Enable PyTorch profiler")
    parser.add_argument("--profiler_start_step", type=int, default=10, help="Step to start profiler")
    parser.add_argument("--profiler_end_step", type=int, default=14, help="Step to end profiler")
    parser.add_argument("--profiler_save_path", type=str, default="./profile", help="Path to save profiler logs")
    parser.add_argument("--report_to_tensorboard", action="store_true", help="Report to TensorBoard")
    parser.add_argument("--npu-backend", type=str, default="mlir")
    parser.add_argument("--mfusion", action="store_true", help="Enable MFusion for graph fusion optimization")

    args = parser.parse_args()
    device_type = detect_device_type()
    os.environ['TORCHINDUCTOR_NPU_BACKEND'] = args.npu_backend
    if args.npu_backend == "akg":
        os.environ['TORCHINDUCTOR_NPU_BACKEND'] = 'mlir'
        os.environ['TORCHINDUCTOR_USE_AKG'] = '1'
    if args.mfusion:
        os.environ['TORCHINDUCTOR_ENABLE_MFUSION'] = '1'
    finer_tuner = SD3LoRAFineTuner(args)
    metrics = finer_tuner.train()
    if args.enable_compile:
        headers, values = torch._dynamo.utils.compile_times("csv")
        for header, value in zip(headers, values):
            if header == "PyCodeCache.load_by_key_path":
                numbers = [float(num.strip()) for num in value.split(',') if num.strip()]
                op_compile_time = sum(numbers)
        print(f"op_compile_time:{op_compile_time * 1e3} ms", )

    print("\n" + "="*50)
    print("SD3 LoRA Training completed successfully!")
    print(f"LoRA weights saved to: {args.output_dir}")
    print(f"Final training loss: {metrics.get('train_loss', 'N/A')}")
    print("="*50)


if __name__ == "__main__":
    main()