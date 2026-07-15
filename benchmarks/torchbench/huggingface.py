#!/usr/bin/env python3
# BSD 3-Clause License

# Copyright (c) 2026, pytorch
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.

# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import functools
import importlib
import logging
import os
import subprocess
import sys
import time
import warnings

import torch
from common import BenchmarkRunner, main, reset_rng_state
from torch._dynamo.testing import collect_results
from torch._dynamo.utils import clone_inputs


MAX_DOWNLOAD_ATTEMPTS = 5


def download_retry_decorator(download_fn):
    @functools.wraps(download_fn)
    def wrapper(self, *args, **kwargs):
        tries = 0
        total_allowed_tries = MAX_DOWNLOAD_ATTEMPTS
        while tries <= total_allowed_tries:
            try:
                model = download_fn(self, *args, **kwargs)
                return model
            except Exception as err:
                tries += 1
                if tries <= total_allowed_tries:
                    wait = tries * 30
                    print(
                        f"Failed to load model: {err}. Trying again ({tries}/{total_allowed_tries}) after {wait}s"
                    )
                    time.sleep(wait)
                else:
                    raise RuntimeError(
                        f"Failed to load model '{args}' with following error(s): {str(err)}."
                    ) from err

    return wrapper


# We are primarily interested in tf32 datatype
torch.backends.cuda.matmul.allow_tf32 = True
try:
    torch.npu.config.allow_internal_format = False
except Exception:
    pass

log = logging.getLogger(__name__)


def pip_install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])


imports = [
    "AlbertForPreTraining",
    "AutoConfig",
    "AutoModelForCausalLM",
    "AutoModelForMaskedLM",
    "AutoModelForSeq2SeqLM",
    "BigBirdConfig",
    "BlenderbotForConditionalGeneration",
    "BlenderbotModel",
    "BlenderbotSmallForConditionalGeneration",
    "BlenderbotSmallModel",
    "CLIPModel",
    "CLIPVisionModel",
    "ElectraForPreTraining",
    "GPT2ForSequenceClassification",
    "GPTJForSequenceClassification",
    "GPTNeoForSequenceClassification",
    "HubertForSequenceClassification",
    "LxmertForPreTraining",
    "LxmertForQuestionAnswering",
    "MarianForCausalLM",
    "MarianModel",
    "MarianMTModel",
    "PegasusForConditionalGeneration",
    "PegasusModel",
    "ReformerConfig",
    "ViTForImageClassification",
    "ViTForMaskedImageModeling",
    "ViTModel",
]


try:
    mod = importlib.import_module("transformers")
except ModuleNotFoundError:
    print("Please install transformers.")
    raise

for cls in imports:
    if not hasattr(mod, cls):
        raise ModuleNotFoundError(f"Missing transformers symbol: {cls}")


# These models contain the models present in huggingface_models_list. It is a
# combination of models supported by HF Fx parser and some manually supplied
# models. For these models, we already know the largest batch size that can fit
# on A100 GPUs - 40 GB.
BATCH_SIZE_KNOWN_MODELS = dict()


# Get the list of models and their batch sizes
MODELS_FILENAME = os.path.join(os.path.dirname(__file__), "huggingface_models_list.txt")
if not os.path.exists(MODELS_FILENAME):
    raise AssertionError
with open(MODELS_FILENAME) as fh:
    lines = fh.readlines()
    lines = [line.rstrip() for line in lines]
    for line in lines:
        model_name, batch_size = line.split(",")
        batch_size = int(batch_size)
        BATCH_SIZE_KNOWN_MODELS[model_name] = batch_size
if not BATCH_SIZE_KNOWN_MODELS:
    raise AssertionError


SKIP = {
    # Difficult to setup accuracy test because .eval() not supported
    "Reformer",
    # Fails deepcopy
    "BlenderbotForConditionalGeneration",
    "GPTNeoForCausalLM",
    "GPTNeoForSequenceClassification",
    # Fails with even batch size = 1
    "GPTJForCausalLM",
    "GPTJForQuestionAnswering",
}

# TODO - Fails even after fake tensors
BATCH_SIZE_DIVISORS = {
    "AlbertForMaskedLM": 2,
    "AlbertForQuestionAnswering": 2,
    "AllenaiLongformerBase": 2,
    "BartForCausalLM": 2,
    "BartForConditionalGeneration": 2,
    "BertForMaskedLM": 2,
    "BertForQuestionAnswering": 2,
    "BlenderbotForCausalLM": 8,
    # "BlenderbotForConditionalGeneration" : 16,
    "BlenderbotSmallForCausalLM": 4,
    "BlenderbotSmallForConditionalGeneration": 2,
    "CamemBert": 2,
    "DebertaForMaskedLM": 4,
    "DebertaForQuestionAnswering": 2,
    "DebertaV2ForMaskedLM": 4,
    "DebertaV2ForQuestionAnswering": 8,
    "DistilBertForMaskedLM": 2,
    "DistilBertForQuestionAnswering": 2,
    "DistillGPT2": 2,
    "ElectraForCausalLM": 2,
    "ElectraForQuestionAnswering": 2,
    "GPT2ForSequenceClassification": 2,
    # "GPTJForCausalLM" : 2,
    # "GPTJForQuestionAnswering" : 2,
    # "GPTNeoForCausalLM" : 32,
    # "GPTNeoForSequenceClassification" : 2,
    "GoogleFnet": 2,
    "LayoutLMForMaskedLM": 2,
    "LayoutLMForSequenceClassification": 2,
    "M2M100ForConditionalGeneration": 4,
    "MBartForCausalLM": 2,
    "MBartForConditionalGeneration": 2,
    "MT5ForConditionalGeneration": 2,
    "MegatronBertForCausalLM": 4,
    "MegatronBertForQuestionAnswering": 2,
    "MobileBertForMaskedLM": 2,
    "MobileBertForQuestionAnswering": 2,
    "OPTForCausalLM": 2,
    "PLBartForCausalLM": 2,
    "PLBartForConditionalGeneration": 2,
    "PegasusForCausalLM": 4,
    "PegasusForConditionalGeneration": 2,
    "RobertaForCausalLM": 2,
    "RobertaForQuestionAnswering": 2,
    "Speech2Text2ForCausalLM": 4,
    "T5ForConditionalGeneration": 2,
    "T5Small": 2,
    "TrOCRForCausalLM": 2,
    "XGLMForCausalLM": 4,
    "XLNetLMHeadModel": 2,
    "YituTechConvBert": 2,
}

SKIP_ACCURACY_CHECK_MODELS = {
    # Models too large to have eager, dynamo and fp64_numbers simultaneosuly
    # even for 40 GB machine.
    "DebertaV2ForMaskedLM",
    "BlenderbotForCausalLM",
}

DECODER_INPUT_MODEL_CLASS_NAMES = {
    "BlenderbotModel",
    "BlenderbotSmallModel",
    "BlenderbotForConditionalGeneration",
    "BlenderbotSmallForConditionalGeneration",
    "PegasusModel",
    "PegasusForConditionalGeneration",
    "MarianModel",
    "MarianMTModel",
}

PRETRAINING_BINARY_LABEL_MODEL_CLASS_NAMES = {
    "ElectraForPreTraining",
    "LxmertForPreTraining",
}

PRETRAINING_SENTENCE_ORDER_MODEL_CLASS_NAMES = {
    "AlbertForPreTraining",
}


REQUIRE_HIGHER_TOLERANCE_TRAINING = {
    "MT5ForConditionalGeneration",
    # AlbertForQuestionAnswering fails in CI GCP A100 but error does not seem
    # harmful.
    "AlbertForQuestionAnswering",
}
REQUIRE_HIGHER_TOLERANCE_INFERENCE = {
    "RobertaForQuestionAnswering",
}


SKIP_FOR_CPU = {
    "OPTForCausalLM",  # OOMs
}

ONLY_EVAL_MODE = {
    "M2M100ForConditionalGeneration",  # Fails with dynamo for train mode
}

FP32_ONLY_MODELS = {"GoogleFnet"}

NPU_FP32_ONLY_MODELS = {
    "MT5ForConditionalGeneration"  # Accuracy Fails in npu amp mode for train mode
}

NPU_ACC_FORCE_BATCH_SIZE = {
    "MT5ForConditionalGeneration"  # There is a bug in batch_size=1, will be fixed soon
}


def get_module_cls_by_model_name(model_cls_name):
    _module_by_model_name = {
        "Speech2Text2Decoder": "transformers.models.speech_to_text_2.modeling_speech_to_text_2",
        "TrOCRDecoder": "transformers.models.trocr.modeling_trocr",
    }
    module_name = _module_by_model_name.get(model_cls_name, "transformers")
    module = importlib.import_module(module_name)
    return getattr(module, model_cls_name)


def get_sequence_length(model_cls, model_name):
    if model_name.startswith(("Blenderbot",)):
        seq_length = 128
    elif model_name.startswith(("GPT2", "Bart", "T5", "PLBart", "MBart")):
        seq_length = 1024
    elif model_name in ("AllenaiLongformerBase", "BigBird"):
        seq_length = 1024
    elif model_name.startswith("OPT"):
        seq_length = 2048
    elif "Reformer" in model_name:
        seq_length = 4096
    elif model_name.startswith(
        (
            "Albert",
            "Deberta",
            "Layout",
            "Electra",
            "XLNet",
            "MegatronBert",
            "Bert",
            "Roberta",
        )
    ) or model_name in ("DistillGPT2", "GoogleFnet", "YituTechConvBert", "CamemBert"):
        seq_length = 512
    elif model_name in ("TrOCRForCausalLM"):
        seq_length = 256
    elif model_name.startswith("MobileBert"):
        seq_length = 128
    elif model_name.startswith("Wav2Vec2"):
        # If too short, will fail with something like
        # ValueError: `mask_length` has to be smaller than `sequence_length`,
        # but got `mask_length`: 10 and `sequence_length`: 9`
        seq_length = 10000  # NB: a more realistic size is 155136
    else:
        log.info(
            "Sequence Length not defined for %s. Choosing 128 arbitrarily",
            model_name,
        )
        seq_length = 128
    return seq_length


def generate_inputs_for_model(
    model_cls, model, model_name, bs, device, include_loss_args=False
):
    # TODO - Check if following values are representative
    num_choices = 3
    num_visual_features = 42
    seq_length = get_sequence_length(model_cls, model_name)
    vocab_size = model.config.vocab_size

    if model_name.startswith("Wav2Vec2"):
        # TODO: If we add more input_values style models, try to work this
        # into the overall control flow
        target_length = 100
        return {
            "input_values": torch.randn((bs, seq_length), device=device),
            # Added because that's what the example training script has
            "attention_mask": rand_int_tensor(device, 0, 2, (bs, seq_length)),
            "labels": rand_int_tensor(device, 0, vocab_size, (bs, target_length)),
        }

    if model_name.endswith("MultipleChoice"):
        input = rand_int_tensor(device, 0, vocab_size, (bs, num_choices, seq_length))
    elif model_name.startswith("Roberta"):
        input = rand_int_tensor(device, 0, 1, (bs, seq_length))
    else:
        input = rand_int_tensor(device, 0, vocab_size, (bs, seq_length))

    if "Bart" in model_name:
        input[:, -1] = model.config.eos_token_id

    input_dict = {"input_ids": input}

    if (
        model_name.startswith(("T5", "M2M100", "MT5"))
        or model_cls.__name__ in DECODER_INPUT_MODEL_CLASS_NAMES
    ):
        input_dict["decoder_input_ids"] = input

    if model_name.startswith("Lxmert"):
        visual_feat_dim, visual_pos_dim = (
            model.config.visual_feat_dim,
            model.config.visual_pos_dim,
        )
        input_dict["visual_feats"] = torch.randn(
            bs, num_visual_features, visual_feat_dim
        )
        input_dict["visual_pos"] = torch.randn(bs, num_visual_features, visual_pos_dim)

    if include_loss_args:
        if model_name.endswith("PreTraining"):
            if model_cls.__name__ in PRETRAINING_BINARY_LABEL_MODEL_CLASS_NAMES:
                input_dict["labels"] = rand_int_tensor(device, 0, 1, (bs, seq_length))
            else:
                label_name = (
                    "sentence_order_label"
                    if model_cls.__name__
                    in PRETRAINING_SENTENCE_ORDER_MODEL_CLASS_NAMES
                    else "next_sentence_label"
                )
                input_dict["labels"] = (
                    rand_int_tensor(device, 0, vocab_size, (bs, seq_length)),
                )
                input_dict[label_name] = rand_int_tensor(device, 0, 1, (bs,))
        elif model_name.endswith("QuestionAnswering"):
            input_dict["start_positions"] = rand_int_tensor(
                device, 0, seq_length, (bs,)
            )
            input_dict["end_positions"] = rand_int_tensor(device, 0, seq_length, (bs,))
        elif model_name.endswith(
            ("MaskedLM", "HeadModel", "CausalLM", "DoubleHeadsModel")
        ):
            input_dict["labels"] = rand_int_tensor(
                device, 0, vocab_size, (bs, seq_length)
            )
        elif model_name.endswith("TokenClassification"):
            input_dict["labels"] = rand_int_tensor(
                device, 0, model.config.num_labels - 1, (bs, seq_length)
            )
        elif model_name.endswith("MultipleChoice"):
            input_dict["labels"] = rand_int_tensor(device, 0, num_choices, (bs,))
        elif model_name.endswith("SequenceClassification"):
            input_dict["labels"] = rand_int_tensor(
                device, 0, model.config.num_labels - 1, (bs,)
            )
        elif model_name.endswith("NextSentencePrediction"):
            input_dict["labels"] = rand_int_tensor(device, 0, 1, (bs,))
        elif model_name.endswith("ForConditionalGeneration"):
            input_dict["labels"] = rand_int_tensor(
                device, 0, vocab_size - 1, (bs, seq_length)
            )
        elif model_name in EXTRA_MODELS:
            input_dict["labels"] = rand_int_tensor(
                device, 0, vocab_size, (bs, seq_length)
            )
        else:
            raise NotImplementedError(
                f"Class {model_name} unsupported for training test "
            )

    return input_dict


def rand_int_tensor(device, low, high, shape):
    return torch.randint(
        low,
        high,
        shape,
        device=device,
        dtype=torch.int64,
        requires_grad=False,
    )


EXTRA_MODELS = {}

NPU_REQUIRE_LEARNING_RATE = {
    "T5ForConditionalGeneration",
}

PAD_TOKEN_MODEL_CLASS_NAMES = {
    "GPT2ForSequenceClassification",
    "GPTNeoForSequenceClassification",
    "GPTJForSequenceClassification",
}


class HuggingfaceRunner(BenchmarkRunner):
    def __init__(self):
        super().__init__()
        self.suite_name = "huggingface"

    @property
    def skip_models_for_cpu(self):
        return SKIP_FOR_CPU

    @property
    def fp32_only_models(self):
        return FP32_ONLY_MODELS.union(NPU_FP32_ONLY_MODELS)

    def _get_model_cls_and_config(self, model_name):
        if model_name not in EXTRA_MODELS:
            model_cls = get_module_cls_by_model_name(model_name)
            config_cls = model_cls.config_class
            config = config_cls()

            # NB: some models need a pad token defined to handle BS > 1
            if (
                model_cls.__name__ in PAD_TOKEN_MODEL_CLASS_NAMES
                or model_cls.__name__.startswith("Roberta")
                or model_cls.__name__.startswith("Marian")
            ):
                config.pad_token_id = 0

        else:
            config, model_cls = EXTRA_MODELS[model_name]

        return model_cls, config

    @download_retry_decorator
    def _download_model(self, model_name):
        model_cls, config = self._get_model_cls_and_config(model_name)
        model = model_cls(config)
        return model

    def load_model(
        self,
        device,
        model_name,
        batch_size=None,
    ):
        is_training = self.args.training
        dtype = torch.float32
        reset_rng_state()
        model_cls, config = self._get_model_cls_and_config(model_name)
        model = self._download_model(model_name)
        model = model.to(device, dtype=dtype)
        if model_name in BATCH_SIZE_KNOWN_MODELS:
            batch_size_default = BATCH_SIZE_KNOWN_MODELS[model_name]
        elif batch_size is None:
            batch_size_default = 16
            log.info(
                "Batch size not specified for %s. Setting batch_size=16", model_name
            )

        if batch_size is None:
            batch_size = batch_size_default
            if model_name in BATCH_SIZE_DIVISORS:
                batch_size = max(int(batch_size / BATCH_SIZE_DIVISORS[model_name]), 1)
                log.info(
                    "Running smaller batch size=%s for %s, orig batch_size=%s",
                    batch_size,
                    model_name,
                    batch_size_default,
                )

        # we need to set batch_size in some specific cases
        if model_name in NPU_ACC_FORCE_BATCH_SIZE:
            batch_size = 4

        example_inputs = generate_inputs_for_model(
            model_cls, model, model_name, batch_size, device, include_loss_args=True
        )

        # So we can check for correct gradients without eliminating the dropout computation
        for attr in dir(config):
            if "drop" in attr and isinstance(getattr(config, attr), float):
                setattr(config, attr, 1e-30)

        if is_training and not (self.args.accuracy and model_name in ONLY_EVAL_MODE):
            model.train()
        else:
            model.eval()

        self.validate_model(model, example_inputs)
        return device, model_name, model, example_inputs, batch_size

    def iter_model_names(self, args):
        model_names = list(BATCH_SIZE_KNOWN_MODELS.keys()) + list(EXTRA_MODELS.keys())
        model_names = set(model_names)
        model_names = sorted(model_names)

        for index, model_name in enumerate(model_names):
            yield model_name

    @property
    def skip_accuracy_checks_large_models_dashboard(self):
        if self.args.dashboard or self.args.accuracy:
            return SKIP_ACCURACY_CHECK_MODELS
        return set()

    def pick_grad(self, name, is_training):
        if is_training:
            return torch.enable_grad()
        else:
            return torch.no_grad()

    def get_learning_rate(self, is_training, current_device, name):
        learning_rate = 1e-2
        if is_training and current_device == "cuda":
            learning_rate = 1e-2
        if is_training and current_device == "npu":
            learning_rate = 1e-2
        if is_training and name in NPU_REQUIRE_LEARNING_RATE:
            learning_rate = 1e-4
        return learning_rate

    def get_tolerance_and_cosine_flag(self, is_training, current_device, name):
        cosine = self.args.cosine
        if is_training:
            if name in REQUIRE_HIGHER_TOLERANCE_TRAINING:
                return 2e-2, cosine
            else:
                return 1e-2, cosine
        else:
            if name in REQUIRE_HIGHER_TOLERANCE_INFERENCE:
                return 4e-3, cosine
        return 1e-3, cosine

    def compute_loss(self, pred):
        return pred[0]

    def forward_pass(self, mod, inputs, collect_outputs=True):
        with self.autocast():
            return mod(**inputs)

    def forward_and_backward_pass(self, mod, inputs, collect_outputs=True):
        cloned_inputs = clone_inputs(inputs)
        self.optimizer_zero_grad(mod)
        with self.autocast():
            pred = mod(**cloned_inputs)
            loss = self.compute_loss(pred)
        self.grad_scaler.scale(loss).backward()
        self.optimizer_step()
        if collect_outputs:
            return collect_results(mod, pred, loss, cloned_inputs)
        return loss.item()


def huggingface_main():
    warnings.filterwarnings("ignore")
    main(HuggingfaceRunner())


if __name__ == "__main__":
    huggingface_main()
