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
import logging
import os
import subprocess
import sys
import time
import warnings

import torch
from common import BenchmarkRunner, main, PathManager
from torch._dynamo.testing import collect_results, reduce_to_scalar_loss
from torch._dynamo.utils import clone_inputs


MAX_DOWNLOAD_ATTEMPTS = 5


def download_retry_decorator(download_fn):
    @functools.wraps(download_fn)
    def wrapper(self, *args, **kwargs):
        tries = 0
        total_allowed_tries = MAX_DOWNLOAD_ATTEMPTS
        while tries < total_allowed_tries:
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


def pip_install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])


try:
    from timm import __version__ as timmversion
    from timm.data import resolve_data_config
    from timm.models import create_model
except ModuleNotFoundError:
    print("Please install pytorch-image-models.")
    raise

TIMM_MODELS = dict()
filename = os.path.join(os.path.dirname(__file__), "timm_models_list.txt")

abspath = os.path.abspath(filename)
PathManager.check_directory_path_readable(abspath)

with open(filename) as fh:
    lines = fh.readlines()
    lines = [line.rstrip() for line in lines]
    for line in lines:
        model_name, batch_size = line.split(" ")
        TIMM_MODELS[model_name] = int(batch_size)

REQUIRE_HIGHER_TOLERANCE = {
    "vit_base_patch16_siglip_256",
    "deit_base_distilled_patch16_224"
}

SCALED_COMPUTE_LOSS = {
    "ese_vovnet19b_dw",
    "fbnetc_100",
    "mnasnet_100",
    "mobilevit_s",
    "sebotnet33ts_256",
}

FORCE_AMP_FOR_FP16_BF16_MODELS = {
    "convit_base",
    "xcit_large_24_p8_224",
}


NPU_REQUIRE_LEARNING_RATE = {
    "adv_inception_v3",
    "beit_base_patch16_224",
    "repvgg_a2",
    "vit_base_patch16_siglip_256",
}


NPU_REQUIRE_LOWER_LEARNING_RATE = {
    "convnextv2_nano.fcmae_ft_in22k_in1k",
    "mobilenetv2_100",
}


class TimmRunner(BenchmarkRunner):
    def __init__(self):
        super().__init__()
        self.suite_name = "timm_models"

    @property
    def force_amp_for_fp16_bf16_models(self):
        return FORCE_AMP_FOR_FP16_BF16_MODELS

    @download_retry_decorator
    def _download_model(self, model_name):
        model = create_model(
            model_name,
            in_chans=3,
            scriptable=False,
            num_classes=None,
            drop_rate=0.0,
            drop_path_rate=None,
            drop_block_rate=None,
            pretrained=True,
        )
        return model

    def load_model(
        self,
        device,
        model_name,
        batch_size=None,
    ):
        is_training = self.args.training
        model = self._download_model(model_name)

        if model is None:
            raise RuntimeError(f"Failed to load model '{model_name}'")
        model.to(device=device)

        self.num_classes = model.num_classes

        data_config = resolve_data_config(
            vars(self._args) if timmversion >= "0.8.0" else self._args,
            model=model,
            use_test_size=not is_training,
        )
        input_size = data_config["input_size"]
        recorded_batch_size = TIMM_MODELS[model_name]

        batch_size = batch_size or recorded_batch_size

        torch.manual_seed(1337)
        input_tensor = torch.randint(
            256, size=(batch_size,) + input_size, device=device
        ).to(dtype=torch.float32)
        mean = torch.mean(input_tensor)
        std_dev = torch.std(input_tensor)
        example_inputs = (input_tensor - mean) / std_dev

        example_inputs = [
            example_inputs,
        ]

        self.loss = torch.nn.CrossEntropyLoss().to(device)

        if model_name in SCALED_COMPUTE_LOSS:
            self.compute_loss = self.scaled_compute_loss

        if is_training:
            model.train()
        else:
            model.eval()

        self.validate_model(model, example_inputs)

        return device, model_name, model, example_inputs, batch_size

    def iter_model_names(self, args):
        model_names = sorted(TIMM_MODELS.keys())
        for index, model_name in enumerate(model_names):
            yield model_name

    def pick_grad(self, name, is_training):
        if is_training:
            return torch.enable_grad()
        else:
            return torch.no_grad()

    def get_learning_rate(self, is_training, current_device, name):
        learning_rate = 1e-2
        if is_training and current_device == "cuda":
            learning_rate = 1e-2
            if name in NPU_REQUIRE_LEARNING_RATE:
                learning_rate = 1e-3
            elif name in NPU_REQUIRE_LOWER_LEARNING_RATE:
                learning_rate = 1e-4
        if is_training and current_device == "npu":
            learning_rate = 1e-2
            if name in NPU_REQUIRE_LEARNING_RATE:
                learning_rate = 1e-3
            elif name in NPU_REQUIRE_LOWER_LEARNING_RATE:
                learning_rate = 1e-4
        return learning_rate

    def get_tolerance_and_cosine_flag(self, is_training, current_device, name):
        cosine = self.args.cosine
        tolerance = 1e-3
        if is_training:
            if name in REQUIRE_HIGHER_TOLERANCE:
                tolerance = 4 * 1e-2
            else:
                tolerance = 2 * 1e-2
        return tolerance, cosine

    def _gen_target(self, batch_size, device):
        return torch.empty((batch_size,) + (), device=device, dtype=torch.long).random_(
            self.num_classes
        )

    def compute_loss(self, pred):
        # High loss values make gradient checking harder, as small changes in
        # accumulation order upsets accuracy checks.
        return reduce_to_scalar_loss(pred)

    def scaled_compute_loss(self, pred):
        # Loss values need zoom out further.
        return reduce_to_scalar_loss(pred) / 1000.0

    def forward_pass(self, mod, inputs, collect_outputs=True):
        with self.autocast():
            return mod(*inputs)

    def forward_and_backward_pass(self, mod, inputs, collect_outputs=True):
        cloned_inputs = clone_inputs(inputs)
        self.optimizer_zero_grad(mod)
        with self.autocast():
            pred = mod(*cloned_inputs)
            if isinstance(pred, tuple):
                pred = pred[0]
            loss = self.compute_loss(pred)
        self.grad_scaler.scale(loss).backward()
        self.optimizer_step()
        if collect_outputs:
            return collect_results(mod, pred, loss, cloned_inputs)
        return loss.item()


def timm_main():
    logging.basicConfig(level=logging.WARNING)
    warnings.filterwarnings("ignore")
    main(TimmRunner())


if __name__ == "__main__":
    timm_main()
