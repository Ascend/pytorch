#!/usr/bin/env python3
# BSD 3-Clause License

# Copyright (c) 2019, pytorch
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

import gc
import importlib
import logging
import os
import re
import sys
import warnings
from os.path import abspath, exists
import torch
from common import BenchmarkRunner, main
from torch._dynamo.testing import collect_results, reduce_to_scalar_loss
from torch._dynamo.utils import clone_inputs
try:
    import torch_npu
except ImportError:
    pass
from benchmark.userbenchmark.dynamo.dynamobench.torchbench import (
    USE_SMALL_BATCH_SIZE, ONLY_TRAINING_MODE, REQUIRE_HIGHER_TOLERANCE, REQUIRE_EVEN_HIGHER_TOLERANCE,
    NONDETERMINISTIC, VERY_SLOW_BENCHMARKS, SLOW_BENCHMARKS, DONT_CHANGE_BATCH_SIZE,
    MAX_BATCH_SIZE_FOR_ACCURACY_CHECK, FORCE_AMP_FOR_FP16_BF16_MODELS, 
)

# We are primarily interested in tf32 datatype
torch.backends.cuda.matmul.allow_tf32 = True
torch.npu.config.allow_internal_format = False


def setup_torchbench_cwd():
    original_dir = abspath(os.getcwd())

    os.environ["KALDI_ROOT"] = "/tmp"  # avoids some spam
    search_dirs = [
        "./benchmark",
        "./torchbenchmark",
        "../torchbenchmark", 
        "../torchbench",
        "../benchmark",
        "../../torchbenchmark",
        "../../torchbench",
        "../../benchmark",
    ]
    torchbench_dir = None
    for path in search_dirs:
        if exists(path):
            torchbench_dir = abspath(path)
            break
    
    if torchbench_dir:
        os.chdir(torchbench_dir)
        sys.path.append(torchbench_dir)

    return original_dir


SKIP = {
    "detectron2_maskrcnn",
    "fambench_xlmr",
    "tacotron2",
    "hf_Bert_large",  # Error: RelaxedUnspecConstraint(L['input_ids'].size()[0]) - inferred constant (4)
    # takes too long, extreme slowdown (< .001)
    "maml",
}


CHECK_NUMPY_VERSION = {
    "soft_actor_critic"
}


REQUIRE_HIGHER_FP16_TOLERANCE = {
    "drq",
}


NPU_REQUIRE_HIGHER_TOLERANCE = {
    "dcgan",
    "mobilenet_v2",
    "shufflenet_v2_x1_0",
    "timm_vovnet",
    "phlippe_resnet",
}


NPU_REQUIRE_HIGHER_FP16_TOLERANCE = {
    "timm_vision_transformer",
    "functorch_dp_cifar10",
    "moco",
    "speech_transformer",
    "timm_vovnet",
}
REQUIRE_HIGHER_FP16_TOLERANCE.update(NPU_REQUIRE_HIGHER_FP16_TOLERANCE)


# models in canary_models that we should run anyway
CANARY_MODELS = {
    "torchrec_dlrm",
}


NPU_REQUIRE_LEARNING_RATE = {
    "alexnet",
    "dcgan",
    "nvidia_deeprecommender",
}


NPU_REQUIRE_LOWER_LEARNING_RATE = {
    "phlippe_densenet",
    "phlippe_resnet",
    "resnet50",
    "resnet152",
    "resnext50_32x4d",
    "densenet121",
    "vgg16",
}


NPU_REUQIRE_EVEN_LOWER_LEARNING_RATE = {
    "mobilenet_v2",
    "resnet18",
    "shufflenet_v2_x1_0",
    "timm_vovnet",
}


NPU_DVM_NO_ACLGRAPH = {

}


NPU_MLIR_NO_ACLGRAPH = {

}


class TorchBenchmarkRunner(BenchmarkRunner):
    def __init__(self):
        super().__init__()
        self.suite_name = "torchbench"
        self.optimizer = None

    @property
    def skip_models(self):
        return SKIP

    @property
    def slow_models(self):
        return SLOW_BENCHMARKS

    @property
    def very_slow_models(self):
        return VERY_SLOW_BENCHMARKS

    @property
    def non_deterministic_models(self):
        return NONDETERMINISTIC

    @property
    def force_amp_for_fp16_bf16_models(self):
        return FORCE_AMP_FOR_FP16_BF16_MODELS

    def load_model(
        self,
        device,
        model_name,
        batch_size=None,
        part=None,
    ):
        is_training = self.args.training
        dynamic_shapes = self.args.dynamic_shapes
        candidates = [
            f"torchbenchmark.models.{model_name}",
            f"torchbenchmark.canary_models.{model_name}",
            f"torchbenchmark.models.fb.{model_name}",
        ]
        for c in candidates:
            try:
                module = importlib.import_module(c)
                break
            except ModuleNotFoundError as e:
                if e.name != c:
                    raise
        else:
            raise ImportError(f"could not import any of {candidates}")
        benchmark_cls = getattr(module, "Model", None)
        if not hasattr(benchmark_cls, "name"):
            benchmark_cls.name = model_name

        cant_change_batch_size = (
            not getattr(benchmark_cls, "ALLOW_CUSTOMIZE_BSIZE", True)
            or model_name in DONT_CHANGE_BATCH_SIZE
        )
        if cant_change_batch_size:
            batch_size = None
        if batch_size is None and is_training and model_name in USE_SMALL_BATCH_SIZE:
            batch_size = USE_SMALL_BATCH_SIZE[model_name]

        # Control the memory footprint for few models
        if (
            (self.args.accuracy or self.args.precision_checker)
            and model_name in MAX_BATCH_SIZE_FOR_ACCURACY_CHECK
        ):
            batch_size = min(batch_size, MAX_BATCH_SIZE_FOR_ACCURACY_CHECK[model_name])

        # workaround "RuntimeError: not allowed to set torch.backends.cudnn flags"
        torch.backends.__allow_nonbracketed_mutation_flag = True
        extra_args = []
        if part:
            extra_args = ["--part", part]

        if model_name == "vision_maskrcnn" and is_training:
            # Output of vision_maskrcnn model is a list of bounding boxes,
            # sorted on the basis of their scores. This makes accuracy
            # comparison hard with torch.compile. torch.compile can cause minor
            # divergences in the output because of how fusion works for amp in
            # TorchInductor compared to eager.  Therefore, instead of looking at
            # all the bounding boxes, we compare only top 5.
            model_kwargs = {"box_detections_per_img": 5}
            benchmark = benchmark_cls(
                test="train",
                device=device,
                batch_size=batch_size,
                extra_args=extra_args,
                model_kwargs=model_kwargs,
            )
        elif is_training:
            benchmark = benchmark_cls(
                test="train",
                device=device,
                batch_size=batch_size,
                extra_args=extra_args,
            )
        else:
            benchmark = benchmark_cls(
                test="eval",
                device=device,
                batch_size=batch_size,
                extra_args=extra_args,
            )
        model, example_inputs = benchmark.get_module()

        # Models that must be in train mode while training
        if is_training and model_name in ONLY_TRAINING_MODE:
            model.train()
        else:
            model.eval()
        gc.collect()
        batch_size = benchmark.batch_size

        self.validate_model(model, example_inputs)
        return device, benchmark.name, model, example_inputs, batch_size

    def _get_models_from_file(self, file_path):
        """Read model names from a file, one per line."""
        if not os.path.exists(file_path):
            return set()

        with open(file_path, 'r') as f:
            models = {line.strip() for line in f if line.strip()}
        return models

    def _get_models_from_directory(self, directory):
        """Get model names from subdirectories in the given directory."""
        if not os.path.exists(directory):
            return set()

        models = set()
        for item in os.listdir(directory):
            item_path = os.path.join(directory, item)
            if os.path.isdir(item_path):
                models.add(item)
        return models
    
    def _filter_models_by_numpy_version(self, model_names, max_version="2.0"):
        import numpy as np
        from packaging import version

        filtered_models = set(model_names)
        numpy_version = np.__version__
        
        # 检查是否需要过滤
        models_to_check = model_names & CHECK_NUMPY_VERSION
        if not models_to_check:
            return filtered_models
        
        # 解析当前numpy版本
        current_version = version.parse(numpy_version)
        max_allowed_version = version.parse(max_version)
        
        # 如果当前版本大于等于2.0，需要过滤
        if current_version > max_allowed_version:
            models_to_remove = set()
            for model_name in models_to_check:
                if model_name in filtered_models:
                    models_to_remove.add(model_name)
                    logging.warning(
                        f"Model {model_name} has been skipped because it requires numpy version < 2.0, but current version is {numpy_version}."
                    )
            
            # 从集合中移除不兼容的模型
            filtered_models -= models_to_remove
    
        return filtered_models

    def iter_model_names(self, args):
        from torchbenchmark import _list_canary_model_paths, _list_model_paths

        # Get models from torchbenchmark repository
        torchbenchmark_models = _list_model_paths()
        torchbenchmark_models += [
            f
            for f in _list_canary_model_paths()
            if os.path.basename(f) in CANARY_MODELS
        ]

        torchbenchmark_model_names = {os.path.basename(m) for m in torchbenchmark_models}

        # Get models from torchbench_models_list.txt file
        script_dir = os.path.dirname(os.path.abspath(__file__))
        list_file = os.path.join(script_dir, "torchbench_models_list.txt")
        file_model_names = self._get_models_from_file(list_file)

        # Find intersection between torchbenchmark and file list
        torchbench_intersection = torchbenchmark_model_names & file_model_names

        # Log models that are in file but not in torchbenchmark
        models_not_in_torchbench = file_model_names - torchbenchmark_model_names
        if models_not_in_torchbench:
            logging.warning(
                "The following models from torchbench_models_list.txt are not available "
                "in the torchbenchmark repository and will be skipped: %s",
                ", ".join(sorted(models_not_in_torchbench))
            )

        # Get models from benchmarks/models directory
        models_dir = os.path.join(script_dir, "models")
        custom_model_names = self._get_models_from_directory(models_dir)

        # Combine: intersection of torchbenchmark + file, union with custom models
        final_model_names = torchbench_intersection | custom_model_names

        # Apply numpy version filter
        final_model_names = self._filter_models_by_numpy_version(final_model_names)

        # Apply skip_models filter
        final_model_names -= self.skip_models

        # Sort for consistent ordering
        sorted_models = sorted(final_model_names)

        for _, model_name in enumerate(sorted_models):
            yield model_name

    def pick_grad(self, name, is_training):
        if is_training or name in ("maml",):
            return torch.enable_grad()
        else:
            return torch.no_grad()

    def get_tolerance_and_cosine_flag(self, is_training, current_device, name):
        tolerance = 1e-4
        cosine = self.args.cosine
        # Increase the tolerance for torch allclose
        if self.args.float16 or self.args.amp:
            if name in REQUIRE_HIGHER_FP16_TOLERANCE:
                return 1e-2, cosine
            return 1e-3, cosine
        if is_training and current_device == "cuda":
            tolerance = 1e-3
            if name in REQUIRE_HIGHER_TOLERANCE:
                tolerance = 1e-3
            elif name in REQUIRE_EVEN_HIGHER_TOLERANCE:
                tolerance = 8 * 1e-2
        if is_training and current_device == "npu":
            tolerance = 1e-3
            if name in NPU_REQUIRE_HIGHER_TOLERANCE:
                tolerance = 1e-2
        return tolerance, cosine
    
    def get_learning_rate(self, is_training, current_device, name):
        learning_rate = 1e-2
        if is_training and current_device == "cuda":
            learning_rate = 1e-2
            if name in NPU_REQUIRE_LEARNING_RATE:
                learning_rate = 1e-3
            elif name in NPU_REQUIRE_LOWER_LEARNING_RATE:
                learning_rate = 1e-4
            elif name in NPU_REUQIRE_EVEN_LOWER_LEARNING_RATE:
                learning_rate = 1e-5

        if is_training and current_device == "npu":
            learning_rate = 1e-2
            if name in NPU_REQUIRE_LEARNING_RATE:
                learning_rate = 1e-3
            elif name in NPU_REQUIRE_LOWER_LEARNING_RATE:
                learning_rate = 1e-4
            elif name in NPU_REUQIRE_EVEN_LOWER_LEARNING_RATE:
                learning_rate = 1e-5
        return learning_rate

    def compute_loss(self, pred):
        return reduce_to_scalar_loss(pred)

    def forward_pass(self, mod, inputs, collect_outputs=True):
        with self.autocast():
            return mod(*inputs)

    def forward_and_backward_pass(self, mod, inputs, collect_outputs=True):
        cloned_inputs = clone_inputs(inputs)
        self.optimizer_zero_grad(mod)
        with self.autocast():
            pred = mod(*cloned_inputs)
            loss = self.compute_loss(pred)
        self.grad_scaler.scale(loss).backward()
        self.optimizer_step()
        if collect_outputs:
            return collect_results(mod, pred, loss, cloned_inputs)
        return loss.item()


def torchbench_main():
    original_dir = setup_torchbench_cwd()
    logging.basicConfig(level=logging.WARNING)
    warnings.filterwarnings("ignore")
    main(TorchBenchmarkRunner(), original_dir)


if __name__ == "__main__":
    torchbench_main()
