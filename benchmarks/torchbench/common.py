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

from __future__ import annotations
import argparse
import collections
import contextlib
import copy
import csv
import dataclasses
import functools
import importlib
import itertools
import logging
import os
import pathlib
import random
import shutil
import signal
import subprocess
import sys
import time
import warnings
from contextlib import contextmanager
from typing import Any, Callable, Mapping, NamedTuple, Optional, Tuple, Type
from unittest.mock import MagicMock
import json
import numpy as np
import pandas as pd
import psutil
import torch
import torch._dynamo
import torch._dynamo.utils
import torch.distributed
import torch.fx._pytree as fx_pytree
import torch.multiprocessing as mp
from scipy.stats import gmean, ttest_ind
from torch._dynamo.profiler import fx_insert_profiling, Profiler
from torch._dynamo.testing import dummy_fx_compile, format_speedup, same, reduce_to_scalar_loss
from torch._dynamo.utils import clone_inputs, graph_break_reasons
from torch._functorch.aot_autograd import set_model_name
from torch._inductor import config as inductor_config
from torch._inductor.utils import fresh_inductor_cache
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.utils import _pytree as pytree
from torch.utils._pytree import tree_map, tree_map_only
from tqdm.auto import tqdm, trange
try:
    import torch_npu
    is_npu_available = torch_npu.npu.is_available()
    from npu_support import patch_model
    from profiler import NPUProfiler
except ImportError:
    # ignore the error if torch_npu is not installed
    is_npu_available = False
    from profiler import CUDAProfiler
from benchmark.userbenchmark.dynamo.dynamobench.common import (
    load_model_from_path, Stats, randomize_input, speedup_experiment_ds,
    baselines, null_experiment, DummyGradScaler, cast_to_bf16, cast_to_fp16,
    cast_to_fp64, cast_to_fp32, maybe_fresh_cache, maybe_init_distributed,
)

log = logging.getLogger(__name__)
# We are primarily interested in TF32
torch.backends.cuda.matmul.allow_tf32 = True
# Suppress torch.profiler spam
os.environ["KINETO_LOG_LEVEL"] = "5"
current_name = ""
current_device = ""
current_batch_size = None
output_filename = None
MAX_DOWNLOAD_ATTEMPTS = 5


class PathManager:
    MAX_PATH_LENGTH = 4096
    MAX_FILE_NAME_LENGTH = 255
    DATA_FILE_AUTHORITY = 0o640
    DATA_DIR_AUTHORITY = 0o750

    @classmethod
    def check_path_owner_consistent(cls, path: str):
        if not os.path.exists(path):
            msg = f"The path does not exist: {path}"
            raise RuntimeError(msg)
        if os.stat(path).st_uid != os.getuid():
            warnings.warn(f"Warning: The {path} owner does not match the current user.")

    @classmethod
    def create_file_safety(cls, path: str):
        msg = f"Failed to create file: {path}"
        if os.path.islink(path):
            raise RuntimeError(msg)
        if os.path.exists(path):
            return
        try:
            path = os.path.realpath(path)
            os.close(os.open(path, os.O_WRONLY | os.O_CREAT, cls.DATA_FILE_AUTHORITY))
        except Exception as err:
            raise RuntimeError(msg) from err
        
    @classmethod
    def check_directory_path_readable(cls, path):
        cls.check_path_owner_consistent(path)
        if os.path.islink(path):
            msg = f"Invalid path is a soft chain: {path}"
            raise RuntimeError(msg)
        if not os.access(path, os.R_OK):
            msg = f"The path permission check failed: {path}"
            raise RuntimeError(msg)

    @classmethod
    def check_directory_path_writeable(cls, path):
        cls.check_path_owner_consistent(path)
        if os.path.islink(path):
            msg = f"Invalid path is a soft chain: {path}"
            raise RuntimeError(msg)
        if not os.access(path, os.W_OK):
            msg = f"The path permission check failed: {path}"
            raise RuntimeError(msg)


callbacks = []


def register_callback(callback):
    callbacks.append(callback)


def model_specified_by_path(path_and_class_str):
    return ":" in path_and_class_str


def output_csv(filename, headers, row):
    abspath = os.path.abspath(filename)
    if os.path.exists(filename):
        PathManager.check_directory_path_readable(abspath)
        with open(filename) as fd:
            lines = list(csv.reader(fd)) or [[]]
            if headers and len(headers) > len(lines[0]):
                # if prior results failed the header might not be filled in yet
                lines[0] = headers
            else:
                headers = lines[0]
    else:
        lines = [headers]
    lines.append([(f"{x:.6f}" if isinstance(x, float) else x) for x in row])
    PathManager.create_file_safety(abspath)
    PathManager.check_directory_path_writeable(abspath)
    with open(filename, "w") as fd:
        writer = csv.writer(fd, lineterminator="\n")
        for line in lines:
            writer.writerow(list(line) + ["0"] * (len(headers) - len(line)))


@functools.lru_cache(None)
def patch_torch_manual_seed():
    """Make torch manual seed deterministic. Helps with accuracy testing."""

    def deterministic_torch_manual_seed(*args, **kwargs):
        from torch._C import default_generator

        seed = 1337

        if not torch.cuda._is_in_bad_fork():
            torch.cuda.manual_seed_all(seed)
        if is_npu_available:
            torch_npu.npu.manual_seed_all(seed)
        return default_generator.manual_seed(seed)

    torch.manual_seed = deterministic_torch_manual_seed


def synchronize():
    pass


def timed(
    model,
    model_iter_fn,
    example_inputs,
    times=1,
    return_result=False,
    collect_outputs=False,
):
    synchronize()
    time_total = 0
    # Dont collect outputs to correctly measure timing
    for _ in range(times):
        # Put this call inside the loop to reset the seed for each iteration.
        # Don't include reset_rng_state() to correctly measure timing
        reset_rng_state()
        t_iter_begin = time.perf_counter()
        result = model_iter_fn(model, example_inputs, collect_outputs=collect_outputs)
        t_iter_end = time.perf_counter()
        time_total += t_iter_end - t_iter_begin

    t_0 = time.perf_counter()
    synchronize()
    t_1 = time.perf_counter()
    time_total += t_1 - t_0
    return (time_total, result) if return_result else time_total


def speedup_experiment(args, model_iter_fn, model, example_inputs, **kwargs):
    """
    Measure speedups over eager.

    Writes to ./speedups.csv
    """

    timings = np.zeros((args.repeat, 2), np.float64)
    # if we randomize the input, we should also check the result is correct
    should_randomize_input = args.randomize_input

    from torch._inductor.utils import maybe_profile

    @contextlib.contextmanager
    def maybe_mark_profile(*args, **kwargs):
        prof: torch.profiler.profile = kwargs.pop("p", None)
        mark = kwargs.pop("mark", None)
        if prof:
            with torch.profiler.record_function(mark):
                yield
        else:
            yield

    times = args.iterations_per_run

    tolerance = 1e-4
    torch._dynamo.config.repro_tolerance = tolerance

    with maybe_profile(args.export_profiler_trace) as p:
        frozen_model_iter_fn = torch._dynamo.run(model_iter_fn)

        for rep in trange(args.repeat, desc="running benchmark"):
            inputs = (
                randomize_input(copy.deepcopy(example_inputs))
                if should_randomize_input
                else example_inputs
            )

            # interleave the runs to handle frequency scaling and load changes
            with maybe_mark_profile(p=p, mark="expected"):
                timings[rep, 0], _ = timed(
                    model,
                    model_iter_fn,
                    inputs,
                    return_result=True,
                    times=times,
                    collect_outputs=args.collect_outputs,
                )

            with maybe_mark_profile(p=p, mark="actual"):
                timings[rep, 1], _ = timed(
                    model,
                    frozen_model_iter_fn,
                    inputs,
                    return_result=True,
                    times=times,
                    collect_outputs=args.collect_outputs,
                )

    if args.export_profiler_trace:
        name = args.profiler_trace_name + "_" + model.name + ".json"
        name = os.path.join(torch._dynamo.config.base_dir, name)
        p.export_chrome_trace(name)
    median = np.median(timings, axis=0)
    speedup = median[0] / median[1]

    first_headers = ["dev", "name", "batch_size"]
    first_fields = [current_device, current_name, current_batch_size]
    if "tag" in kwargs:
        first_headers.append("tag")
        first_fields.append(kwargs["tag"])
    headers = first_headers + ["speedup", "abs_latency"]
    row = first_fields + [float(speedup), median[1] * 1000]
    msg = f"{speedup:.3f}x"
    if args.baseline:
        headers.extend(
            [
                "baseline",
                "speedup_vs_baseline",
            ]
        )
        df = pd.read_csv(args.baseline)
        try:
            baseline_speedup = df[df["name"] == current_name]["speedup"].item()
            row.extend([baseline_speedup, speedup / baseline_speedup])
            msg = f"{baseline_speedup:.3f}x -> {speedup:.3f}x [{speedup / baseline_speedup:.3f}x]"
        except (KeyError, ZeroDivisionError):
            row.extend(
                [
                    0.0,
                    0.0,
                ]
            )
    if "compilation_latency" in kwargs:
        headers += [
            "compilation_latency",
            "compression_ratio",
            "eager_peak_mem",
            "dynamo_peak_mem",
        ]
        row.append(kwargs["compilation_latency"])
        row.append(kwargs["compression_ratio"])
        row.append(kwargs["eager_peak_mem"])
        row.append(kwargs["dynamo_peak_mem"])
    if "dynamo_stats" in kwargs:
        for k, v in kwargs["dynamo_stats"].items():
            headers.append(k)
            row.append(v)
    output_csv(
        output_filename,
        headers,
        row,
    )
    headers, data = torch._dynamo.utils.compile_times(repr="csv", aggregate=True)
    if output_filename.find(".csv") <= 0:
        raise AssertionError(f"expected output_filename to be a .csv, but got {output_filename}")
    output_csv(
        output_filename[:-4] + "_compilation_metrics.csv",
        first_headers + headers,
        first_fields + data,
    )
    return msg


def read_batch_size_from_file(args, filename, model_name):
    batch_size = None
    if os.path.exists("benchmarks"):
        filename = os.path.join("benchmarks", filename)
    if not os.path.exists(filename):
        raise AssertionError(filename)
    abspath = os.path.abspath(filename)
    PathManager.check_directory_path_readable(abspath)
    with open(filename) as f:
        lines = f.readlines()
        lines = [i.split(",") for i in lines if len(i.strip()) > 0]
        for val in lines:
            cur_name, b = val
            if model_name == cur_name:
                batch_size = int(b)
    if batch_size is None:
        log.warning("Could not find batch size for %s", model_name)
    elif batch_size == -1:
        raise RuntimeError(
            f"Batch size is unset for {model_name} in {args.batch_size_file}"
        )
    print(f"batch size: {batch_size}")
    return batch_size


def get_peak_memory():
    return torch.cuda.max_memory_allocated() / 10**9


def get_peak_memory_npu():
    return torch_npu.npu.max_memory_allocated() / 10**9


def reset_rng_state():
    torch.manual_seed(1337)
    random.seed(1337)
    np.random.seed(1337)


def get_dynamo_stats():
    # adding a helper to do subtraction on it
    return collections.Counter(
        {
            "calls_captured": torch._dynamo.utils.counters["stats"]["calls_captured"],
            "unique_graphs": torch._dynamo.utils.counters["stats"]["unique_graphs"],
            "graph_breaks": sum(torch._dynamo.utils.counters["graph_break"].values()),
            # NB: The plus removes zero counts
            "unique_graph_breaks": len(+torch._dynamo.utils.counters["graph_break"]),
        }
    )


class BenchmarkRunner:
    def __init__(self):
        self.model_iter_fn = None
        self.grad_scaler = DummyGradScaler()
        self.autocast = contextlib.nullcontext
        self.optimizer = None
        self._args = None

    def setup_amp(self):
        if self.args.only in self.fp32_only_models:
            return

        if self.args.amp and self.args.devices == ["cuda"]:
            # AMP training can lead to small loss values which can undeflow
            # gradient values returning in zero gradients. To solve this
            # problem, PyTorch introduces GradScaler. GradScaler is a stateful
            # structure, that scales the loss values to prevent underflow. Loss
            # values are big at the beginning of training (therefore not
            # requiring scaling), while loss value tends to be small as network
            # starts getting better (requiring scaling). GradScaler manages all
            # of this fine tuning, checking the gradients are turning to inf,
            # discarding such batches.

            # Since we are not running a long iteration, default value of
            # init_scale 65536 is going to turn all gradients to inf. Therefore,
            # we just use a init_scale of 2.0 for benchmarking purpose.

            # Disabling Gradscaler because
            #  1) Benchmark setup runs 2 iterations of fwd-bwd. So, not useful.
            #  2) Current setup shares grad_scaler for eager and dynamo model,
            #  which is bad as Gradscaler has state and can adjust the scaling
            #  factor between eager and dynamo run, making accuracy check
            #  harder.
            self.autocast = torch.cuda.amp.autocast
        elif (self.args.bfloat16 or self.args.amp) and self.args.devices == ["cpu"]:
            self.autocast = torch.cpu.amp.autocast
        elif self.args.amp and self.args.devices == ["npu"]:
            self.autocast = torch_npu.npu.amp.autocast

    def init_optimizer(self, name, device, params, learning_rate=0.01):
        if device == "cuda" and self.args.training:
            self.optimizer = torch.optim.SGD(params, lr=learning_rate, foreach=True)
        elif device == "npu" and self.args.training:
            # Currently, npu dynamo doesn't support foreach=True.
            # There may be changes here in the future.
            self.optimizer = torch.optim.SGD(params, lr=learning_rate, foreach=False)
        else:
            self.optimizer = None

    @property
    def args(self):
        return self._args

    @args.setter
    def args(self, args):
        self._args = args

    @property
    def skip_models(self):
        return set()

    @property
    def skip_models_for_cuda(self):
        return set()

    @property
    def slow_models(self):
        return set()

    @property
    def very_slow_models(self):
        return set()

    @property
    def non_deterministic_models(self):
        return set()

    @property
    def fp32_only_models(self):
        return set()

    @property
    def force_amp_for_fp16_bf16_models(self):
        return set()

    @property
    def failing_torchinductor_models(self):
        return set()

    def get_tolerance_and_cosine_flag(self, is_training, curr_device, name):
        raise NotImplementedError()
    
    def get_learning_rate(self, is_training, curr_device, name):
        raise NotImplementedError()

    @property
    def equal_nan(self):
        equal_nan = True
        if self.args.float32:
            equal_nan = False
        return equal_nan

    def iter_models(self, args):
        for model_name in self.iter_model_names(args):
            for device in args.devices:
                try:
                    yield self.load_model(
                        device,
                        model_name,
                        batch_size=args.batch_size,
                    )
                except NotImplementedError:
                    continue  # bad benchmark implementation

    def deepcopy_model(self, model):
        return copy.deepcopy(model)

    def cast_based_on_args(self, model, example_inputs):
        if self.args.float32 or self.args.only in self.fp32_only_models:
            if not self.args.float32:
                log.warning("Model %s supports float32 only", self.args.only)
            model, example_inputs = cast_to_fp32(model, example_inputs)
        elif self.args.float16:
            if self.args.only in self.force_amp_for_fp16_bf16_models:
                log.warning(
                    "Model %s does not support float16, running with amp instead",
                    self.args.only,
                )
                self.args.amp = True
                self.setup_amp()
            else:
                model, example_inputs = cast_to_fp16(model, example_inputs)
        elif self.args.bfloat16:
            if self.args.only in self.force_amp_for_fp16_bf16_models:
                log.warning(
                    "Model %s does not support bfloat16, running with amp instead",
                    self.args.only,
                )
                self.args.amp = True
                self.setup_amp()
            else:
                model, example_inputs = cast_to_bf16(model, example_inputs)

        return model, example_inputs

    def validate_model(self, model, example_inputs):
        """
        Runs the eager model with example inputs to ensure that eager passes.
        """
        model = self.deepcopy_model(model)
        example_inputs = clone_inputs(example_inputs)
        model, example_inputs = self.cast_based_on_args(model, example_inputs)
        try:
            self.model_iter_fn(model, example_inputs)
        except Exception as e:
            raise NotImplementedError("Eager model failed to run") from e

    def maybe_cast(self, model, example_inputs):
        model = self.deepcopy_model(model)
        example_inputs = clone_inputs(example_inputs)
        model, example_inputs = self.cast_based_on_args(model, example_inputs)
        return model, example_inputs

    def run_n_iterations(self, mod, inputs, run_mode=None):
        n = self.args.iterations
        if run_mode is None:
            for _ in range(n - 1):
                self.model_iter_fn(mod, inputs, collect_outputs=False)
            return self.model_iter_fn(mod, inputs, collect_outputs=True)

        start_step = int(n * 0.3)
        end_step = min(int(n * 0.8) - 1, n - 1)

        step_times = []
        prof_output_dir = os.path.join(self.args.prof_output_path, self.args.only, run_mode)
        if is_npu_available:
            prof = NPUProfiler(enable=self.args.enable_profiler, warmup=10, active=n, save_path=prof_output_dir)
        else:
            prof = CUDAProfiler(enable=self.args.enable_profiler, warmup=10, active=n, save_path=prof_output_dir)

        prof.start()
        for i in range(n):
            start = time.perf_counter()
            output = self.model_iter_fn(mod, inputs, collect_outputs=(i == n - 1))
            synchronize()
            end = time.perf_counter()
            elapsed_ms = (end - start) * 1000
            prof.step()
            step_times.append(elapsed_ms)
            if i != n - 1:
                print(f"[{run_mode}] step: {i+1} step_time: {elapsed_ms} ms loss: {output}")
            else:
                print(f"[{run_mode}] step: {i+1} step_time: {elapsed_ms} ms")
        prof.stop()

        steps = step_times[start_step:end_step + 1]
        if steps:
            total_ms = sum(steps)
            avg_ms = total_ms / len(steps)
            print(f"[{run_mode}] summary [{start_step+1}-{end_step+1}] "
                f"total steps time: {total_ms:.2f} ms, "
                f"avg step time: {avg_ms:.2f} ms")

        return output

    def optimizer_zero_grad(self, mod):
        if self.optimizer is not None:
            self.optimizer.zero_grad(True)
        else:
            mod.zero_grad(True)

    def optimizer_step(self):
        if self.optimizer is not None:
            self.optimizer.step()

    def deepcopy_and_maybe_ddp(self, model):
        model = self.deepcopy_model(model)
        if self.args.ddp:
            if not torch.distributed.is_available():
                raise AssertionError("Can't use DDP without a distributed enabled build")
            from torch.nn.parallel import DistributedDataParallel as DDP

            model = DDP(model, find_unused_parameters=True)
        return model

    def check_accuracy(
        self, name, model, example_inputs, optimize_ctx, experiment, tag
    ):
        """
        Checks accuracy.
        1) Collect the outputs with fp64 datatype. This is useful for error checking.
        2) Checks if eager itself has variations.
        """
        start_stats = get_dynamo_stats()
        lr = self.get_learning_rate(self.args.training, current_device, name)
        print(f"learning rate: {lr}")
        
        def record_status(accuracy_status, dynamo_start_stats):
            """
            Records the status in the csv file
            """
            if current_name in self.non_deterministic_models:
                if accuracy_status in (
                    "pass_accuracy",
                    "eager_two_runs_differ",
                    "fail_accuracy",
                ):
                    accuracy_status = "pass_accuracy"

            headers = ["dev", "name", "batch_size", "accuracy"]
            fields = [current_device, current_name, current_batch_size, accuracy_status]

            if tag is not None:
                headers.insert(3, "tag")
                fields.insert(3, tag)

            dynamo_stats = get_dynamo_stats()
            dynamo_stats.subtract(dynamo_start_stats)
            for k, v in dynamo_stats.items():
                headers.append(k)
                fields.append(v)

            output_csv(output_filename, headers, fields)
            return accuracy_status

        # Collect the fp64 reference outputs to be used later for accuracy checking.
        fp64_outputs = None
        try:
            model_fp64, inputs_fp64 = cast_to_fp64(
                self.deepcopy_and_maybe_ddp(model),
                clone_inputs(example_inputs),
            )
            self.init_optimizer(name, current_device, model_fp64.parameters(), lr)
            fp64_outputs = self.run_n_iterations(model_fp64, inputs_fp64)
            fp64_outputs = tree_map(
                lambda x: x.to(torch.float64)
                if isinstance(x, torch.Tensor) and x.is_floating_point()
                else x,
                fp64_outputs,
            )
        except Exception:
            log.warning(
                "fp64 golden ref were not generated for %s. Setting accuracy check to cosine",
                name,
            )
            self.args.cosine = True
            fp64_outputs = None

        tolerance, cos_similarity = self.get_tolerance_and_cosine_flag(
            self.args.training, current_device, name
        )
        print(f"tolerance: {tolerance}")

        # Cast the model to float16/float32 as necessary
        model, example_inputs = self.maybe_cast(model, example_inputs)
        accuracy_status = "pass_accuracy"


        with self.pick_grad(name, self.args.training):
            # Get results of native pytorch
            reset_rng_state()
            try:
                model_copy = self.deepcopy_and_maybe_ddp(model)
                self.init_optimizer(name, current_device, model_copy.parameters(), lr)
                correct_result = self.run_n_iterations(
                    model_copy, clone_inputs(example_inputs), "eager"
                )
            except Exception as e:
                accuracy_status = (
                    "eager_1st_run_OOM"
                    if isinstance(e, torch.cuda.OutOfMemoryError)
                    else "eager_1st_run_fail"
                )
                log.exception(e)
                return record_status(accuracy_status, dynamo_start_stats=start_stats)

            # Rerun native pytorch
            reset_rng_state()
            try:
                model_copy = self.deepcopy_and_maybe_ddp(model)
                self.init_optimizer(name, current_device, model_copy.parameters(), lr)
                correct_rerun_result = self.run_n_iterations(
                    model_copy, clone_inputs(example_inputs)
                )
            except Exception as e:
                accuracy_status = (
                    "eager_2nd_run_OOM"
                    if isinstance(e, torch.cuda.OutOfMemoryError)
                    else "eager_2nd_run_fail"
                )
                return record_status(accuracy_status, dynamo_start_stats=start_stats)

            # Two eager runs should have exactly same result
            is_same = True
            try:
                if (
                    not same(
                        correct_result,
                        correct_rerun_result,
                        fp64_ref=None,
                        cos_similarity=False,
                        tol=0,
                        equal_nan=self.equal_nan,
                    )
                ):
                    is_same = False
            except Exception as e:
                # Sometimes torch.allclose may throw RuntimeError
                is_same = False

            if not is_same:
                accuracy_status = "eager_two_runs_differ"
                return record_status(accuracy_status, dynamo_start_stats=start_stats)

            correct_rerun_result = None

            # Run with Dynamo
            reset_rng_state()
            torch._dynamo.reset()
            try:
                model_copy = self.deepcopy_and_maybe_ddp(model)
                self.init_optimizer(name, current_device, model_copy.parameters(), lr)
                optimized_model = optimize_ctx(model_copy)
                new_result = self.run_n_iterations(optimized_model, example_inputs, "compile")
            except Exception as e:
                log.exception(e)
                print(
                    "TorchDynamo optimized model failed to run because of following error"
                )
                accuracy_status = (
                    "OOM"
                    if isinstance(e, torch.cuda.OutOfMemoryError)
                    else "fail_to_run"
                )
                return record_status(accuracy_status, dynamo_start_stats=start_stats)

            try:
                if not same(
                    correct_result,
                    new_result,
                    fp64_outputs,
                    equal_nan=self.equal_nan,
                    cos_similarity=cos_similarity,
                    tol=tolerance,
                ):
                    is_same = False
            except Exception as e:
                # Sometimes torch.allclose may throw RuntimeError
                is_same = False

            if not is_same:
                accuracy_status = "fail_accuracy"
                return record_status(accuracy_status, dynamo_start_stats=start_stats)

        return record_status(accuracy_status, dynamo_start_stats=start_stats)

    def run_performance_test(
        self, name, model, example_inputs, optimize_ctx, experiment, tag=None
    ):
        def warmup(fn, model, example_inputs, mode, niters=5):
            peak_mem = 0
            start_stats = get_dynamo_stats()
            try:
                if current_device == "cuda":
                    torch.cuda.reset_peak_memory_stats()
                    torch.cuda.empty_cache()
                elif current_device == "npu":
                    torch_npu.npu.reset_peak_memory_stats()
                    torch_npu.npu.empty_cache()
                t0 = time.perf_counter()
                for _ in range(niters):
                    fn(model, example_inputs)
                t1 = time.perf_counter()
                latency = t1 - t0
                if current_device == "cuda":
                    peak_mem = get_peak_memory()
                elif current_device == "npu":
                    peak_mem = get_peak_memory_npu()
                elif current_device == "cpu":
                    total = psutil.virtual_memory().total
                    percentage = psutil.Process(os.getpid()).memory_percent()
                    peak_mem = percentage * total / 10**9
            except Exception as e:
                log.exception("Backend %s failed in warmup()", mode)
                raise RuntimeError(f"Backend {mode} failed in warmup()") from e
            dynamo_stats = get_dynamo_stats()
            dynamo_stats.subtract(start_stats)
            return latency, peak_mem, dynamo_stats

        # Cast the model to float16/float32 as necessary
        model, example_inputs = self.maybe_cast(model, example_inputs)

        model = self.deepcopy_and_maybe_ddp(model)

        self.init_optimizer(name, current_device, model.parameters())
        with self.pick_grad(name, self.args.training):
            ok, total = Stats.reset_counters()
            experiment_kwargs = {}
            if tag is not None:
                experiment_kwargs["tag"] = tag
            results = []
            eager_latency, eager_peak_mem, _ = warmup(
                self.model_iter_fn, model, example_inputs, "eager"
            )
            optimized_model_iter_fn = optimize_ctx(self.model_iter_fn)
            dynamo_latency, dynamo_peak_mem, dynamo_stats = warmup(
                optimized_model_iter_fn, model, example_inputs, "dynamo"
            )

            compilation_time = dynamo_latency - eager_latency
            compression_ratio = (
                eager_peak_mem / dynamo_peak_mem if dynamo_peak_mem else 0.0
            )

            if experiment.func is speedup_experiment:
                experiment_kwargs["compilation_latency"] = compilation_time
                experiment_kwargs["compression_ratio"] = compression_ratio
                experiment_kwargs["eager_peak_mem"] = eager_peak_mem
                experiment_kwargs["dynamo_peak_mem"] = dynamo_peak_mem
                experiment_kwargs["dynamo_stats"] = dynamo_stats

            if not hasattr(model, name):
                model.name = name
            results.append(experiment(model, example_inputs, **experiment_kwargs))
            return " ".join(map(str, results))
        

    def _get_precision_checker_cast_dtype(self):
        if self.args.precision_checker_cast_dtype is None:
            return None
        dtype_map = {
            "bf16": torch.bfloat16,
            "bfloat16": torch.bfloat16,
            "fp16": torch.float16,
            "float16": torch.float16,
            "fp32": torch.float32,
            "float32": torch.float32,
        }
        return dtype_map[self.args.precision_checker_cast_dtype]


    def run_precision_checker_test(
        self, name, model, example_inputs, optimize_ctx, experiment, tag=None
    ):
        import msprobe
        from msprobe.pytorch.compile_accuracy_checker.precision_checker import PrecisionChecker

        start_stats = get_dynamo_stats()

        def record_status(status, result=None, error=None):
            headers = ["dev", "name", "batch_size", "precision_checker"]
            fields = [current_device, current_name, current_batch_size, status]
            if tag is not None:
                headers.insert(3, "tag")
                fields.insert(3, tag)

            if result is not None:
                def diff_passed(diff):
                    if diff.note:
                        return diff.note.startswith("SKIP") or diff.note == "IGNORED"
                    return all(
                        vals is None or all(item.allclose for item in vals)
                        for vals in (
                            diff.fwd_input,
                            diff.fwd_output,
                            diff.grad_input,
                            diff.grad_output,
                        )
                    )

                failed_modules = sum(
                    1
                    for diff in result.diffs
                    if not diff_passed(diff)
                )
                headers.extend(
                    ["loss_eager", "loss_compiled", "loss_diff", "failed_modules"]
                )
                fields.extend(
                    [
                        result.loss_eager,
                        result.loss_compiled,
                        result.loss_diff,
                        failed_modules,
                    ]
                )
            if error is not None:
                headers.append("error")
                fields.append(type(error).__name__)

            dynamo_stats = get_dynamo_stats()
            dynamo_stats.subtract(start_stats)
            for k, v in dynamo_stats.items():
                headers.append(k)
                fields.append(v)
            output_csv(output_filename, headers, fields)
            return status

        model, example_inputs = self.maybe_cast(model, example_inputs)
        model = self.deepcopy_and_maybe_ddp(model)

        checker = PrecisionChecker(
            backend="aot_eager",
            dump_graphs=self.args.precision_checker_dump_graphs,
            graph_dir=self.args.precision_checker_graph_dir,
            cast_dtype=self._get_precision_checker_cast_dtype(),
            capture_input=self.args.precision_checker_capture_input,
            single_pass=self.args.precision_checker_single_pass,
        )
        checker.wrap(model)
        print(f"precision_checker: backend=aot_eager")

        def run_step(mod):
            cloned_inputs = clone_inputs(example_inputs)
            mod.zero_grad(True)
            with self.autocast():
                output = mod(*cloned_inputs)
                loss = reduce_to_scalar_loss(output)
            if loss.requires_grad:
                self.grad_scaler.scale(loss).backward()
            return loss

        reset_rng_state()
        torch._dynamo.reset()
        try:
            with torch.enable_grad():
                result = checker.compare(run_step, model)
        except Exception as e:
            log.exception(e)
            status = (
                "OOM"
                if isinstance(e, torch.cuda.OutOfMemoryError)
                else "fail_to_run"
            )
            return record_status(status, error=e)

        checker.report(result)
        status = "pass_precision_checker" if result.all_pass else "fail_precision_checker"
        return record_status(status, result=result)
    

    def run_one_model(
        self,
        name,
        model,
        example_inputs,
        optimize_ctx,
        experiment,
        tag=None,
    ):
        mode = "train" if self.args.training else "eval"
        msg = f"{current_device:4} {mode:5} {current_name:34} "
        if tag:
            msg += f" {tag:26}"
        print(msg, flush=True)

        start_stats = get_dynamo_stats()

        if self.args.accuracy:
            status = self.check_accuracy(
                name, model, example_inputs, optimize_ctx, experiment, tag
            )
            print(status)
            if self.args.dump_compile_time:
                headers, values = torch._dynamo.utils.compile_times("csv")
                for header, value in zip(headers, values):
                    if header == "async_compile.wait":
                        numbers = [float(num.strip()) for num in value.split(',') if num.strip()]
                        op_compile_time = sum(numbers)
                print(f"op_compile_time:{op_compile_time * 1e3} ms", )
            
        elif self.args.performance:
            status = self.run_performance_test(
                name, model, example_inputs, optimize_ctx, experiment, tag
            )
            print(status)
        elif self.args.precision_checker:
            status = self.run_precision_checker_test(
                name, model, example_inputs, optimize_ctx, experiment, tag
            )
            print(status)
        stats = get_dynamo_stats()
        stats.subtract(start_stats)

        if self.args.log_graph_breaks or self.args.print_graph_breaks:
            filename = f"{output_filename.rstrip('.csv')}_graph_breaks.csv"

            def add_double_quotes(x):
                # Delimiter because reason could have comma
                return f'"{x}"'

            for graph_break in graph_break_reasons:
                reason = add_double_quotes(graph_break.reason)
                user_stack = add_double_quotes(
                    ", ".join([str(x) for x in graph_break.user_stack])
                )
                output_csv(
                    filename,
                    ["model", "reason", "user_stack"],
                    [current_name, reason, user_stack],
                )

        if self.args.stats:
            Stats.print_summary()


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--devices", "--device", "-d", action="append", help="cpu or cuda"
    )
    parser.add_argument("--device-index", help="CUDA device index")
    parser.add_argument(
        "--repeat", "-n", type=int, default=30, help="number of timing runs"
    )
    iterations_per_run_help = """
        Run this may iterations for each time measurement. This is mainly used for
        XLA training. We want to run multiple iterations per measurement so the
        tracing and computation for different iteartions can overlap with each
        other. This makes sure we have an accurate xla baseline.
    """
    parser.add_argument(
        "--iterations-per-run", type=int, default=1, help=iterations_per_run_help
    )
    parser.add_argument(
        "--randomize-input",
        action="store_true",
        help="Whether to randomize the input values. Dimensions will be kept the same.",
    )
    parser.add_argument(
        "--nopython", action="store_true", help="Turn graph breaks into errors"
    )
    parser.add_argument(
        "--no-skip",
        action="store_true",
        help="run models that are in the global SKIP list",
    )
    parser.add_argument(
        "--batch-size", "--batch_size", type=int, help="batch size for benchmarking"
    )
    parser.add_argument(
        "--iterations", type=int, default=50, help="how many iterations to run"
    )
    parser.add_argument(
        "--batch-size-file", type=str, help="String to load batch size from"
    )
    parser.add_argument("--cosine", action="store_true", help="use cosine similarity")
    parser.add_argument(
        "--cpp-wrapper", action="store_true", help="turn on cpp/cuda wrapper codegen"
    )
    parser.add_argument(
        "--freezing", action="store_true", help="turn on freezing", default=False
    )
    parser.add_argument(
        "--only",
        help="""Run just one model from torchbench. Or
        specify the path and class name of the model in format like:
        --only=path:<MODEL_FILE_PATH>,class:<CLASS_NAME>

        Due to the fact that dynamo changes current working directory,
        the path should be an absolute path.

        The class should have a method get_example_inputs to return the inputs
        for the model. An example looks like
        ```
        class LinearModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(10, 10)

            def forward(self, x):
                return self.linear(x)

            def get_example_inputs(self):
                return (torch.randn(2, 10),)
        ```
    """,
    )
    parser.add_argument(
        "--multiprocess",
        action="store_true",
        help="Create n processes based on the number of devices (distributed use case).",
    )
    parser.add_argument(
        "--ddp",
        action="store_true",
        help="Wraps model in DDP before running it, and uses dynamo DDPOptmizer (graph breaks) by default.",
    )
    parser.add_argument(
        "--distributed-master-port",
        default="6789",
        help="Port to bind for for torch.distributed.  Use the default unless it's conflicting with another user",
    )
    parser.add_argument(
        "--dynamic-shapes",
        action="store_true",
        help="Runs a dynamic shapes version of the benchmark, if available.",
    )
    parser.add_argument(
        "--output",
        help="Overrides the output filename",
    )
    parser.add_argument(
        "--output-directory",
        help="Overrides the directory to place output files.",
    )
    parser.add_argument(
        "--baseline",
        help="Compare with a prior --output",
    )
    parser.add_argument(
        "--part",
        default=None,
        help="Specify the part of the model to run.",
    )
    parser.add_argument(
        "--export-profiler-trace",
        action="store_true",
        help="exports trace of kineto profiler",
    )
    parser.add_argument(
        "--profiler-trace-name",
        "--profiler_trace_name",
        help="Overwrites exported trace name",
    )
    parser.add_argument(
        "--tag", default=None, help="Specify a tag to be included in csv files."
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="print graph counter stats",
    )
    parser.add_argument(
        "--cold-start-latency",
        "--cold_start_latency",
        action="store_true",
        help="Use a fresh triton cachedir when running each model, to force cold-start compile.",
    )
    parser.add_argument(
        "--disable-aclgraph",
        action="store_true",
        help="Disables aclgraph for NPU Inductor",
    )
    parser.add_argument(
        "--disable-split-reductions",
        action="store_true",
        help="Disables split reductions for Inductor",
    )
    parser.add_argument(
        "--disable-persistent-reductions",
        action="store_true",
        help="Disables split reductions for Inductor",
    )
    parser.add_argument(
        "--print-graph-breaks",
        action="store_true",
        help="Show a warning whenever graph break",
    )
    parser.add_argument(
        "--log-graph-breaks",
        action="store_true",
        help="log graph breaks in a file",
    )
    parser.add_argument(
        "--collect-outputs",
        action="store_true",
        help="""Whether to collect outputs for training. Set this to true if we
        want to verify the numerical correctness of graidents. But that may
        cause time measurement not accurate""",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=2000,
        help="timeout (second) for benchmarking.",
    )
    parser.add_argument(
        "--enable-profiler",
        action="store_true",
        help="Enable profile for NPU and GPU."
    )
    parser.add_argument(
        "--prof-output-path",
        help="Overrides the profile output path",
        default="./profile"
    )
    parser.add_argument(
        "--dump-compile-time",
        action="store_true",
        help="dump compile time",
    )
    group_prec = parser.add_mutually_exclusive_group()
    group_prec.add_argument("--float16", action="store_true", help="cast model to fp16")
    group_prec.add_argument(
        "--bfloat16", action="store_true", help="cast model to bf16"
    )
    group_prec.add_argument("--float32", action="store_true", help="cast model to fp32")
    group_prec.add_argument(
        "--amp", action="store_true", help="use automatic mixed precision"
    )
    parser.add_argument(
        "--backend",
        choices=torch._dynamo.list_backends(exclude_tags=None),
        default="inductor",
        help="measure speedup with a given backend",
    )
    parser.add_argument(
        "--npu-backend",
        choices=["mlir", "dvm", "triton", "default"],
        default="default",
        help="Specify NPU backend (only effective when --backend is inductor)",
    )
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument(
        "--accuracy",
        action="store_true",
        help="Checks accuracy with small batch size and eval mode",
    )
    mode_group.add_argument(
        "--performance", action="store_true", help="Measures performance speedup"
    )
    mode_group.add_argument(
        "--precision-checker",
        action="store_true",
        help="Runs torch_npu precision checker for eager vs compiled module diffs",
    )
    parser.add_argument(
        "--precision-checker-capture-input",
        action="store_true",
        help="Also compare inputs captured by forward pre-hooks",
    )
    parser.add_argument(
        "--precision-checker-single-pass",
        action="store_true",
        help="Run only the compiled pass and compare wrapped module outputs in hooks",
    )
    parser.add_argument(
        "--precision-checker-dump-graphs",
        action="store_true",
        help="Dump FX graphs captured during precision checker compilation",
    )
    parser.add_argument(
        "--precision-checker-graph-dir",
        default="./graph_dump",
        help="Directory for precision checker graph dumps",
    )
    parser.add_argument(
        "--precision-checker-cast-dtype",
        choices=["bf16", "bfloat16", "fp16", "float16", "fp32", "float32"],
        help="Cast compiled-side inputs to this dtype inside precision checker",
    )
    parser.add_argument(
        "--precision-checker-modules",
        nargs="+",
        help="Exact module names to wrap. Defaults to all leaf modules.",
    )
    parser.add_argument(
        "--precision-checker-module-types",
        nargs="+",
        help="Module class names to wrap, for example Linear Conv2d LayerNorm.",
    )
    parser.add_argument(
        "--precision-checker-ignore-modules",
        nargs="+",
        help="Exact module names to ignore in precision checker reports.",
    )
    run_mode_group = parser.add_mutually_exclusive_group(required=True)
    run_mode_group.add_argument(
        "--training",
        action="store_true",
        help="Performs training",
    )
    run_mode_group.add_argument(
        "--inference", action="store_true", help="Performs inference"
    )
    return parser.parse_args(args)


def process_entry(rank, runner, original_dir, args):
    args.rank = rank
    with maybe_init_distributed(
        args.use_distributed,
        rank=rank,
        world_size=args.world_size,
        port=args.distributed_master_port,
    ):
        return maybe_fresh_cache(
            run, (args.cold_start_latency and args.only)
        )(runner, args, original_dir)


def main(runner, original_dir=None):
    if original_dir:
        os.chdir(original_dir)
    args = parse_args()
    if args.baseline:
        args.baseline = os.path.abspath(args.baseline)

    args.use_distributed = (args.ddp) and args.only
    if args.multiprocess:
        # NB: Do NOT query device count before CUDA initialization; we're
        # going to overwrite CUDA_VISIBLE_DEVICES and this will result in
        device_count = torch.cuda.device_count()
        if device_count <= 1:
            log.warning(
                "The use multiprocess flag is set but there are <= 1 devices available."
            )
        # multiprocess path
        args.world_size = device_count
        mp.spawn(process_entry, args=(runner, original_dir, args), nprocs=device_count)
    else:
        # single process path just uses the main process
        args.world_size = 1
        process_entry(0, runner, original_dir, args)


def run(runner, args, original_dir=None):
    # Pass the parsed args object to benchmark runner object
    runner.args = args
    experiment = null_experiment
    global current_name, current_device, current_batch_size, output_filename
    optimize_ctx = contextlib.nullcontext()

    if args.backend:
        optimize_ctx = configure_compile_options(args)
        experiment = speedup_experiment
        if args.accuracy:
            output_filename = f"accuracy_{args.backend}.csv"
        elif args.precision_checker:
            output_filename=f"precision_checker_{args.backend}.csv"
        else:
            output_filename = f"speedup_{args.backend}.csv"

    if args.ddp:
        # but just to measure impact on singlenode of performing graph-breaks.
        # Left it as a follow up to keep this PR isolated.
        if not args.accuracy:
            raise AssertionError("DDP benchmark is currently only hooked up to --accuracy bench")
        if not args.training:
            raise AssertionError("DDP benchmark requires --training mode")
    if args.accuracy or args.precision_checker:
        # Use small batch size. We use >1 batch size to ensure we test
        # batch_norm type of operators that work on batch dims.
        if args.batch_size is None:
            if runner.suite_name == "huggingface":
                args.batch_size = 1
            elif runner.suite_name == "torchbench":
                args.batch_size = 4
            else:
                # Larger batch size of TIMM models to have stable batch_norm
                if runner.suite_name != "timm_models":
                    raise AssertionError
                args.batch_size = 8

        # Remove sources of randomness
        inductor_config.fallback_random = True
        if args.only is not None and args.only not in {
            "alexnet",
            "Background_Matting",
            "pytorch_CycleGAN_and_pix2pix",
            "pytorch_unet",
            "Super_SloMo",
            "vgg16",
            "Wav2Vec2ForCTC",
            "Wav2Vec2ForPreTraining",
            "sam",
        }:
            # some of the models do not support use_deterministic_algorithms
            torch.use_deterministic_algorithms(True)
        else:
            log.warning("Currently, all models keep deterministic open on npu. "
                        "But on gpu, this model does not support use_deterministic_algorithms. "
                        "Please check it to prevent bugs.")
            torch.use_deterministic_algorithms(True)

        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False

        # Remove randomeness when torch manual seed is called
        patch_torch_manual_seed()

        # Some models e.g. yolov3 assert batch size on n_gpus
        if "CUDA_VISIBLE_DEVICES" not in os.environ:
            args.device_index = "0"

    if args.device_index is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device_index

    def __check_if_transfer_to_npu():
        return torch.cuda.is_available == is_npu_available

    if not args.devices:
        if torch.cuda.is_available():
            if __check_if_transfer_to_npu():
                args.devices = ["npu"]
            else:
                args.devices = ["cuda"]
        elif is_npu_available:
            args.devices = ["npu"]
        else:
            log.warning("torch.cuda.is_available() == False, using CPU")
            args.devices = ["cpu"]

    if args.devices != ["cpu"] and torch.cuda.is_available():
        global synchronize
        synchronize = torch.cuda.synchronize
    elif is_npu_available:
        synchronize = torch_npu.npu.synchronize

    if (
        args.devices == ["cuda"]
        and torch.cuda.get_device_properties(0).total_memory < 25 * 2**30
    ):
        # OOM errors on an RTX 3090 with 24gb RAM
        runner.skip_models.update(
            {
                # torchbench
                "hf_Longformer",
                "timm_nfnet",
                "timm_efficientdet",
            }
        )
        if args.training:
            runner.skip_models.add("hf_T5")

    if args.print_graph_breaks:
        torch._dynamo.config.print_graph_breaks = True

    if args.training:
        runner.model_iter_fn = runner.forward_and_backward_pass
    else:
        runner.model_iter_fn = runner.forward_pass

    if args.devices == ["cpu"]:
        runner.skip_models.update(runner.very_slow_models)

    if args.no_skip:
        runner.skip_models.clear()

    runner.setup_amp()

    if args.output:
        output_filename = args.output

    if output_filename:
        if args.output_directory:
            output_filename = os.path.join(args.output_directory, output_filename)
        else:
            output_filename = os.path.join(
                # pytorch use torch._dynamo.config.base_dir originally, 
                # but the generated file will be saved under directory where pytorch was installed,
                # change the default output saved directory.
                os.path.dirname(os.path.abspath(__file__)), output_filename
            )

    if args.export_profiler_trace:
        if args.profiler_trace_name is None:
            if args.backend:
                args.profiler_trace_name = args.backend
            else:
                args.profiler_trace_name = "profile"

    experiment = functools.partial(experiment, args, runner.model_iter_fn)

    if args.only:
        # use aclnn by default, otherwise compared with aclop
        if os.environ.get("USE_ACLOP", "0").upper() in ["1", "ON"]:
            torch_npu.npu.set_compile_mode(jit_compile=True)

        if is_npu_available:
            patch_model(args.only)

        model_name = args.only
        for device in args.devices:
            batch_size = args.batch_size
            if args.batch_size_file:
                batch_size = read_batch_size_from_file(
                    args, args.batch_size_file, model_name
                )
            if model_specified_by_path(args.only):
                model, example_inputs = load_model_from_path(args.only)
                name = model.__class__.__name__
                model = model.to(device=device)
                example_inputs = tree_map_only(
                    torch.Tensor, lambda x: x.to(device=device), example_inputs
                )
            else:
                try:
                    with tqdm(desc="loading model"):
                        if args.part:
                            (
                                device,
                                name,
                                model,
                                example_inputs,
                                batch_size,
                            ) = runner.load_model(
                                device,
                                model_name,
                                batch_size=batch_size,
                                part=args.part,
                            )
                        else:
                            (
                                device,
                                name,
                                model,
                                example_inputs,
                                batch_size,
                            ) = runner.load_model(
                                device, model_name, batch_size=batch_size
                            )
                except NotImplementedError as e:
                    print(e)
                    import traceback

                    print(traceback.format_exc())
                    logging.warning("%s failed to load", args.only)
                    continue  # bad benchmark implementation

            current_name = name
            current_device = device
            current_batch_size = batch_size
            set_model_name(name)

            # Look for stuff that looks like batch size, and mark it dynamic.
            # Better integration would integrate directly with benchmark suite
            # but cannot conveniently do this
            # NB: This must be done late enough so that we don't do more
            # conversions on the inputs
            # NB: Assumes only the first batch-y like dimension is the batch
            marked = False

            def detect_and_mark_batch(t, target_size=batch_size):
                nonlocal marked
                for i, s in enumerate(t.size()):
                    if s == target_size:
                        torch._dynamo.mark_dynamic(t, i)
                        marked = True
                        break

            model, example_inputs = runner.cast_based_on_args(model, example_inputs)
            runner.run_one_model(
                name,
                model,
                example_inputs,
                optimize_ctx,
                experiment,
                tag=args.tag,
            )
        # exec callback functions registered in npu_support.py
        for fn in callbacks:
            fn()
    else:
        if output_filename and os.path.exists(output_filename):
            os.unlink(output_filename)
        if original_dir:
            os.chdir(original_dir)
        model_names = list(runner.iter_model_names(args))
        nmodels = len(model_names)
        for i, name in enumerate(model_names):
            current_name = name
            placeholder_batch_size = 0
            print(f"Running model {i+1}/{nmodels}", flush=True)

            def write_csv(status, name=name, placeholder_batch_size=placeholder_batch_size):
                if args.accuracy:
                    headers = ["dev", "name", "batch_size", "accuracy"]
                    rows = [
                        [device, name, placeholder_batch_size, status]
                        for device in args.devices
                    ]
                elif args.precision_checker:
                    headers = ["dev", "name", "batch_size", "precision_checker"]
                    rows = [
                        [device, name, placeholder_batch_size, status]
                        for device in args.devices
                    ]
                elif args.performance:
                    headers = ["dev", "name", "batch_size", "speedup", "abs_latency"]
                    rows = [
                        [device, name, placeholder_batch_size, 0.0, 0.0]
                        for device in args.devices
                    ]
                else:
                    headers = []
                    rows = [
                        [device, name, placeholder_batch_size, 0.0]
                        for device in args.devices
                    ]

                for row in rows:
                    output_csv(output_filename, headers, row)

            try:
                subprocess.check_call(
                    [sys.executable] + sys.argv + [f"--only={name}"], timeout=args.timeout
                )
            except subprocess.TimeoutExpired:
                print("TIMEOUT", file=sys.stderr)
                write_csv("timeout")
            except subprocess.SubprocessError:
                print("ERROR", file=sys.stderr)
                write_csv("infra_error")


def get_npu_backend(args):
    if not hasattr(get_npu_backend, "_config_cache"):
        try:
            with open("./npu_backend_config.json", "r") as f:
                get_npu_backend._config_cache = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            log.warning(f"Failed to load npu_backend_config.json: {e}, using default backend 'dvm'")
            get_npu_backend._config_cache = None

    if get_npu_backend._config_cache is None:
        return "dvm"

    for item in get_npu_backend._config_cache:
        if item.get("model") == args.only:
            return item.get("mode", "dvm")

    return "dvm"


def configure_compile_options(args):
    try:
        from torchbench import NPU_DVM_NO_ACLGRAPH, NPU_MLIR_NO_ACLGRAPH
    except ImportError:
        NPU_DVM_NO_ACLGRAPH = set()
        NPU_MLIR_NO_ACLGRAPH = set()
    npu_backend = args.npu_backend
    # mode Config
    if not args.disable_aclgraph:
        mode = "max-autotune"
    else:
        mode = None
    if args.only is not None:
        if npu_backend == "dvm" and args.only in NPU_DVM_NO_ACLGRAPH:
            mode = None
        elif npu_backend == "mlir" and args.only in NPU_MLIR_NO_ACLGRAPH:
            mode = None
    # Backend Config
    backend = None
    if args.backend:
        backend = args.backend
    else:
        backend = None

    # Dynamic shape Config
    dynamic = None
    if args.dynamic_shapes:
        dynamic = True

    if backend:
        compile_kwargs = {
            "backend": backend,
            "fullgraph": args.nopython,
            "dynamic": dynamic,
        }
        compile_kwargs["mode"] = mode
        if backend == "inductor" and hasattr(args, 'npu_backend'):
            if npu_backend == "default":
                npu_backend = get_npu_backend(args)
            if npu_backend in ["mlir", "dvm"]:
                os.environ['TORCHINDUCTOR_NPU_BACKEND'] = npu_backend
        optimize_ctx = functools.partial(torch.compile, **compile_kwargs)
    else:
        optimize_ctx = contextlib.nullcontext()

    return optimize_ctx


if __name__ == "__main__":
    raise RuntimeError(
        f"You shouldn't run {sys.argv[0]} directly, instead try torchbench.py"
    )
