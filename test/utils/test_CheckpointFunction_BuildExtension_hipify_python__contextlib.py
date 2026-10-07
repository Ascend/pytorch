# Copyright (c) 2026 Huawei Technologies Co., Ltd
# All rights reserved.
#
# Licensed under the BSD 3-Clause License  (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://opensource.org/licenses/BSD-3-Clause
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Add validation cases for torch.utils checkpoint, cpp_extension, hipify and
contextlib related APIs on NPU.

This file validates the core behavior of the following APIs:
1. torch.utils.checkpoint.CheckpointFunction.backward
2. torch.utils.cpp_extension.BuildExtension.with_options
3. torch.utils.hipify.hipify_python.hipify
4. torch.utils._contextlib._DecoratorContextManager
"""

import os
import tempfile
import textwrap

import torch
from setuptools import Distribution
from torch.utils.checkpoint import CheckpointFunction, checkpoint
from torch.utils.cpp_extension import BuildExtension
from torch.utils.hipify import hipify_python
from torch.utils._contextlib import _DecoratorContextManager
from torch_npu.testing.testcase import TestCase, run_tests


def npu_available():
    return hasattr(torch, "npu") and torch.npu.is_available()


class TestTorchUtilsCheckpointFunctionBackward(TestCase):

    def test_checkpoint_function_backward_exists(self):
        self.assertTrue(hasattr(CheckpointFunction, "backward"))
        self.assertTrue(callable(CheckpointFunction.backward))

    def test_checkpoint_backward_matches_normal_backward_on_npu(self):
        if not npu_available():
            self.skipTest("NPU is not available")

        def run_fn(x):
            return torch.sin(x) * torch.cos(x) + x * x

        x = torch.randn(4, 4, device="npu", dtype=torch.float32, requires_grad=True)
        x_ref = x.detach().clone().requires_grad_(True)

        checkpoint_out = checkpoint(run_fn, x, use_reentrant=True)
        normal_out = run_fn(x_ref)

        checkpoint_out.sum().backward()
        normal_out.sum().backward()

        self.assertEqual(checkpoint_out.device.type, "npu")
        self.assertIsNotNone(x.grad)
        self.assertEqual(x.grad.device.type, "npu")
        self.assertRtolEqual(checkpoint_out.detach().cpu(), normal_out.detach().cpu())
        self.assertRtolEqual(x.grad.cpu(), x_ref.grad.cpu())

    def test_checkpoint_backward_with_non_tensor_argument_on_npu(self):
        if not npu_available():
            self.skipTest("NPU is not available")

        def run_fn(x, scale):
            return x * scale + x.relu()

        x = torch.randn(3, 3, device="npu", dtype=torch.float32, requires_grad=True)
        x_ref = x.detach().clone().requires_grad_(True)
        scale = 2.5

        checkpoint_out = checkpoint(run_fn, x, scale, use_reentrant=True)
        normal_out = run_fn(x_ref, scale)

        checkpoint_out.sum().backward()
        normal_out.sum().backward()

        self.assertRtolEqual(checkpoint_out.detach().cpu(), normal_out.detach().cpu())
        self.assertRtolEqual(x.grad.cpu(), x_ref.grad.cpu())

    def test_checkpoint_backward_with_multiple_outputs_on_npu(self):
        if not npu_available():
            self.skipTest("NPU is not available")

        def run_fn(x):
            return x * 2.0, x * x

        x = torch.randn(2, 3, device="npu", dtype=torch.float32, requires_grad=True)
        x_ref = x.detach().clone().requires_grad_(True)

        out1, out2 = checkpoint(run_fn, x, use_reentrant=True)
        ref1, ref2 = run_fn(x_ref)

        loss = out1.sum() + out2.sum()
        ref_loss = ref1.sum() + ref2.sum()

        loss.backward()
        ref_loss.backward()

        self.assertRtolEqual(out1.detach().cpu(), ref1.detach().cpu())
        self.assertRtolEqual(out2.detach().cpu(), ref2.detach().cpu())
        self.assertRtolEqual(x.grad.cpu(), x_ref.grad.cpu())

    def test_checkpoint_backward_rng_state_with_dropout_on_npu(self):
        if not npu_available():
            self.skipTest("NPU is not available")
        if not hasattr(torch.npu, "get_rng_state") or not hasattr(torch.npu, "set_rng_state"):
            self.skipTest("NPU RNG state APIs are not available")

        torch.manual_seed(1234)
        torch.npu.manual_seed(1234)

        dropout = torch.nn.Dropout(p=0.5)

        def run_fn(x):
            return dropout(x)

        x = torch.randn(128, device="npu", dtype=torch.float32, requires_grad=True)

        cpu_state = torch.get_rng_state()
        npu_state = torch.npu.get_rng_state()

        checkpoint_out = checkpoint(run_fn, x, use_reentrant=True)
        checkpoint_out.sum().backward()
        checkpoint_grad = x.grad.detach().clone()

        torch.set_rng_state(cpu_state)
        torch.npu.set_rng_state(npu_state)
        x_ref = x.detach().clone().requires_grad_(True)

        normal_out = run_fn(x_ref)
        normal_out.sum().backward()

        self.assertRtolEqual(checkpoint_grad.cpu(), x_ref.grad.cpu())


class TestTorchUtilsBuildExtensionWithOptions(TestCase):

    def test_build_extension_with_options_returns_subclass(self):
        custom_build_extension = BuildExtension.with_options(use_ninja=False)

        self.assertTrue(issubclass(custom_build_extension, BuildExtension))
        self.assertIsNot(custom_build_extension, BuildExtension)

    def test_build_extension_with_options_injects_options(self):
        custom_build_extension = BuildExtension.with_options(
            use_ninja=False,
            no_python_abi_suffix=True,
        )

        dist = Distribution({"name": "test_extension"})
        cmd = custom_build_extension(dist)

        self.assertFalse(cmd.use_ninja)
        self.assertTrue(cmd.no_python_abi_suffix)

    def test_build_extension_with_options_overrides_constructor_kwargs(self):
        custom_build_extension = BuildExtension.with_options(use_ninja=False)

        dist = Distribution({"name": "test_extension"})
        cmd = custom_build_extension(dist, use_ninja=True)

        self.assertFalse(cmd.use_ninja)

    def test_build_extension_with_options_multiple_classes_are_independent(self):
        no_ninja_build = BuildExtension.with_options(use_ninja=False)
        ninja_build = BuildExtension.with_options(use_ninja=True)

        dist1 = Distribution({"name": "test_extension_1"})
        dist2 = Distribution({"name": "test_extension_2"})

        cmd1 = no_ninja_build(dist1)
        cmd2 = ninja_build(dist2)

        self.assertFalse(cmd1.use_ninja)
        self.assertIsInstance(cmd2.use_ninja, bool)


class TestTorchUtilsHipifyPythonHipify(TestCase):

    def test_import_hipify_python_and_hipify_exists(self):
        self.assertTrue(hasattr(hipify_python, "hipify"))
        self.assertTrue(callable(hipify_python.hipify))

    def test_hipify_python_hipify_cuda_file(self):
        with tempfile.TemporaryDirectory() as project_dir:
            output_dir = os.path.join(project_dir, "hip_output")
            os.makedirs(output_dir, exist_ok=True)

            cuda_file = os.path.join(project_dir, "kernel.cu")
            with open(cuda_file, "w", encoding="utf-8") as f:
                f.write(
                    textwrap.dedent(
                        """
                        #include <cuda_runtime.h>

                        __global__ void add_kernel(float* x) {
                            int i = threadIdx.x;
                            x[i] += 1.0f;
                        }
                        """
                    )
                )

            result = hipify_python.hipify(
                project_directory=project_dir,
                output_directory=output_dir,
                extra_files=(cuda_file,),
                hipify_extra_files_only=True,
                show_progress=False,
            )

            self.assertIsInstance(result, dict)
            self.assertTrue(len(result) > 0)
            self.assertTrue(
                any("kernel" in os.path.basename(path) for path in result.keys())
            )

    def test_hipify_python_hipify_extra_file_only(self):
        with tempfile.TemporaryDirectory() as project_dir:
            cuda_file = os.path.join(project_dir, "extra_kernel.cu")
            with open(cuda_file, "w", encoding="utf-8") as f:
                f.write(
                    textwrap.dedent(
                        """
                        #include <cuda_runtime.h>

                        __global__ void extra_kernel(float* x) {
                            x[threadIdx.x] = 1.0f;
                        }
                        """
                    )
                )

            result = hipify_python.hipify(
                project_directory=project_dir,
                output_directory=project_dir,
                extra_files=(cuda_file,),
                hipify_extra_files_only=True,
                show_progress=False,
            )

            self.assertIsInstance(result, dict)
            self.assertTrue(
                any("extra_kernel" in os.path.basename(path) for path in result.keys())
            )

    def test_hipify_python_hipify_ignores_file(self):
        with tempfile.TemporaryDirectory() as project_dir:
            ignored_file = os.path.join(project_dir, "ignored_kernel.cu")
            with open(ignored_file, "w", encoding="utf-8") as f:
                f.write(
                    textwrap.dedent(
                        """
                        #include <cuda_runtime.h>

                        __global__ void ignored_kernel(float* x) {
                            x[threadIdx.x] = 1.0f;
                        }
                        """
                    )
                )

            result = hipify_python.hipify(
                project_directory=project_dir,
                output_directory=project_dir,
                includes=("*",),
                ignores=("*ignored_kernel.cu",),
                show_progress=False,
            )

            self.assertIsInstance(result, dict)
            self.assertFalse(
                any("ignored_kernel" in os.path.basename(path) for path in result.keys())
            )


class TestTorchUtilsDecoratorContextManager(TestCase):

    def test_decorator_context_manager_as_function_decorator(self):
        log = []

        class SampleContext(_DecoratorContextManager):
            def __enter__(self):
                log.append("enter")

            def __exit__(self, exc_type, exc_value, traceback):
                log.append("exit")

            def clone(self):
                return SampleContext()

        ctx = SampleContext()
        device = "npu" if npu_available() else "cpu"

        @ctx
        def decorated_func(x):
            log.append("body")
            return x + 1

        x = torch.ones(2, 2, device=device)
        y = decorated_func(x)

        self.assertEqual(log, ["enter", "body", "exit"])
        self.assertEqual(y.device.type, torch.device(device).type)
        self.assertRtolEqual(y.cpu(), torch.ones(2, 2) + 1)

    def test_decorator_context_manager_exits_on_exception(self):
        log = []

        class SampleContext(_DecoratorContextManager):
            def __enter__(self):
                log.append("enter")

            def __exit__(self, exc_type, exc_value, traceback):
                log.append("exit")

            def clone(self):
                return SampleContext()

        ctx = SampleContext()

        @ctx
        def decorated_func():
            log.append("body")
            raise RuntimeError("expected error")

        with self.assertRaisesRegex(RuntimeError, "expected error"):
            decorated_func()

        self.assertEqual(log, ["enter", "body", "exit"])

    def test_decorator_context_manager_as_generator_decorator(self):
        log = []

        class SampleContext(_DecoratorContextManager):
            def __enter__(self):
                log.append("enter")

            def __exit__(self, exc_type, exc_value, traceback):
                log.append("exit")

            def clone(self):
                return SampleContext()

        ctx = SampleContext()
        device = "npu" if npu_available() else "cpu"

        @ctx
        def decorated_generator():
            log.append("yield_1")
            yield torch.ones(1, device=device)
            log.append("yield_2")
            yield torch.ones(1, device=device) + 1

        outputs = list(decorated_generator())

        yield_events = [item for item in log if item.startswith("yield")]

        self.assertEqual(yield_events, ["yield_1", "yield_2"])
        self.assertEqual(log.count("enter"), log.count("exit"))
        self.assertGreaterEqual(log.count("enter"), 1)

        self.assertEqual(outputs[0].device.type, torch.device(device).type)
        self.assertEqual(outputs[1].device.type, torch.device(device).type)
        self.assertRtolEqual(outputs[0].cpu(), torch.ones(1))
        self.assertRtolEqual(outputs[1].cpu(), torch.ones(1) + 1)

    def test_torch_no_grad_decorator_behavior_on_npu(self):
        if not npu_available():
            self.skipTest("NPU is not available")

        @torch.no_grad()
        def decorated_func(x):
            return x * 2

        x = torch.randn(2, 2, device="npu", requires_grad=True)
        y = decorated_func(x)

        self.assertEqual(y.device.type, "npu")
        self.assertFalse(y.requires_grad)


if __name__ == "__main__":
    run_tests()
