# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# Licensed under the BSD 3-Clause License (the "License");
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
Minimal direct-API validation for the two AOT joint-with-descriptors entrypoints on NPU:
1. torch._functorch.aot_autograd.aot_export_joint_with_descriptors
2. torch._functorch.aot_autograd.aot_compile_joint_with_descriptors

This file deliberately avoids reusing the upstream
test/functorch/test_aot_joint_with_descriptors.py scaffolding (named module
classes, assertExpectedInline FX graph text comparison, decomposition_table,
full backward correctness). Instead it asserts only the observable contract
of the API pair on NPU against an eager reference built from nn.Sequential:
the export call returns a JointWithDescriptors exposing graph_module and
_aot_state; the compile call returns a callable whose forward result matches
the eager module.

Invocation contract: the callable returned by aot_compile_joint_with_descriptors
flattens (params, inputs) into positional args via fx_pytree, so it must be
called as compiled(*params, *inputs), matching the upstream release/2.9+
test convention `parallel_model_fn(*dict(model.named_parameters()).values(), *inputs)`.
"""
from contextlib import ExitStack

import torch

from torch._functorch.aot_autograd import (
    aot_compile_joint_with_descriptors,
    aot_export_joint_with_descriptors,
)
from torch.testing._internal.common_utils import run_tests, TestCase

device_type = (
    acc.type if (acc := torch.accelerator.current_accelerator()) else "cpu"
)


def _build_export_target():
    """An eager nn.Sequential reference plus matching inputs on the active device."""
    model = torch.nn.Sequential(torch.nn.Linear(2, 1)).to(device_type)
    inputs = (torch.randn(3, 2, device=device_type),)
    return model, inputs


class TestAOTJointWithDescriptorsNPU(TestCase):
    def test_export_returns_joint_with_descriptors(self):
        """aot_export_joint_with_descriptors returns a JointWithDescriptors
        exposing graph_module and _aot_state on NPU.
        """
        model, inputs = _build_export_target()
        with ExitStack() as stack:
            exported = aot_export_joint_with_descriptors(stack, model, inputs)
        self.assertIsNotNone(exported)
        self.assertIsNotNone(exported.graph_module)
        self.assertIsNotNone(exported._aot_state)

    def test_export_preserves_npu_device(self):
        """aot_export_joint_with_descriptors leaves input tensors and
        module parameters on the NPU device.
        """
        model, inputs = _build_export_target()
        with ExitStack() as stack:
            aot_export_joint_with_descriptors(stack, model, inputs)
        self.assertEqual(inputs[0].device.type, device_type)
        for p in model.parameters():
            self.assertEqual(p.device.type, device_type)

    def test_compile_runs_and_matches_eager(self):
        """aot_compile_joint_with_descriptors runs end-to-end on NPU and
        the compiled forward matches the eager module output.
        """
        model, inputs = _build_export_target()
        with ExitStack() as stack:
            exported = aot_export_joint_with_descriptors(stack, model, inputs)
            compiled = aot_compile_joint_with_descriptors(exported)
        self.assertTrue(callable(compiled))
        expected = model(*inputs)
        actual = compiled(*dict(model.named_parameters()).values(), *inputs)
        self.assertEqual(actual.shape, expected.shape)
        self.assertEqual(actual.device, expected.device)
        self.assertEqual(actual, expected, rtol=1e-3, atol=1e-3)


if __name__ == "__main__":
    run_tests()
