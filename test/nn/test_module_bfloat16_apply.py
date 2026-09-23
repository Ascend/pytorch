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
Add validation cases for torch.nn APIs on NPU:
1. PyTorch community lacks direct test cases for torch.nn.Module.bfloat16 and
   torch.nn.Module.apply, so this file is added.
2. This file validates torch.nn.Module.bfloat16, torch.nn.Module.apply (extendable).
"""

import torch
import torch.nn as nn
from torch.testing._internal.common_utils import TestCase, run_tests

device_type = acc.type if (acc := torch.accelerator.current_accelerator()) else "cpu"


class TestModuleBFloat16(TestCase):
    """Test torch.nn.Module.bfloat16 method."""

    def _make_module(self):
        module = nn.Linear(4, 4)
        module.register_parameter(
            "param_fp64", nn.Parameter(torch.randn(3, 3, dtype=torch.float64)))
        module.register_buffer("buffer_fp32", torch.randn(4))
        module.register_buffer("buffer_int64", torch.arange(4, dtype=torch.int64))
        return module

    def test_bfloat16_casts_all_floating_point_params_and_buffers(self):
        """Verify bfloat16 casts all floating point params and buffers, including nested ones."""
        module = nn.Sequential(self._make_module(), self._make_module()).to(device_type)
        module.bfloat16()

        for submodule in module:
            self.assertEqual(submodule.weight.dtype, torch.bfloat16)
            self.assertEqual(submodule.bias.dtype, torch.bfloat16)
            self.assertEqual(submodule.param_fp64.dtype, torch.bfloat16)
            self.assertEqual(submodule.buffer_fp32.dtype, torch.bfloat16)

    def test_bfloat16_keeps_non_floating_point_tensors(self):
        """Verify bfloat16 keeps non floating point tensors unchanged."""
        module = self._make_module().to(device_type)
        module.bfloat16()

        self.assertEqual(module.buffer_int64.dtype, torch.int64)
        self.assertEqual(module.buffer_int64.tolist(), [0, 1, 2, 3])

    def test_bfloat16_returns_self_and_keeps_device(self):
        """Verify bfloat16 modifies the module in place, returns self and keeps device."""
        module = self._make_module().to(device_type)
        result = module.bfloat16()

        self.assertIs(result, module)
        for tensor in list(module.parameters()) + list(module.buffers()):
            self.assertEqual(tensor.device.type, device_type)

    def test_bfloat16_module_without_floating_point_tensors(self):
        """Verify bfloat16 works on a module without floating point tensors."""
        module = nn.Module()
        module.register_buffer("step", torch.tensor(3, dtype=torch.int64))
        module.to(device_type)
        result = module.bfloat16()

        self.assertIs(result, module)
        self.assertEqual(module.step.dtype, torch.int64)

    def test_bfloat16_matches_cpu_result(self):
        """Verify bfloat16 cast result on NPU matches the CPU side."""
        torch.manual_seed(1234)
        module_cpu = self._make_module()

        torch.manual_seed(1234)
        module_npu = self._make_module().to(device_type)

        module_cpu.bfloat16()
        module_npu.bfloat16()

        self.assertEqual(module_cpu.weight, module_npu.weight.to("cpu"))
        self.assertEqual(module_cpu.param_fp64, module_npu.param_fp64.to("cpu"))
        self.assertEqual(module_cpu.buffer_fp32, module_npu.buffer_fp32.to("cpu"))


class TestModuleApply(TestCase):
    """Test torch.nn.Module.apply method."""

    def _make_tree(self):
        root = nn.Module()
        root.linear = nn.Linear(4, 4)
        root.seq = nn.Sequential(nn.Linear(4, 4), nn.ReLU())
        return root

    def test_apply_calls_fn_on_self_and_all_submodules(self):
        """Verify apply calls fn on self and every submodule exactly once."""
        root = self._make_tree().to(device_type)
        visited = []
        root.apply(visited.append)

        expected = {root, root.linear, root.seq, root.seq[0], root.seq[1]}
        self.assertEqual(set(visited), expected)
        self.assertEqual(len(visited), len(expected))

    def test_apply_visits_submodules_before_parent(self):
        """Verify apply visits submodules recursively before their parent."""
        root = self._make_tree().to(device_type)
        order = []
        root.apply(lambda module: order.append(type(module).__name__))

        self.assertEqual(order, ["Linear", "Linear", "ReLU", "Sequential", "Module"])

    def test_apply_returns_self(self):
        """Verify apply returns self."""
        root = self._make_tree().to(device_type)

        self.assertIs(root.apply(lambda module: None), root)

    def test_apply_moves_module_to_device(self):
        """Verify apply can be used to move all submodules to the device."""
        root = self._make_tree()
        root.apply(lambda module: module.to(device_type))

        for param in root.parameters():
            self.assertEqual(param.device.type, device_type)

    def test_apply_raises_type_error_for_non_callable_fn(self):
        """Verify apply raises TypeError when fn is not callable."""
        root = self._make_tree().to(device_type)

        with self.assertRaises(TypeError):
            root.apply(None)


if __name__ == "__main__":
    run_tests()
