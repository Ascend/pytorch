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

"""Regression cases for the ``Module.to()``/``Module.npu()`` weight format cast.

torch_npu used to reach its weight format casting through
``self.cast_weight(device)`` and ``sub_module.cast_weight(device)``.  Frameworks
such as ComfyUI define their own ``cast_weight(self, input)`` hook on
``nn.Module`` subclasses (see ``comfy/ops.py``), so ``Module.to(device)`` called
that hook with a ``torch.device`` and failed with
``AttributeError: 'torch.device' object has no attribute 'dtype'``.
"""

# Owner(s): ["module: nn"]
import torch
import torch.nn as nn

import torch_npu
import torch_npu.testing  # noqa: F401 - installs NPU test hooks
from torch_npu.utils._module import cast_weight
from torch.testing._internal.common_utils import TestCase, run_tests

FRACTAL_NZ_FAMILY = (
    torch_npu.Format.FRACTAL_NZ,
    torch_npu.Format.FRACTAL_NZ_C0_2,
    torch_npu.Format.FRACTAL_NZ_C0_4,
    torch_npu.Format.FRACTAL_NZ_C0_8,
    torch_npu.Format.FRACTAL_NZ_C0_16,
    torch_npu.Format.FRACTAL_NZ_C0_32,
)


class ComfyStyleOps(nn.Module):
    """Mimics ``comfy.ops`` classes, which define ``cast_weight(self, input)``."""

    def __init__(self, in_features=4, out_features=8):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.randn(out_features))
        self.cast_weight_inputs = []

    def cast_weight(self, input):
        # ComfyUI reads ``input.dtype`` here, a torch.device raises AttributeError.
        self.cast_weight_inputs.append(input)
        return input


class ComfyStyleBlock(nn.Module):
    """ComfyUI style wrapper holding nested ``ComfyStyleOps`` submodules."""

    def __init__(self):
        super().__init__()
        self.linear = ComfyStyleOps(4, 8)
        self.nested = nn.Sequential(ComfyStyleOps(8, 4))
        self.cast_weight_inputs = []

    def cast_weight(self, input):
        self.cast_weight_inputs.append(input)
        return input

    def overrides(self):
        return [
            self.cast_weight_inputs,
            self.linear.cast_weight_inputs,
            self.nested[0].cast_weight_inputs,
        ]


class TestModuleCastWeightDispatch(TestCase):
    def setUp(self):
        super().setUp()
        self.device = torch.device("npu", torch.npu.current_device())

    def _assert_no_override_called(self, block):
        for calls in block.overrides():
            self.assertEqual(calls, [], "framework cast_weight hook must not be dispatched into")

    def _assert_moved(self, block):
        for param in block.parameters():
            self.assertEqual(param.device, self.device)

    def test_to_does_not_call_override(self):
        """``Module.to(torch.device)`` must not dispatch into ``module.cast_weight``."""
        block = ComfyStyleBlock()
        block.to(self.device)

        self._assert_no_override_called(block)
        self._assert_moved(block)

    def test_to_string_device_does_not_call_override(self):
        """ComfyUI calls ``module.to(device)`` with a torch.device, keep str form working too."""
        block = ComfyStyleBlock()
        block.to(str(self.device))

        self._assert_no_override_called(block)
        self._assert_moved(block)

    def test_npu_does_not_call_override(self):
        """``Module.npu()`` must not dispatch into ``module.cast_weight``."""
        block = ComfyStyleBlock()
        block.npu()

        self._assert_no_override_called(block)
        self._assert_moved(block)

    def test_nested_submodule_override_is_not_called(self):
        """Root module without the hook: the recursion must not dispatch into children."""
        container = nn.Module()
        container.block = ComfyStyleBlock()

        container.to(self.device)

        self.assertEqual(container.block.cast_weight_inputs, [])
        self.assertEqual(container.block.linear.cast_weight_inputs, [])
        self.assertEqual(container.block.nested[0].cast_weight_inputs, [])
        for param in container.parameters():
            self.assertEqual(param.device, self.device)

    def test_cast_weight_is_not_injected_into_nn_module(self):
        """``cast_weight`` is a generic name owned by upstream frameworks, not by torch_npu."""
        self.assertNotIn("cast_weight", vars(torch.nn.Module))

    def test_format_cast_still_applied(self):
        """The NPU format cast itself must stay unchanged when the helper is called with a device."""
        linear = nn.Linear(4, 8)
        origin_nd = torch.npu.get_mm_bmm_format_nd()
        torch.npu.set_mm_bmm_format_nd(False)
        try:
            cast_weight(linear, self.device)
        finally:
            torch.npu.set_mm_bmm_format_nd(origin_nd)

        self.assertEqual(linear.weight.device, self.device)
        self.assertIn(torch_npu.get_npu_format(linear.weight), FRACTAL_NZ_FAMILY)

    def test_to_rejects_non_float_dtype(self):
        """The dtype guard of ``Module.to`` is untouched by this change."""
        with self.assertRaisesRegex(
            TypeError, "only accepts floating point or complex"
        ):
            nn.Linear(4, 8).to(self.device, torch.int32)

    def test_to_meta_tensor_hints_to_empty(self):
        """The meta tensor guidance of ``Module.to`` is untouched by this change."""
        class MetaModule(nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(torch.empty(1, 2, device="meta"))
                self.bias = nn.Parameter(torch.empty(3, device="meta"))

        with self.assertRaisesRegex(
            NotImplementedError,
            "Please use torch.nn.Module.to_empty\\(\\) instead of torch.nn.Module.to\\(\\)",
        ):
            MetaModule().to(self.device)


if __name__ == "__main__":
    run_tests()
