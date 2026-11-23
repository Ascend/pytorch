# Copyright (c) 2026 Huawei Technologies Co., Ltd
# All rights reserved.
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
Validate torch.sym_float on NPU.

torch.sym_float is a pure Python, device-independent cast defined in
torch/__init__.py. Upstream PyTorch does not ship a direct, complete test
module for it. This file checks ordinary numbers, SymInt to SymFloat
conversion, SymFloat identity, custom __sym_float__ objects,
__torch_function__ dispatch, invalid inputs, NPU tensor sizes, dynamic
compile reuse, and CPU/NPU agreement.
"""

import torch
from torch._dynamo.source import LocalSource, TensorProperty, TensorPropertySource
from torch._dynamo.testing import CompileCounter
from torch.fx.experimental.symbolic_shapes import DimDynamic, ShapeEnv
from torch.testing._internal.common_utils import TestCase, run_tests


device_type = acc.type if (acc := torch.accelerator.current_accelerator()) else "cpu"


class TestSymFloat(TestCase):
    def _backed_symint(self, hint):
        # ShapeEnv is the supported way to build a SymInt with a concrete hint.
        # The same constructors exist from PyTorch 2.7.1 through current main.
        env = ShapeEnv()
        source = TensorPropertySource(LocalSource("x"), TensorProperty.SIZE, 0)
        expr = env.create_symbol(
            int(hint),
            source=source,
            dynamic_dim=DimDynamic.DYNAMIC,
        )
        return env.create_symintnode(expr, hint=int(hint), source=source)

    def test_plain_int(self):
        self.assertIsInstance(torch.sym_float(7), float)
        self.assertEqual(torch.sym_float(0), 0.0)
        self.assertEqual(torch.sym_float(7), 7.0)
        self.assertEqual(torch.sym_float(-3), -3.0)

    def test_plain_float(self):
        self.assertIsInstance(torch.sym_float(2.5), float)
        self.assertEqual(torch.sym_float(0.0), 0.0)
        self.assertEqual(torch.sym_float(2.5), 2.5)
        self.assertEqual(torch.sym_float(-1.25), -1.25)

    def test_symint_to_symfloat(self):
        hint = 6
        symint = self._backed_symint(hint)
        converted = torch.sym_float(symint)

        self.assertIsInstance(symint, torch.SymInt)
        self.assertIsInstance(converted, torch.SymFloat)
        self.assertEqual(float(converted), float(hint))

        # An unbacked SymInt has no numeric hint; conversion must still
        # return a SymFloat and must not require a concrete value.
        unbacked = ShapeEnv().create_unbacked_symint()
        unbacked_float = torch.sym_float(unbacked)
        self.assertIsInstance(unbacked, torch.SymInt)
        self.assertIsInstance(unbacked_float, torch.SymFloat)

    def test_symfloat_returned_unchanged(self):
        backed = torch.sym_float(self._backed_symint(4))
        self.assertIsInstance(backed, torch.SymFloat)
        self.assertIs(torch.sym_float(backed), backed)

        unbacked = ShapeEnv().create_unbacked_symfloat()
        self.assertIsInstance(unbacked, torch.SymFloat)
        self.assertIs(torch.sym_float(unbacked), unbacked)

    def test_custom_sym_float(self):
        # The return value of __sym_float__ is used as-is.
        sentinel = object()

        class HasSymFloat:
            def __sym_float__(self):
                return sentinel

        self.assertIs(torch.sym_float(HasSymFloat()), sentinel)

    def test_torch_function_dispatch(self):
        # Objects implementing __torch_function__ take the override branch
        # before SymFloat identity and __sym_float__.
        sentinel = object()

        class WithTorchFunction:
            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                if func is torch.sym_float:
                    return sentinel
                return NotImplemented

        self.assertIs(torch.sym_float(WithTorchFunction()), sentinel)

    def test_invalid_input(self):
        with self.assertRaises(TypeError):
            torch.sym_float(None)
        with self.assertRaises(TypeError):
            torch.sym_float(object())
        with self.assertRaises(ValueError):
            torch.sym_float("not-a-number")

    def test_npu_tensor_shape(self):
        self.assertEqual(device_type, "npu")
        values = torch.ones(3, 5, dtype=torch.float32, device=device_type)
        indices = torch.zeros(2, 4, dtype=torch.int64, device=device_type)

        self.assertEqual(values.device.type, device_type)
        self.assertEqual(indices.device.type, device_type)
        # Sizes are ordinary ints outside a symbolic region. sym_float casts
        # those ints; it does not read the tensor dtype.
        self.assertEqual(torch.sym_float(values.shape[0]), 3.0)
        self.assertEqual(torch.sym_float(values.shape[1]), 5.0)
        self.assertEqual(torch.sym_float(values.numel()), 15.0)
        self.assertEqual(torch.sym_float(indices.shape[1]), 4.0)

        scaled = values * torch.sym_float(values.shape[1])
        self.assertEqual(scaled.device.type, device_type)
        self.assertEqual(scaled, torch.full((3, 5), 5.0, device=device_type))

    def test_compile_reuses_graph_for_two_npu_shapes(self):
        def fn(x):
            return x * torch.sym_float(x.shape[0])

        counter = CompileCounter()
        torch._dynamo.reset()
        compiled = torch.compile(fn, backend=counter, dynamic=True, fullgraph=True)

        # Both inputs are real NPU tensors. The dynamic dimension is the
        # tensor size itself, not a SymInt built from a concrete int.
        first = torch.ones(4, 3, device=device_type)
        second = torch.ones(7, 3, device=device_type)
        torch._dynamo.mark_dynamic(first, 0)
        torch._dynamo.mark_dynamic(second, 0)

        first_out = compiled(first)
        second_out = compiled(second)

        self.assertEqual(first_out.device.type, device_type)
        self.assertEqual(second_out.device.type, device_type)
        self.assertEqual(first_out, first * 4)
        self.assertEqual(second_out, second * 7)
        # One frame for two different sizes means the dynamic graph is reused.
        self.assertEqual(counter.frame_count, 1)

    def test_cpu_npu_agreement(self):
        # CPU is only the reference for this comparison. The NPU tensor is
        # the input that must follow the accelerator result.
        cpu = torch.empty(4, 3)
        npu = torch.empty(4, 3, device=device_type)

        self.assertEqual(npu.device.type, device_type)
        self.assertEqual(torch.sym_float(cpu.shape[0]), torch.sym_float(npu.shape[0]))
        self.assertEqual(torch.sym_float(cpu.shape[1]), torch.sym_float(npu.shape[1]))
        self.assertEqual(torch.sym_float(cpu.numel()), torch.sym_float(npu.numel()))
        self.assertEqual(torch.sym_float(npu.shape[0]), torch.sym_float(4))


if __name__ == "__main__":
    run_tests()
