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
Add validation cases for torch.autograd.function.FunctionCtx.set_materialize_grads on NPU:
1. The PyTorch community covers this flag only indirectly, so this file is added.
2. This file validates single-output and multi-output cases (extendable).
"""

import torch
from torch.testing._internal.common_utils import TestCase, run_tests

device_type = acc.type if (acc := torch.accelerator.current_accelerator()) else "cpu"


class TestSetMaterializeGrads(TestCase):

    def _check_single_output(self, flag, expect_materialized):
        class Copy(torch.autograd.Function):

            @staticmethod
            def forward(ctx, inp):
                if flag is not None:
                    ctx.set_materialize_grads(flag)
                return inp

            @staticmethod
            def backward(ctx, grad_out):
                if expect_materialized:
                    self.assertEqual(grad_out, torch.zeros_like(inp))
                else:
                    self.assertIsNone(grad_out)
                return grad_out

        inp = torch.arange(4.0, device=device_type, requires_grad=True)
        torch._C._functions.UndefinedGrad()(Copy.apply(inp)).sum().backward()

    def test_flag_disabled(self):
        self._check_single_output(False, False)

    def test_flag_enabled(self):
        self._check_single_output(True, True)

    def test_flag_default(self):
        self._check_single_output(None, True)

    def test_partial_outputs(self):
        class Pair(torch.autograd.Function):

            @staticmethod
            def forward(lhs, rhs):
                return lhs, rhs

            @staticmethod
            def setup_context(ctx, inputs, output):
                ctx.set_materialize_grads(False)

            @staticmethod
            def backward(ctx, grad_lhs, grad_rhs):
                self.assertIsNotNone(grad_lhs)
                self.assertIsNone(grad_rhs)
                return grad_lhs, grad_rhs

        def first_squared(rhs, lhs):
            lhs_out, _ = Pair.apply(lhs, rhs)
            return lhs_out * lhs_out

        lhs = torch.tensor(2.5, device=device_type)
        rhs = torch.tensor(4.5, device=device_type)
        torch.func.grad(first_squared)(rhs, lhs)
        torch.func.grad(torch.func.grad(first_squared))(rhs, lhs)

    def test_with_compiled_function(self):
        class Scale(torch.autograd.Function):

            @staticmethod
            def forward(ctx, inp):
                ctx.set_materialize_grads(True)
                return inp * 4

            @staticmethod
            def backward(ctx, grad_out):
                return grad_out * 4

        @torch.compile(backend="eager", fullgraph=True)
        def compiled(inp):
            return Scale.apply(inp)

        inp = torch.tensor(2.0, device=device_type, requires_grad=True)
        out = compiled(inp)
        out.backward()
        self.assertEqual(out, Scale.apply(inp))
        self.assertEqual(inp.grad, torch.tensor(4.0, device=device_type))


if __name__ == "__main__":
    run_tests()
