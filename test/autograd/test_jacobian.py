# Copyright (c) 2026 Huawei Technologies Co., Ltd.
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
Add validation cases for torch.autograd.functional.jacobian on NPU.

1. PyTorch community has jacobian cases in test/autograd/test_functional.py, but
   they build CPU tensors, so the NPU execution path is not asserted directly.
2. This file validates torch.autograd.functional.jacobian on NPU for the
   create_graph, strict, vectorize and strategy arguments, for single/multiple
   inputs and outputs, and for the error paths (extendable).
"""

import torch
import torch.autograd.functional as autogradF
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests

device_type = acc.type if (acc := torch.accelerator.current_accelerator()) else "cpu"


def _to_cpu(value):
    if isinstance(value, torch.Tensor):
        return value.cpu()
    return tuple(_to_cpu(item) for item in value)


class TestJacobian(TestCase):
    def test_jacobian_output(self):
        def exp_reducer(x):
            return x.exp().sum(dim=1)

        def identity(x):
            return x.clone()

        for vectorize in (False, True):
            inputs_cpu = torch.rand(4, 4)
            expected = autogradF.jacobian(exp_reducer, inputs_cpu, vectorize=vectorize)
            res = autogradF.jacobian(exp_reducer, inputs_cpu.to(device_type), vectorize=vectorize)
            self.assertEqual(res.device.type, device_type)
            self.assertEqual(res.cpu(), expected)

            inputs_cpu = torch.rand(4)
            expected = autogradF.jacobian(identity, inputs_cpu, vectorize=vectorize)
            res = autogradF.jacobian(identity, inputs_cpu.to(device_type), vectorize=vectorize)
            self.assertEqual(res.cpu(), expected)

    def test_jacobian_multi_input(self):
        def f(x, y):
            return (x.cos() * x) @ y.sin()

        x_cpu = torch.randn(2, 3)
        y_cpu = torch.randn(3, 5)
        for vectorize in (False, True):
            expected = autogradF.jacobian(f, (x_cpu, y_cpu), vectorize=vectorize)
            res = autogradF.jacobian(
                f, (x_cpu.to(device_type), y_cpu.to(device_type)), vectorize=vectorize
            )
            self.assertIsInstance(res, tuple)
            self.assertEqual(len(res), 2)
            for res_i, expected_i in zip(res, expected):
                self.assertEqual(res_i.cpu(), expected_i)

    def test_jacobian_multi_output(self):
        def f(x):
            return x.sum(), x * x

        x_cpu = torch.randn(3, 3)
        for vectorize in (False, True):
            expected = autogradF.jacobian(f, x_cpu, vectorize=vectorize)
            res = autogradF.jacobian(f, x_cpu.to(device_type), vectorize=vectorize)
            self.assertIsInstance(res, tuple)
            self.assertEqual(len(res), 2)
            for res_i, expected_i in zip(res, expected):
                self.assertEqual(res_i.cpu(), expected_i)

    def test_jacobian_scalar_and_zero_dim(self):
        def reducer(x):
            return x.sum()

        def expander(x):
            return x.unsqueeze(0).repeat(4)

        inputs_cpu = torch.randn(4, 4)
        res = autogradF.jacobian(reducer, inputs_cpu.to(device_type))
        self.assertEqual(res.cpu(), autogradF.jacobian(reducer, inputs_cpu))

        inputs_cpu = torch.randn([])
        res = autogradF.jacobian(expander, inputs_cpu.to(device_type))
        self.assertEqual(res.cpu(), autogradF.jacobian(expander, inputs_cpu))

    def test_jacobian_create_graph(self):
        def exp_reducer(x):
            return x.exp().sum(dim=1)

        inputs_cpu = torch.rand(4, 4, dtype=torch.double, requires_grad=True)
        for vectorize in (False, True):
            expected = autogradF.jacobian(
                exp_reducer, inputs_cpu, create_graph=True, vectorize=vectorize
            )
            res = autogradF.jacobian(
                exp_reducer,
                inputs_cpu.to(device_type),
                create_graph=True,
                vectorize=vectorize,
            )
            self.assertIsNotNone(res.grad_fn)
            self.assertEqual(res.cpu(), expected)

    def test_jacobian_strict(self):
        def detached(x):
            return x.detach()

        def disconnected(x):
            # Non-leaf tensor that requires grad but is not connected to the input.
            return x.long().float().requires_grad_().clone()

        inputs_cpu = torch.rand(4)
        with self.assertRaisesRegex(RuntimeError, "does not require gradients"):
            autogradF.jacobian(detached, inputs_cpu.to(device_type), strict=True)
        res = autogradF.jacobian(detached, inputs_cpu.to(device_type), strict=False)
        self.assertEqual(res.cpu(), autogradF.jacobian(detached, inputs_cpu, strict=False))

        with self.assertRaisesRegex(RuntimeError, "independent of input 0"):
            autogradF.jacobian(disconnected, inputs_cpu.to(device_type), strict=True)
        res = autogradF.jacobian(disconnected, inputs_cpu.to(device_type), strict=False)
        self.assertEqual(
            res.cpu(), autogradF.jacobian(disconnected, inputs_cpu, strict=False)
        )

    def test_jacobian_strategy_forward_mode(self):
        def f(x):
            return 3 * x ** 2

        inputs_cpu = torch.randn(2, 3, 5)
        expected = autogradF.jacobian(f, inputs_cpu, vectorize=False)
        res = autogradF.jacobian(
            f, inputs_cpu.to(device_type), strategy="forward-mode", vectorize=True
        )
        self.assertEqual(res.cpu(), expected)

        with self.assertRaisesRegex(NotImplementedError, "not supported together"):
            autogradF.jacobian(
                f,
                inputs_cpu.to(device_type),
                create_graph=True,
                strategy="forward-mode",
                vectorize=True,
            )

    def test_jacobian_err_check(self):
        def foo(a):
            return 3 * a.narrow(0, 0, 3)

        def bar(a):
            return 3 * a.narrow(0, 0, 3), "bar"

        inp_cpu = torch.rand(4)
        inp = inp_cpu.to(device_type)
        with self.assertRaisesRegex(TypeError, "must be either a Tensor"):
            autogradF.jacobian(foo, (inp, 2))
        with self.assertRaisesRegex(TypeError, "outputs of the user-provided function"):
            autogradF.jacobian(bar, inp)
        with self.assertRaisesRegex(RuntimeError, "not supported together"):
            autogradF.jacobian(foo, inp, strict=True, vectorize=True)


if __name__ == "__main__":
    print('torch_npu version is ', torch_npu.__version__)
    print('torch version is', torch.__version__)
    print('device_type is', device_type)
    run_tests()