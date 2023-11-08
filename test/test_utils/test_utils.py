# Copyright 2021 Huawei Technologies Co., Ltd
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
import unittest
import random

import torch
import torch.nn as nn
from torch.testing._internal.common_utils import TestCase, run_tests
from torch.utils.checkpoint import checkpoint, checkpoint_sequential

import torch_npu


class TestCheckpoint(TestCase):

    # This runs checkpoint_sequential on each of the nets in
    # module_lists_to_compare, and compares them against the uncheckpointed model.
    # To compare, it checks outputs as well as input gradients and parameter gradients
    def _check_checkpoint_sequential(
        self,
        model,
        module_lists_to_compare,
        num_chunks,
        input_1,
    ):

        # not checkpointed
        out = model(input_1)
        out_not_checkpointed = out.detach().clone()
        model.zero_grad()
        out.sum().backward()
        grad_not_checkpointed = {
            name: param.grad.detach().clone()
            for name, param in model.named_parameters()
        }
        input_grad_not_checkpointed = input_1.grad.detach().clone()
        for model_to_compare in module_lists_to_compare:
            # checkpointed model by passing list of modules
            detached = input_1.detach()
            detached.requires_grad = True

            # pass list of modules to checkpoint
            out = checkpoint_sequential(model_to_compare, num_chunks, detached)
            out_checkpointed = out.detach().clone()
            model.zero_grad()
            out.sum().backward()
            grad_checkpointed = {
                name: param.grad.detach().clone()
                for name, param in model.named_parameters()
            }
            input_grad_checkpointed = detached.grad.detach().clone()
            # compare outputs as well as the gradients of input and parameters
            self.assertEqual(out_checkpointed, out_not_checkpointed)
            self.assertEqual(input_grad_not_checkpointed, input_grad_checkpointed)
            for name in grad_checkpointed:
                self.assertEqual(grad_checkpointed[name], grad_not_checkpointed[name])

    # Test whether checkpoint is being triggered or not. For this, we check
    # the number of times forward pass happens
    def test_checkpoint_trigger(self):

        class Net(nn.Module):

            def __init__(self):
                super(Net, self).__init__()
                self.counter = 0

            def forward(self, input_var):
                self.counter += 1
                return input_var

        # checkpointed
        modules = [Net() for _ in range(10)]
        for m in modules:
            self.assertEqual(m.counter, 0)
        input_var = torch.randn(3, 4, requires_grad=True)
        out = checkpoint_sequential(modules, 2, input_var)
        for m in modules:
            self.assertEqual(m.counter, 1)
        out.sum().backward()
        for m in modules[:(len(modules) // 2)]:
            self.assertEqual(m.counter, 2)
        for m in modules[(len(modules) // 2):]:
            self.assertEqual(m.counter, 1)

    def test_checkpoint_valid(self):
        model = nn.Sequential(
            nn.Linear(100, 50),
            nn.ReLU(),
            nn.Linear(50, 20),
            nn.ReLU(),
            nn.Linear(20, 5),
            nn.ReLU()
        )

        input_var = torch.randn(1, 100, requires_grad=True)

        # checkpointed
        chunks = 2
        modules = list(model.children())
        out = checkpoint_sequential(modules, chunks, input_var)
        with self.assertRaisesRegex(RuntimeError, "Checkpointing is not compatible"):
            torch.autograd.grad(
                outputs=[out], grad_outputs=[torch.ones(1, 5)], inputs=[input_var], create_graph=True
            )

    def test_checkpoint(self):
        model = nn.Sequential(
            nn.Linear(100, 50),
            nn.ReLU(),
            nn.Linear(50, 20),
            nn.ReLU(),
            nn.Linear(20, 5),
            nn.ReLU()
        )

        # Compare uncheckpointed model with its checkpointed counterparts
        # In addition to running checkpoint_sequential on the nn.Sequential
        # instance, we also run the function on the list of functions within
        # the module.
        self._check_checkpoint_sequential(
            model,
            [list(model.children()), model],
            2,
            torch.randn(1, 100, requires_grad=True)
        )

    def test_checkpoint_module_list(self):
        class ModuleListNet(nn.Module):
            def __init__(self):
                super(ModuleListNet, self).__init__()
                module_list = [
                    nn.Linear(100, 50),
                    nn.ReLU(),
                    nn.Linear(50, 20),
                    nn.ReLU(),
                    nn.Linear(20, 5),
                    nn.ReLU(),
                ]
                self.module_list = nn.ModuleList(module_list)

            def forward(self, input_1):
                for layer in self.module_list:
                    input_1 = layer(input_1)
                return input_1

        model = ModuleListNet()

        # Compare uncheckpointed model with its checkpointed counterparts.
        self._check_checkpoint_sequential(
            model,
            [list(model.module_list.children()), model.module_list],
            2,
            torch.randn(1, 100, requires_grad=True),
        )

    def test_checkpoint_sequential_deprecated_multiple_args(self):
        class Two(nn.Module):
            def forward(self, a, b):
                return a, b

        model = nn.Sequential(Two())
        a = torch.randn(1, 100, requires_grad=True)
        b = torch.randn(1, 100, requires_grad=True)

        with self.assertRaises(TypeError):
            checkpoint_sequential(model, 1, a, b)  # type: ignore[call-arg]

    def test_checkpoint_sequential_deprecated_no_args(self):
        class Noop(nn.Module):
            def forward(self):
                pass

        model = nn.Sequential(Noop())

        with self.assertRaises(TypeError):
            checkpoint_sequential(model, 1)  # type: ignore[call-arg]

    def test_checkpoint_rng_cpu(self):
        for _ in range(5):
            inp = torch.randn(20000, device='cpu').requires_grad_()
            phase1 = torch.nn.Dropout()
            phase2 = torch.nn.Dropout()

            def run_fn(input_1):
                return phase2(input_1)

            state = torch.get_rng_state()

            out = phase1(inp)
            out = checkpoint(run_fn, out)
            out.sum().backward()
            grad_with_checkpointing = inp.grad

            torch.set_rng_state(state)

            inp.grad = None

            out = phase1(inp)
            out = run_fn(out)
            out.sum().backward()
            grad_no_checkpointing = inp.grad

            self.assertEqual(grad_with_checkpointing, grad_no_checkpointing)
    
    @unittest.skip("skip test_checkpoint_rng_npu now")
    def test_checkpoint_rng_npu(self):
        for _ in range(5):
            inp = torch.randn(20000, device='npu').requires_grad_()
            phase1 = torch.nn.Dropout()
            phase2 = torch.nn.Dropout()

            def run_fn(input_1):
                return phase2(input_1)

            state = torch.npu.get_rng_state()

            out = phase1(inp)
            out = checkpoint(run_fn, out)
            out.sum().backward()
            grad_with_checkpointing = inp.grad

            torch.npu.set_rng_state(state)

            inp.grad = None

            out = phase1(inp)
            out = run_fn(out)
            out.sum().backward()
            grad_no_checkpointing = inp.grad

            self.assertEqual(grad_with_checkpointing, grad_no_checkpointing)

    def test_checkpoint_non_tensor(self):

        def run_fn(tensor1, tensor2):
            if tensor2 is None:
                return tensor1
            return tensor1 + tensor2

        input_var = torch.randn(1, 100, requires_grad=True)
        out = checkpoint(run_fn, input_var, None)
        out.sum().backward()

    def test_checkpoint_partial_grad(self):
        def run_fn(tensor1, tensor2):
            # tensor 2 is used for other application logic
            return tensor1, tensor2
        input_var = torch.randn(1, 4, requires_grad=True)
        input_var2 = torch.randn(1, 4, requires_grad=False)
        out = checkpoint(run_fn, input_var, input_var2)
        out[0].sum().backward()

        def run_fn2(tensor1, tensor2):
            return tensor1
        input_var = torch.randn(1, 4, requires_grad=False)
        input_var2 = torch.randn(1, 4, requires_grad=True)
        with self.assertRaisesRegex(
            RuntimeError,
            r"none of output has requires_grad=True, this checkpoint\(\) is not necessary"
        ):
            out = checkpoint(run_fn2, input_var, input_var2)
            out.sum().backward()


if __name__ == '__main__':
    run_tests()
