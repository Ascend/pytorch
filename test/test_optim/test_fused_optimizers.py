# Copyright (c) 2023, Huawei Technologies.All rights reserved.
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

from copy import deepcopy

import torch
from torch.optim import SGD, AdamW
from torch_npu.npu.amp import GradScaler, autocast
from torch_npu.optim import NpuFusedSGD, NpuFusedAdamW
from torch_npu.testing.testcase import TestCase, run_tests


class TestFusedOptim(TestCase):
    def __init__(self, method_name='runTest'):
        super(TestCase, self).__init__(method_name)

        self.optim_cases = [
            (SGD, NpuFusedSGD, dict(lr=0.01, momentum=0.9, weight_decay=0.001)),
            (AdamW, NpuFusedAdamW, dict(eps=1e-8, betas=(0.9, 0.999), lr=2e-3, weight_decay=0.05)),
        ]

        # Full tests for these optimizers will be run, including a small model.
        # Otherwise, only test the optimizer-related APIs.
        self.base_cases = [SGD, AdamW]

    def _create_optimizer_cases(self, all_cases=False):
        optim_cases = self.optim_cases
        if not all_cases:
            optim_cases = list(filter(lambda x: x[0] in self.base_cases, optim_cases))

        return optim_cases

    def _create_simple_model(self):
        model = torch.nn.Sequential(
            torch.nn.Conv2d(1, 8, kernel_size=3),
            torch.nn.BatchNorm2d(8, momentum=0.3),
            torch.nn.ReLU(),
            torch.nn.Linear(22, 12),
        )
        model.to("npu:0")
        return model

    def _create_simple_params_and_grads(self):
        params = [
            torch.rand(2,
                       3,
                       dtype=torch.float32,
                       device='npu:0',
                       requires_grad=True),
            torch.rand(4,
                       3,
                       dtype=torch.float32,
                       device='npu:0',
                       requires_grad=True),
            torch.rand(2,
                       3,
                       dtype=torch.float16,
                       device='npu:0',
                       requires_grad=True),
            torch.rand(2,
                       3,
                       dtype=torch.float16,
                       device='npu:0',
                       requires_grad=True),
            torch.rand(2,
                       3,
                       dtype=torch.float32,
                       device='npu:0',
                       requires_grad=True),
            torch.rand(5,
                       3,
                       dtype=torch.float32,
                       device='npu:0',
                       requires_grad=True),
            torch.rand(2,
                       3,
                       dtype=torch.float16,
                       device='npu:0',
                       requires_grad=True),
            torch.rand(6,
                       3,
                       dtype=torch.float16,
                       device='npu:0',
                       requires_grad=True),
            torch.randint(1024, (2, 3),
                          dtype=torch.float32,
                          device='npu:0',
                          requires_grad=False),
        ]

        for p in params:
            if p.requires_grad:
                p.grad = torch.rand_like(p, device=p.device, dtype=p.dtype)

        return params

    def _create_params_clone(self, params):
        params_clone = []
        for p in params:
            p_clone = p.clone().detach()
            if p.requires_grad:
                p_clone.requires_grad = True
                p_clone.grad = p.grad.clone().detach()
                params_clone.append(p_clone)
        return params_clone

    def test_zero_grad(self):
        optim_cases = self._create_optimizer_cases()
        for opt_obj, fused_opt_obj, opt_kwargs in optim_cases:
            params = self._create_simple_params_and_grads()
            params_clone = self._create_params_clone(params)
            with torch.no_grad():
                opt = opt_obj(params, **opt_kwargs)
                opt.zero_grad()

                fused_opt = fused_opt_obj(params_clone, **opt_kwargs)
                fused_opt.zero_grad()

            for p, p_clone in zip(params, params_clone):
                if p.grad is not None:
                    self.assertEqual(p.grad, p_clone.grad)
                    self.assertEqual(p.grad, torch.zeros_like(p.grad))

    def test_step(self):
        optim_cases = self._create_optimizer_cases(all_cases=True)
        num_iters = 10
        for opt_obj, fused_opt_obj, opt_kwargs in optim_cases:
            params = self._create_simple_params_and_grads()
            params_clone = self._create_params_clone(params)
            opt = opt_obj(params, **opt_kwargs)
            fused_opt = fused_opt_obj(params_clone, **opt_kwargs)
            with torch.no_grad():
                for _ in range(num_iters):
                    opt.step()
                    fused_opt.step()
                    for p, p_clone in zip(params, params_clone):
                        if p.grad is not None:
                            self.assertEqual(p, p_clone)

    def test_unscale(self):
        model = self._create_simple_model()
        input_tensor = torch.rand(3, 1, 24, 24).to("npu:0")
        optim_cases = self._create_optimizer_cases()
        for _, fused_opt_obj, opt_kwargs in optim_cases:
            m = deepcopy(model)
            optimizer = fused_opt_obj(m.parameters(), **opt_kwargs)
            t = input_tensor.detach().clone()
            scaler = GradScaler(init_scale=128.0)
            with autocast():
                output = m(t)
                loss = output.mean()
            scaler.scale(loss).backward()
            grads_before_unscale = dict()
            for p in m.parameters():
                if p.grad is not None:
                    grads_before_unscale[p] = p.grad.clone().detach()
            scaler.unscale_(optimizer)
            for p in m.parameters():
                if p.grad is not None:
                    self.assertEqual(grads_before_unscale[p] / 128, p.grad)

    def test_simple_model_train_dynamic(self):
        model = self._create_simple_model()
        optim_cases = self._create_optimizer_cases()
        num_iters = 10
        for opt_obj, fused_opt_obj, opt_kwargs in optim_cases:
            m = deepcopy(model)
            opt = opt_obj(m.parameters(), **opt_kwargs)
            scaler = GradScaler()

            m_clone = deepcopy(model)
            opt_fused = fused_opt_obj(m_clone.parameters(), **opt_kwargs)
            scaler_fused = GradScaler()
            for _ in range(num_iters):
                input_tensor = torch.rand(3, 1, 24, 24).to("npu:0")

                with autocast():
                    output = m(input_tensor)
                    loss = output.mean()
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()

                with autocast():
                    output_fused = m_clone(input_tensor)
                    loss_fused = output_fused.mean()
                scaler_fused.scale(loss_fused).backward()
                scaler_fused.step(opt_fused)
                scaler_fused.update()
                self.assertRtolEqual(loss, loss_fused)

    def test_simple_model_train_static(self):
        model = self._create_simple_model()
        optim_cases = self._create_optimizer_cases()
        num_iters = 10
        for opt_obj, fused_opt_obj, opt_kwargs in optim_cases:
            m = deepcopy(model)
            opt = opt_obj(m.parameters(), **opt_kwargs)
            scaler = GradScaler(dynamic=False, init_scale=128)

            m_clone = deepcopy(model)
            opt_fused = fused_opt_obj(m_clone.parameters(), **opt_kwargs)
            scaler_fused = GradScaler(dynamic=False, init_scale=128)
            for _ in range(num_iters):
                input_tensor = torch.rand(3, 1, 24, 24).to("npu:0")

                with autocast():
                    output = m(input_tensor)
                loss = output.float().mean()
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()

                with autocast():
                    output_fused = m_clone(input_tensor)
                loss_fused = output_fused.float().mean()
                scaler_fused.scale(loss_fused).backward()
                scaler_fused.step(opt_fused)
                scaler_fused.update()
                self.assertRtolEqual(loss, loss_fused)

    def test_clip_grad_norm_fused(self):
        optim_cases = self._create_optimizer_cases()
        for _, fused_opt_obj, opt_kwargs in optim_cases:
            params = self._create_simple_params_and_grads()
            params_clone = self._create_params_clone(params)

            fused_opt = fused_opt_obj(params_clone, **opt_kwargs)

            grad_norm = torch.nn.utils.clip_grad_norm_(params, 5.0)
            grad_norm_fused = fused_opt.clip_grad_norm_fused_(5.0)
            for p, p_clone in zip(params, params_clone):
                if p.grad is not None:
                    self.assertRtolEqual(p.grad, p_clone.grad)
            self.assertRtolEqual(grad_norm, grad_norm_fused)


if __name__ == "__main__":
    run_tests()
