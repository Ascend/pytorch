import unittest
from copy import deepcopy

import torch
from torch.optim import SGD, Adam, AdamW, Adadelta, RMSprop
from torch_npu.npu.amp import GradScaler, autocast
from torch_npu.optim import (
    NpuFusedSGD, NpuFusedAdam, NpuFusedAdamW, NpuFusedAdamP,
    NpuFusedLamb, NpuFusedAdadelta, NpuFusedBertAdam,
    NpuFusedRMSprop, NpuFusedRMSpropTF
)

from torch_npu.testing.testcase import TestCase, run_tests


class TestFusedOptim(TestCase):
    def __init__(self, method_name='runTest'):
        super(TestCase, self).__init__(method_name)

        self.optim_cases = [
            (SGD, NpuFusedSGD, dict(lr=0.01, momentum=0.9, weight_decay=0.001)),
            (Adam, NpuFusedAdam, dict(eps=1e-8, betas=(0.9, 0.999), lr=2e-3, weight_decay=0.05)),
            (AdamW, NpuFusedAdamW, dict(eps=1e-8, betas=(0.9, 0.999), lr=2e-3, weight_decay=0.05)),
            (Adadelta, NpuFusedAdadelta, dict(lr=1.0, rho=0.9, eps=1e-6, weight_decay=0.05)),
            (RMSprop, NpuFusedRMSprop, dict(eps=0.001, lr=0.01, weight_decay=1e-5)),

            # 3rd-party optimizers
            (None, NpuFusedAdamP, dict(eps=1e-5, betas=(0.9, 0.999), lr=2e-3, weight_decay=0.05)),
            (None, NpuFusedLamb, dict(lr=0.01, eps=1e-5)),
            (None, NpuFusedBertAdam, dict(lr=0.01, warmup=0.1, t_total=20, max_grad_norm=-1)),
            (None, NpuFusedRMSpropTF, dict(eps=0.001, lr=0.01, weight_decay=1e-5)),
        ]

        # Full tests for these optimizers will be run, including a small model.
        # Otherwise, only test the optimizer-related APIs.
        self.base_cases = [SGD, Adam, AdamW, Adadelta, RMSprop]

        # Cases and baseline for 3rd-party optimizers
        self.third_optim_baseline = dict()
        self.third_optim_baseline[NpuFusedAdamP] = [14.885, 65.714, 14.882, 65.75, 104.615, 152.75]
        self.third_optim_baseline[NpuFusedBertAdam] = [12.982, 61.537, 13.023, 61.5625, 99.305, 146.125]
        self.third_optim_baseline[NpuFusedLamb] = [13.407, 62.683, 13.414, 62.625, 101.258, 149.0]
        self.third_optim_baseline[NpuFusedRMSpropTF] = [14.9797, 65.911, 15.0, 66.0, 104.8588, 153.0]

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
            torch.arange(6).reshape(2, 3).float().npu(),
            torch.arange(12).reshape(4, 3).float().npu(),
            torch.arange(6).reshape(2, 3).half().npu(),
            torch.arange(12).reshape(4, 3).half().npu(),
            torch.arange(15).reshape(5, 3).float().npu(),
            torch.arange(18).reshape(6, 3).half().npu(),
            torch.arange(6).reshape(2, 3).float().npu(),
        ]

        for i, p in enumerate(params):
            if i < len(params) - 1:
                p.requires_grad = True
                p.grad = p.clone().detach() / 100.

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
            if opt_obj is None:
                continue
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
                            self.assertRtolEqual(p, p_clone, prec=1e-3)

    def test_step_3rd_optims(self):
        optim_cases = self._create_optimizer_cases(all_cases=True)
        num_iters = 10
        for _, fused_opt_obj, opt_kwargs in optim_cases:
            if fused_opt_obj not in self.third_optim_baseline:
                continue
            params = self._create_simple_params_and_grads()
            fused_opt = fused_opt_obj(params, **opt_kwargs)
            with torch.no_grad():
                for _ in range(num_iters):
                    fused_opt.step()
            for i, p in enumerate(params):
                if p.grad is not None:
                    self.assertRtolEqual(p.sum().item(), self.third_optim_baseline[fused_opt_obj][i])

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
                self.assertRtolEqual(loss, loss_fused, prec=1e-3)

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
                    self.assertRtolEqual(p.grad, p_clone.grad, prec=1e-3)
            self.assertRtolEqual(grad_norm.float(), grad_norm_fused, prec=1e-3)


if __name__ == "__main__":
    run_tests()
