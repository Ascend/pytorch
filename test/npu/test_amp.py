import unittest
from itertools import chain

import torch

import torch_npu
from torch_npu.npu.amp import GradScaler, autocast
from torch_npu.testing.common_utils import SupportedDevices
from torch_npu.testing.testcase import TestCase, run_tests


def make_device_overflow_1():
    float_tensor = torch.tensor([40000.0], dtype=torch.float16).npu()
    float_tensor = float_tensor + float_tensor


def make_device_overflow_2(model):
    for param in model.parameters():
        if param.grad is not None:
            param.grad = torch.full_like(param.grad, float("inf"))
            break


class TestAmp(TestCase):

    def test_grad_scaling_scale(self):
        scaler = GradScaler(init_scale=2.)
        t0 = torch.full((1,), 4.0, dtype=torch.float32, device="npu")
        t1 = torch.full((1,), 4.0, dtype=torch.float32, device="npu")
        # Create some nested iterables of tensors on different devices.
        outputs = (t1.clone(), (t0.clone(), t1.clone()), [t0.clone(), (t1.clone(), t0.clone())])
        outputs = scaler.scale(outputs)
        self.assertTrue(outputs[0] == 8.0 and outputs[1][0] == 8.0 and outputs[1][1] == 8.0 and
                        outputs[2][0] == 8.0 and outputs[2][1][0] == 8.0 and outputs[2][1][1] == 8.0)
        self.assertTrue(scaler._scale.device == t1.device)

    def test_grad_scaling_state_dict(self):
        for lazy_init_scale in True, False:
            s0 = GradScaler(init_scale=3., growth_factor=4., backoff_factor=.5, growth_interval=2)
            s1 = GradScaler(init_scale=6., growth_factor=7., backoff_factor=.8, growth_interval=1)

            # sets a random value for load_state_dict to overwrite
            s1._init_growth_tracker = 7

            if lazy_init_scale:
                # Dummy scale() call to ensure the scale tensor is lazily initialized.
                s1.scale(torch.full((1,), 4.0, dtype=torch.float32, device="npu"))
                self.assertTrue(isinstance(s1._scale, torch.npu.FloatTensor))

            s1.load_state_dict(s0.state_dict())

            self.assertTrue(s1.get_scale() == 3.)
            self.assertTrue(s1.get_growth_factor() == 4.)
            self.assertTrue(s1.get_backoff_factor() == .5)
            self.assertTrue(s1.get_growth_interval() == 2)
            self.assertTrue(s1._init_growth_tracker == 0)

    def _create_scaling_models_optimizers(self, device="npu"):
        # Create a module+optimizer that will use scaling, and a control module+optimizer
        # that will not use scaling, against which the scaling-enabled module+optimizer can be compared.
        mod_control = torch.nn.Sequential(torch.nn.Linear(8, 8), torch.nn.Linear(8, 8)).to(device=device)
        mod_scaling = torch.nn.Sequential(torch.nn.Linear(8, 8), torch.nn.Linear(8, 8)).to(device=device)
        for c, s in zip(mod_control.parameters(), mod_scaling.parameters()):
            s.data.copy_(c.data)

        opt_control = torch.optim.SGD(mod_control.parameters(), lr=1.0)
        opt_scaling = torch.optim.SGD(mod_scaling.parameters(), lr=1.0)

        ret = (mod_control, mod_scaling, opt_control, opt_scaling)
        return ret

    def _create_scaling_case(self, device="npu", dtype=torch.float):
        data = [(torch.randn((8, 8), dtype=dtype, device=device), torch.randn((8, 8), dtype=dtype, device=device)),
                (torch.randn((8, 8), dtype=dtype, device=device), torch.randn((8, 8), dtype=dtype, device=device)),
                (torch.randn((8, 8), dtype=dtype, device=device), torch.randn((8, 8), dtype=dtype, device=device)),
                (torch.randn((8, 8), dtype=dtype, device=device), torch.randn((8, 8), dtype=dtype, device=device))]

        loss_fn = torch.nn.MSELoss().npu()

        skip_iter = 2

        return self._create_scaling_models_optimizers(device=device) + (data, loss_fn, skip_iter)

    # _run_scaling_case generalizes some single-optimizer test logic to avoid too much copy-pasting below.
    def _run_scaling_case(self, run, unskipped, skipped, atol=1e-7):
        # Ensure scaling can be disabled without changing user control flow.
        for enabled in True, False:
            mod_control, mod_scaling, opt_control, opt_scaling, data, loss_fn, skip_iter = self._create_scaling_case()

            # For functionality, test with a modest initial scale, and an unrealistically-large growth factor
            # so any potential errors with the growth factor handling will be magnified.
            scaler = GradScaler(init_scale=128., growth_factor=2.0, enabled=enabled, growth_interval=1)

            _ = run(data, mod_control, opt_control, scaler, loss_fn, skip_iter, False)
            ret = run(data, mod_scaling, opt_scaling, scaler, loss_fn, skip_iter, True)

            # Allows run() to optionally return a different scaler instance.
            scaler = ret if ret else scaler

            # If scaling was enabled, the scale factor should have been multiplied by the growth factor
            # len(data) - skipped times and the backoff factor "skipped" times.
            if enabled:
                net_growth = scaler.get_growth_factor()**unskipped if unskipped > 0 else 1.0
                net_backoff = scaler.get_backoff_factor()**skipped if skipped > 0 else 1.0
                self.assertTrue(scaler.get_scale() == (128. * net_growth * net_backoff))
            else:
                self.assertTrue(scaler.get_scale() == 1.0)

            for c, s in zip(mod_control.parameters(), mod_scaling.parameters()):
                c = c.cpu().to(torch.float).detach().numpy()
                s = s.cpu().to(torch.float).detach().numpy()
                self.assertRtolEqual(c, s, atol)

    # Compares no scaling + no autocasting against scaling + autocasting.
    @SupportedDevices(['Ascend910A', 'Ascend910P'])
    def test_grad_scaling_autocast_1(self):
        def run(data, model, optimizer, scaler, loss_fn, skip_iter, try_scaling_api):
            for i, (input_data, target) in enumerate(data):
                optimizer.zero_grad()
                with torch.autocast('npu', enabled=try_scaling_api):
                    output = model(input_data)
                    loss = loss_fn(output, target)
                if try_scaling_api:
                    scaler.scale(loss).backward()
                    if i == skip_iter and scaler.is_enabled():
                        make_device_overflow_1()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    if (not scaler.is_enabled()) or (i != skip_iter):
                        optimizer.step()
            return scaler

        # sets atol=1e-3 because we're comparing pure fp32 arithmetic vs a mixture of fp16 and fp32
        self._run_scaling_case(run, unskipped=3, skipped=1, atol=1e-3)

    @SupportedDevices(['Ascend910B'])
    def test_grad_scaling_autocast_2(self):
        def run(data, model, optimizer, scaler, loss_fn, skip_iter, try_scaling_api):
            for i, (input_data, target) in enumerate(data):
                optimizer.zero_grad()
                with torch.autocast('npu', enabled=try_scaling_api):
                    output = model(input_data)
                    loss = loss_fn(output, target)
                if try_scaling_api:
                    scaler.scale(loss).backward()
                    if i == skip_iter and scaler.is_enabled():
                        make_device_overflow_2(model)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    if (not scaler.is_enabled()) or (i != skip_iter):
                        optimizer.step()
            return scaler

        # sets atol=1e-3 because we're comparing pure fp32 arithmetic vs a mixture of fp16 and fp32
        self._run_scaling_case(run, unskipped=3, skipped=1, atol=1e-3)

    @SupportedDevices(['Ascend910A', 'Ascend910P'])
    def test_grad_scaling_clipping_1(self):
        def run(data, model, optimizer, scaler, loss_fn, skip_iter, try_scaling_api):
            max_norm = 0.2  # A reasonable value that actually has an effect, based on printouts of grads
            for i, (input_data, target) in enumerate(data):
                optimizer.zero_grad()
                output = model(input_data)
                loss = loss_fn(output, target)
                if try_scaling_api:
                    scaler.scale(loss).backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm * scaler.get_scale())
                    if i == skip_iter and scaler.is_enabled():
                        make_device_overflow_1()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
                    if (not scaler.is_enabled()) or (i != skip_iter):
                        optimizer.step()

        self._run_scaling_case(run, unskipped=3, skipped=1, atol=1e-6)

    @SupportedDevices(['Ascend910B'])
    def test_grad_scaling_clipping_2(self):
        def run(data, model, optimizer, scaler, loss_fn, skip_iter, try_scaling_api):
            max_norm = 0.2  # A reasonable value that actually has an effect, based on printouts of grads
            for i, (input_data, target) in enumerate(data):
                optimizer.zero_grad()
                output = model(input_data)
                loss = loss_fn(output, target)
                if try_scaling_api:
                    scaler.scale(loss).backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm * scaler.get_scale())
                    if i == skip_iter and scaler.is_enabled():
                        make_device_overflow_2(model)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
                    if (not scaler.is_enabled()) or (i != skip_iter):
                        optimizer.step()

        self._run_scaling_case(run, unskipped=3, skipped=1, atol=1e-6)

    @SupportedDevices(['Ascend910A', 'Ascend910P'])
    def test_grad_scaling_clipping_separate_unscale_1(self):
        def run(data, model, optimizer, scaler, loss_fn, skip_iter, try_scaling_api):
            max_norm = 0.2  # A reasonable value that actually has an effect, based on printouts of grads
            for i, (input_data, target) in enumerate(data):
                optimizer.zero_grad()
                output = model(input_data)
                loss = loss_fn(output, target)
                if try_scaling_api:
                    scaler.scale(loss).backward()
                    if i == skip_iter and scaler.is_enabled():
                        make_device_overflow_1()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
                    if (not scaler.is_enabled()) or (i != skip_iter):
                        optimizer.step()

        self._run_scaling_case(run, unskipped=3, skipped=1)

    @SupportedDevices(['Ascend910B'])
    def test_grad_scaling_clipping_separate_unscale_2(self):
        def run(data, model, optimizer, scaler, loss_fn, skip_iter, try_scaling_api):
            max_norm = 0.2  # A reasonable value that actually has an effect, based on printouts of grads
            for i, (input_data, target) in enumerate(data):
                optimizer.zero_grad()
                output = model(input_data)
                loss = loss_fn(output, target)
                if try_scaling_api:
                    scaler.scale(loss).backward()
                    if i == skip_iter and scaler.is_enabled():
                        make_device_overflow_2(model)
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
                    if (not scaler.is_enabled()) or (i != skip_iter):
                        optimizer.step()

        self._run_scaling_case(run, unskipped=3, skipped=1)

    @SupportedDevices(['Ascend910A', 'Ascend910P'])
    def test_grad_scaling_penalty_1(self):
        def run(data, model, optimizer, scaler, loss_fn, skip_iter, try_scaling_api):
            for i, (input_data, target) in enumerate(data):
                optimizer.zero_grad()
                output = model(input_data)
                loss = loss_fn(output, target)

                if try_scaling_api:
                    grad_params = torch.autograd.grad(scaler.scale(loss),
                                                      model.parameters(), create_graph=True)
                    inv_scale = 1. / scaler.get_scale()
                    grad_params = [p * inv_scale for p in grad_params]
                else:
                    grad_params = torch.autograd.grad(loss, model.parameters(), create_graph=True)

                grad_norm = 0
                for grad in grad_params:
                    grad_norm += grad.pow(2).sum()
                grad_norm = grad_norm.sqrt()
                loss = loss + grad_norm

                if try_scaling_api:
                    scaler.scale(loss).backward()
                    if i == skip_iter and scaler.is_enabled():
                        make_device_overflow_1()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    if (not scaler.is_enabled()) or (i != skip_iter):
                        optimizer.step()

        self._run_scaling_case(run, unskipped=3, skipped=1)

    @SupportedDevices(['Ascend910B'])
    def test_grad_scaling_penalty_2(self):
        def run(data, model, optimizer, scaler, loss_fn, skip_iter, try_scaling_api):
            for i, (input_data, target) in enumerate(data):
                optimizer.zero_grad()
                output = model(input_data)
                loss = loss_fn(output, target)

                if try_scaling_api:
                    grad_params = torch.autograd.grad(scaler.scale(loss),
                                                      model.parameters(), create_graph=True)
                    inv_scale = 1. / scaler.get_scale()
                    grad_params = [p * inv_scale for p in grad_params]
                else:
                    grad_params = torch.autograd.grad(loss, model.parameters(), create_graph=True)

                grad_norm = 0
                for grad in grad_params:
                    grad_norm += grad.pow(2).sum()
                grad_norm = grad_norm.sqrt()
                loss = loss + grad_norm

                if try_scaling_api:
                    scaler.scale(loss).backward()
                    if i == skip_iter and scaler.is_enabled():
                        make_device_overflow_2(model)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    if (not scaler.is_enabled()) or (i != skip_iter):
                        optimizer.step()

        self._run_scaling_case(run, unskipped=3, skipped=1)

    def test_grad_scaling_accumulation(self):
        def run(data, model, optimizer, scaler, loss_fn, skip_iter, try_scaling_api):
            iters_to_accumulate = 2
            for i, (input_data, target) in enumerate(data):
                output = model(input_data)
                loss = loss_fn(output, target)
                loss = loss / iters_to_accumulate
                if try_scaling_api:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
                if (i + 1) % iters_to_accumulate == 0:
                    if try_scaling_api:
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad()
                    else:
                        optimizer.step()
                        optimizer.zero_grad()

        self._run_scaling_case(run, unskipped=2, skipped=0)

    @SupportedDevices(['Ascend910A', 'Ascend910P'])
    def test_grad_scaling_multiple_1(self):
        # Tests gradient scaling with 2 models and 2 optimizers that both receive gradients from 2 losses.
        # Some of the logic here cannot reuse the generic helper functions created for the 1-optimizer cases.
        for enabled in True, False:
            mod_control0, mod_scaling0, opt_control0, opt_scaling0, data, loss_fn, skip_iter = \
                self._create_scaling_case()
            mod_control1, mod_scaling1, opt_control1, opt_scaling1 = \
                self._create_scaling_models_optimizers()

            scaler = GradScaler(init_scale=128., growth_factor=2.0, enabled=enabled, growth_interval=1)

            def run(model0, model1, optimizer0, optimizer1, try_scaling_api):
                for i, (input_data, target) in enumerate(data):
                    optimizer0.zero_grad()
                    optimizer1.zero_grad()
                    output0 = model0(input_data)
                    output1 = model1(input_data)
                    loss0 = loss_fn(0.3 * output0 + 0.7 * output1, target)
                    loss1 = loss_fn(0.6 * output0 - 0.4 * output1, target)

                    if try_scaling_api:
                        scaler.scale(loss0).backward(retain_graph=True)
                        scaler.scale(loss1).backward()
                        if i == skip_iter and scaler.is_enabled():
                            make_device_overflow_1()

                        # As an additional stress test, separately unscale for one of the optimizers.
                        scaler.unscale_(optimizer0)

                        scaler.step(optimizer0)
                        scaler.step(optimizer1)
                        scaler.update()
                    else:
                        loss0.backward(retain_graph=True)
                        loss1.backward()
                        if (not scaler.is_enabled()) or (i != skip_iter):
                            optimizer0.step()
                            optimizer1.step()

            run(mod_control0, mod_control1, opt_control0, opt_control1, False)
            run(mod_scaling0, mod_scaling1, opt_scaling0, opt_scaling1, True)

            # The loss scale should have been multiplied by the growth factor 3 times and the backoff factor once.
            self.assertTrue(scaler.get_scale() == (128. * scaler.get_growth_factor()**3 *
                                                   scaler.get_backoff_factor()**1) if enabled else 1.0)

            for c, s in zip(chain(mod_control0.parameters(), mod_control1.parameters()),
                            chain(mod_scaling0.parameters(), mod_scaling1.parameters())):
                c = c.cpu().to(torch.float).detach().numpy()
                s = s.cpu().to(torch.float).detach().numpy()
                self.assertRtolEqual(c, s, 1e-7)

    @SupportedDevices(['Ascend910B'])
    def test_grad_scaling_multiple_2(self):
        # Tests gradient scaling with 2 models and 2 optimizers that both receive gradients from 2 losses.
        # Some of the logic here cannot reuse the generic helper functions created for the 1-optimizer cases.
        for enabled in True, False:
            mod_control0, mod_scaling0, opt_control0, opt_scaling0, data, loss_fn, skip_iter = \
                self._create_scaling_case()
            mod_control1, mod_scaling1, opt_control1, opt_scaling1 = \
                self._create_scaling_models_optimizers()

            scaler = GradScaler(init_scale=128., growth_factor=2.0, enabled=enabled, growth_interval=1)

            def run(model0, model1, optimizer0, optimizer1, try_scaling_api):
                for i, (input_data, target) in enumerate(data):
                    optimizer0.zero_grad()
                    optimizer1.zero_grad()
                    output0 = model0(input_data)
                    output1 = model1(input_data)
                    loss0 = loss_fn(0.3 * output0 + 0.7 * output1, target)
                    loss1 = loss_fn(0.6 * output0 - 0.4 * output1, target)

                    if try_scaling_api:
                        scaler.scale(loss0).backward(retain_graph=True)
                        scaler.scale(loss1).backward()
                        if i == skip_iter and scaler.is_enabled():
                            make_device_overflow_2(model0)

                        # As an additional stress test, separately unscale for one of the optimizers.
                        scaler.unscale_(optimizer0)

                        scaler.step(optimizer0)
                        scaler.step(optimizer1)
                        scaler.update()
                    else:
                        loss0.backward(retain_graph=True)
                        loss1.backward()
                        if (not scaler.is_enabled()) or (i != skip_iter):
                            optimizer0.step()
                            optimizer1.step()

            run(mod_control0, mod_control1, opt_control0, opt_control1, False)
            run(mod_scaling0, mod_scaling1, opt_scaling0, opt_scaling1, True)

            # The loss scale should have been multiplied by the growth factor 3 times and the backoff factor once.
            self.assertTrue(scaler.get_scale() == (128. * scaler.get_growth_factor() ** 3 *
                                                   scaler.get_backoff_factor() ** 1) if enabled else 1.0)

            for c, s in zip(chain(mod_control0.parameters(), mod_control1.parameters()),
                            chain(mod_scaling0.parameters(), mod_scaling1.parameters())):
                c = c.cpu().to(torch.float).detach().numpy()
                s = s.cpu().to(torch.float).detach().numpy()
                self.assertRtolEqual(c, s, 1e-7)

    def test_autocast_custom_enabled(self):
        class MyMM(torch.autograd.Function):
            @staticmethod
            @torch.npu.amp.custom_fwd
            def forward(ctx, a, b):
                self.assertTrue(ctx._dtype is torch.get_autocast_dtype("npu"))
                self.assertTrue(a.dtype is torch.float32)
                self.assertTrue(b.dtype is torch.float32)
                self.assertTrue(torch.npu.is_autocast_enabled())
                ctx.save_for_backward(a, b)
                return a.mm(b)

            @staticmethod
            @torch.npu.amp.custom_bwd
            def backward(ctx, grad):
                self.assertTrue(ctx._dtype is torch.get_autocast_dtype("npu"))
                self.assertTrue(torch.npu.is_autocast_enabled())
                a, b = ctx.saved_tensors
                return grad.mm(b.t()), a.t().mm(grad)

        mymm = MyMM.apply

        x = torch.randn((8, 8), device="npu", dtype=torch.float32, requires_grad=True)
        y = torch.randn((8, 8), device="npu", dtype=torch.float32, requires_grad=True)

        with torch.npu.amp.autocast():
            output = mymm(x, y)
            self.assertTrue(output.dtype is torch.float16)
            loss = output.sum()
        loss.backward()

    def test_autocast_custom_cast_inputs(self):
        class MyMM(torch.autograd.Function):
            @staticmethod
            @torch.npu.amp.custom_fwd(cast_inputs=torch.float32)
            def forward(ctx, a, container, expect_type):
                self.assertTrue(ctx._dtype is torch.get_autocast_dtype("npu"))
                b = container[1][0]
                self.assertTrue(a.dtype is expect_type)
                self.assertTrue(b.dtype is expect_type)
                self.assertFalse(torch.npu.is_autocast_enabled())
                ctx.save_for_backward(a, b)
                return a.mm(b)

            @staticmethod
            @torch.npu.amp.custom_bwd
            def backward(ctx, grad):
                self.assertTrue(ctx._dtype is torch.get_autocast_dtype("npu"))
                a, b = ctx.saved_tensors
                return grad.mm(b.t()), None, None

        mymm = MyMM.apply

        x = torch.randn((8, 8), device="npu", dtype=torch.float16, requires_grad=True)
        # Puts one input tensor in a nested container.  y's contained Tensor won't receive a gradient,
        # because torch.autograd.Function can't hand gradients back to non-Tensor forward arguments.
        # Sets requires_grad=False explicitly so we don't lie about expecting a gradient.
        y = (0, {0: torch.randn((8, 8), device="npu", dtype=torch.float16, requires_grad=False)})

        with torch.autocast('npu', ):
            output = mymm(x, y, torch.float32)
            self.assertTrue(output.dtype is torch.float32)
            loss = output.sum()
        loss.backward()

        # Tests if custom_fwd becomes a no-op when mymm runs outside an autocast-enabled region.
        output = mymm(x, y, torch.float16)
        self.assertTrue(output.dtype is torch.float16)
        loss = output.sum()
        loss.backward()


if __name__ == "__main__":
    run_tests()
