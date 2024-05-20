import os
from copy import deepcopy

import torch
import torch.nn.functional as F
import torch.nn.parallel as dp
from torch import nn
from torch.testing._internal.common_utils import run_tests, TestCase

import torch_npu
from torch_npu.npu.amp.autocast_mode import autocast
from torch_npu.testing.common_distributed import skipIfUnsupportMultiNPU


class TestDataParallel(TestCase):
    def setUp(self):
        super().setUp()
        os.environ['PYTORCH_NPU_ALLOC_CONF'] = 'expandable_segments:False'

    @skipIfUnsupportMultiNPU(2)
    def test_data_parallel_rnn(self):
        class TestModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.rnn = torch.nn.LSTM(
                    300, 1024, 1, batch_first=True, bidirectional=True
                )

            def forward(self, x):
                self.rnn.flatten_parameters()
                return self.rnn(x)

        def step(model):
            opt = torch.optim.SGD(model.parameters(), lr=10)
            input_t = torch.ones(4, 4, 300).to("npu:0")
            output = model(input_t)
            loss = F.mse_loss(output[0], torch.zeros_like(output[0]))
            loss.backward()
            opt.step()

        with torch.no_grad():
            model = TestModule().to("npu:0").eval()
            model_dp = torch.nn.DataParallel(deepcopy(model))

            model_dp(torch.rand(2, 4, 300).to("npu:0"))

        step(model)
        step(model_dp)

        for p1, p2 in zip(model.parameters(), model_dp.parameters()):
            self.assertTrue(p1.allclose(p2, rtol=1e-04, atol=1e-04, equal_nan=True))

    @skipIfUnsupportMultiNPU(2)
    def test_parallel_apply(self):
        l1 = nn.Linear(10, 5).to("npu:0", torch.float)
        l2 = nn.Linear(10, 5).to("npu:1", torch.float)
        i1 = torch.randn(2, 10, device="npu:0", dtype=torch.float)
        i2 = torch.randn(2, 10, device="npu:1", dtype=torch.float)
        expected1 = l1(i1)
        expected2 = l2(i2)
        modules = (l1, l2)
        expected_outputs = (expected1, expected2)

        # each input can be either a collection of positional arguments
        #                       or an object representing the single argument
        for inputs in [((i1,), (i2,)), (i1, i2)]:
            outputs = torch_npu.utils._module._parallel_apply(modules, inputs, None)
            for out, expected in zip(outputs, expected_outputs):
                self.assertEqual(out, expected)

    @skipIfUnsupportMultiNPU(2)
    def test_data_parallel_small_back(self):
        fc = nn.Linear(10, 5).float().npu(0)
        input_t = torch.randn(20, 10, dtype=torch.float, device="npu:0")
        output = dp.data_parallel(fc, input_t, (0, 1))
        self.assertEqual(output, fc(input_t))

    @skipIfUnsupportMultiNPU(2)
    def test_data_parallel_no_grad(self):
        test = self

        class Layer(nn.Module):
            def forward(self, x):
                test.assertFalse(torch.is_grad_enabled())
                return x

        test_layer = Layer()
        input_t = torch.randn(20, 10, dtype=torch.float, device="npu:0")
        with torch.no_grad():
            dp.data_parallel(test_layer, input_t, (0, 1))
        self.assertRaises(AssertionError, lambda: dp.data_parallel(test_layer, input_t, (0, 1)))

    @skipIfUnsupportMultiNPU(2)
    def test_data_parallel_module_zero_inputs(self):
        class TestModule(nn.Module):
            def forward(self):
                t = torch.eye(2, 3, device="npu:0")
                return t + (1 - t)

        def test_helper(output, expected):
            self.assertEqual(output.get_device(), 0)
            self.assertEqual(output, expected)

        expected = torch.ones(2, 3, device="npu:0")
        model = TestModule()

        test_helper(nn.DataParallel(model, [0])(), expected)
        test_helper(nn.DataParallel(model, [0, 1])(), expected)
        test_helper(dp.data_parallel(model, None, [0]), expected)
        test_helper(dp.data_parallel(model, (), [0, 1]), expected)

    @skipIfUnsupportMultiNPU(2)
    def test_data_parallel_device_args(self):
        npu0 = torch.device("npu:0")
        npu1 = torch.device("npu:1")

        # test output_device
        fc = nn.Linear(10, 5).to(npu0, torch.float)
        input_t = torch.randn(20, 10, dtype=torch.float, device=npu0, requires_grad=True)
        output = dp.data_parallel(fc, input_t, device_ids=(0, 1), output_device=npu0)
        self.assertEqual(output, fc(input_t))

        # test device_ids
        fc = nn.Linear(10, 5).to(npu0, torch.float)
        input_t = torch.randn(20, 10, dtype=torch.float, device=npu0, requires_grad=True)
        output = dp.data_parallel(fc, input_t, device_ids=(npu0, npu1), output_device=npu0)
        self.assertEqual(output, fc(input_t))


    def _test_scatter(self, tensor):
        x = tensor.detach().requires_grad_()
        result = dp.scatter(x, (0, 1))
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0], x[:2])
        self.assertEqual(result[0].get_device(), 0)
        self.assertEqual(result[1], x[2:])
        self.assertEqual(result[1].get_device(), 1)
        grad = result[0].detach().clone().fill_(2)
        result[0].backward(grad)
        self.assertEqual(x.grad[:2], grad)
        self.assertEqual(x.grad[2:], grad.clone().zero_())

    @skipIfUnsupportMultiNPU(2)
    def test_scatter_cpu(self):
        self._test_scatter(torch.randn((4, 4), dtype=torch.double))

    @skipIfUnsupportMultiNPU(2)
    def test_scatter_npu(self):
        self._test_scatter(torch.randn((4, 4), dtype=torch.double).npu(0))


    def _test_gather(self, output_device):
        inputs = (
            torch.randn(2, 4, device="npu:0", requires_grad=True, dtype=torch.double),
            torch.randn(2, 4, device="npu:1", requires_grad=True, dtype=torch.double),
        )
        result = dp.gather(inputs, output_device)
        self.assertEqual(result.size(), torch.Size([4, 4]))
        self.assertEqual(result[:2], inputs[0])
        self.assertEqual(result[2:], inputs[1])
        if output_device != -1:
            self.assertEqual(result.get_device(), output_device)
        else:
            self.assertFalse(result.is_npu)
        grad = torch.randn((4, 4), dtype=torch.double)
        if output_device != -1:
            grad = grad.npu(output_device)
        result.backward(grad)
        self.assertEqual(inputs[0].grad, grad[:2])
        self.assertEqual(inputs[1].grad, grad[2:])


        # test scalar inputs, should stack into a vector in this case
        inputs = (
            torch.randn((), device="npu:0", requires_grad=True, dtype=torch.double),
            torch.randn((), device="npu:1", requires_grad=True, dtype=torch.double),
        )
        result = dp.gather(inputs, output_device)
        self.assertEqual(result.size(), torch.Size([2]))
        self.assertEqual(result[0], inputs[0])
        self.assertEqual(result[1], inputs[1])
        if output_device != -1:
            self.assertEqual(result.get_device(), output_device)
        else:
            self.assertFalse(result.is_npu)
        grad = torch.randn(2, dtype=torch.double)
        if output_device != -1:
            grad = grad.npu(output_device)
        result.backward(grad)
        self.assertEqual(inputs[0].grad, grad[0])
        self.assertEqual(inputs[1].grad, grad[1])

    @skipIfUnsupportMultiNPU(2)
    def test_gather_cpu(self):
        self._test_gather(-1)

    @skipIfUnsupportMultiNPU(2)
    def test_gather_npu(self):
        self._test_gather(0)

    @skipIfUnsupportMultiNPU(2)
    def test_gather_different_len_dicts(self):
        inputs = (
            {"a": torch.randn(1, 2, requires_grad=True, device="npu:0")},
            {
                "b": torch.randn(1, 2, requires_grad=True, device="npu:1"),
                "a": torch.randn(1, 2, requires_grad=True, device="npu:1"),
            },
        )
        with self.assertRaises(ValueError):
            _ = dp.gather(inputs, target_device=0)

    @skipIfUnsupportMultiNPU(2)
    def test_replicate(self):
        module = nn.Linear(10, 5).float().npu()
        input_t = torch.randn(2, 10, dtype=torch.float, device="npu:0")
        expected_output = module(input_t)
        for devices in [(0, 1), [0, 1]]:
            replicas = dp.replicate(module, devices)
            for i, replica in enumerate(replicas):
                for p in replica.parameters():
                    self.assertEqual(p.get_device(), i)
                replica_input = input_t.npu(i)
                self.assertEqual(replica(replica_input), expected_output)

    @skipIfUnsupportMultiNPU(2)
    def test_replicate_buffers(self):
        net = nn.Module()
        net.bn = nn.BatchNorm2d(10)
        net.npu()
        for devices in [(0, 1), [0, 1]]:
            replicas = dp.replicate(net, devices)
            for i, replica in enumerate(replicas):
                self.assertEqual(
                    replica.bn.running_mean.get_device(),
                    i,
                    msg="buffer on wrong device",
                )
                self.assertEqual(
                    replica.bn.running_var.get_device(), i, msg="buffer on wrong device"
                )
                self.assertEqual(
                    replica.bn.num_batches_tracked.get_device(),
                    i,
                    msg="buffer on wrong device",
                )

    @skipIfUnsupportMultiNPU(2)
    def test_zero_grad(self):
        # zero_grad should warn about using gradients inside forward

        class Net(torch.nn.Module):
            def __init__(self, testcase):
                super().__init__()
                self._testcase = testcase

            def forward(self, x):
                with self._testcase.assertWarnsRegex(
                    UserWarning,
                    r"Calling \.zero_grad\(\) from a module created with nn\.DataParallel\(\) has no effect.",
                ):
                    self.zero_grad()
                return x

        module = Net(self).npu()
        dpm = dp.DataParallel(module)
        dpm(torch.rand(4, 3, 6, 5))

    @skipIfUnsupportMultiNPU(2)
    def test_autocast(self):
        class Model(torch.nn.Linear):
            def __init__(self):
                super().__init__(8, 8)

            def forward(self, input_t):
                with autocast():
                    return super().forward(input_t)

        model = dp.DataParallel(Model().npu().to(dtype=torch.float32))
        input_t = torch.randn((8, 8), dtype=torch.float32, device="npu:0")
        self.assertTrue(model(input_t).dtype is torch.float16)


if __name__ == "__main__":
    TestCase._default_dtype_check_enabled = True
    run_tests()
