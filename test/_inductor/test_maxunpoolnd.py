import torch
import torch.nn as nn
from torch.testing._internal.common_utils import run_tests, parametrize, instantiate_parametrized_tests
from testutils import TestUtils
import torch_npu


class TestMaxUnpool1d(TestUtils):
    @parametrize('shape', [(1, 2, 500)])
    @parametrize('dtype', ['float16', 'float32'])
    @parametrize('kernel_size', [2])
    @parametrize('stride', [2])
    def test_maxunpool1d(self, shape, dtype, kernel_size, stride):
        torch.use_deterministic_algorithms(True)

        input_tensor = self._generate_tensor(shape, dtype)

        pool = nn.MaxPool1d(kernel_size=kernel_size, stride=stride, return_indices=True)
        x_cpu, indices_cpu = pool(input_tensor.cpu())

        x = x_cpu.npu()
        indices = indices_cpu.npu()

        unpool = nn.MaxUnpool1d(kernel_size=kernel_size, stride=stride).npu()

        def unpool_forward(x, indices):
            return unpool(x, indices)

        maxunpool_res = unpool_forward(x, indices)
        unpool_compiled = torch.compile(unpool_forward, backend="inductor")
        inductor_res = unpool_compiled(x, indices)

        self.assertEqual(maxunpool_res, inductor_res, atol=1e-3, rtol=1e-3)


class TestMaxUnpool2d(TestUtils):
    @parametrize('shape', [(2, 8, 64, 128)])
    @parametrize('dtype', ['float16', 'float32'])
    @parametrize('kernel_size', [2])
    @parametrize('stride', [2])
    def test_maxunpool2d(self, shape, dtype, kernel_size, stride):
        torch.use_deterministic_algorithms(True)

        input_tensor = self._generate_tensor(shape, dtype)

        pool = nn.MaxPool2d(kernel_size=kernel_size, stride=stride, return_indices=True)
        x_cpu, indices_cpu = pool(input_tensor.cpu())

        x = x_cpu.npu()
        indices = indices_cpu.npu()

        unpool = nn.MaxUnpool2d(kernel_size=kernel_size, stride=stride).npu()

        def unpool_forward(x, indices):
            return unpool(x, indices)

        maxunpool_res = unpool_forward(x, indices)
        unpool_compiled = torch.compile(unpool_forward, backend="inductor")
        inductor_res = unpool_compiled(x, indices)

        self.assertEqual(maxunpool_res, inductor_res, atol=1e-3, rtol=1e-3)


class TestMaxUnpool3d(TestUtils):
    @parametrize('shape', [(20, 16, 51, 33)])
    @parametrize('dtype', ['float16', 'float32'])
    @parametrize('kernel_size', [3])
    @parametrize('stride', [2])
    def test_maxunpool3d(self, shape, dtype, kernel_size, stride):
        torch.use_deterministic_algorithms(True)

        input_tensor = self._generate_tensor(shape, dtype)

        pool = nn.MaxPool3d(kernel_size=kernel_size, stride=stride, return_indices=True)
        x_cpu, indices_cpu = pool(input_tensor.cpu())

        x = x_cpu.npu()
        indices = indices_cpu.npu()

        unpool = nn.MaxUnpool3d(kernel_size=kernel_size, stride=stride).npu()

        def unpool_forward(x, indices):
            return unpool(x, indices)

        maxunpool_res = unpool_forward(x, indices)
        unpool_compiled = torch.compile(unpool_forward, backend="inductor")
        inductor_res = unpool_compiled(x, indices)

        self.assertEqual(maxunpool_res, inductor_res, atol=1e-3, rtol=1e-3)


instantiate_parametrized_tests(TestMaxUnpool1d)
instantiate_parametrized_tests(TestMaxUnpool2d)
instantiate_parametrized_tests(TestMaxUnpool3d)


if __name__ == "__main__":
    run_tests()
