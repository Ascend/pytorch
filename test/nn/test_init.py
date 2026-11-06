"""Additional NPU validation cases for torch.nn.init.dirac_.

Upstream lacks full dtype/layout and parameter-boundary coverage on NPU.
Valid results use independent expected values; CPU tensors only check invalid-input parity.
"""

# Owner(s): ["module: nn"]
import random
import unittest
import math
import string
from functools import reduce
from itertools import product
from operator import mul

import torch
import torch.nn.init as init
import torch.nn.functional as F
import torch_npu
import torch_npu.testing  # noqa: F401 - installs NPU test hooks
from torch.testing._internal.common_utils import (
    TestCase, TEST_SCIPY, skipIfNoLapack, skipIfTorchDynamo, run_tests
)

if TEST_SCIPY:
    from scipy import stats


class TestNNInit(TestCase):
    def setUp(self):
        super().setUp()
        random.seed(123)

    def _is_normal(self, tensor, mean, std):
        samples = tensor.view(-1).tolist()
        p_value = stats.kstest(samples, 'norm', args=(mean, std))[1]
        return p_value > 0.0001

    def _is_trunc_normal(self, tensor, mean, std, a, b):
        # scipy's trunc norm is suited for data drawn from N(0, 1),
        # so we need to transform our data to test it using scipy.
        z_samples = (tensor.view(-1) - mean) / std
        z_samples = z_samples.tolist()
        a0 = (a - mean) / std
        b0 = (b - mean) / std
        p_value = stats.kstest(z_samples, 'truncnorm', args=(a0, b0))[1]
        return p_value > 0.0001

    def _is_uniform(self, tensor, a, b):
        samples = tensor.view(-1).tolist()
        p_value = stats.kstest(samples, 'uniform', args=(a, (b - a)))[1]
        return p_value > 0.0001

    def _create_random_nd_tensor(self, dims, size_min, size_max):
        size = [random.randint(size_min, size_max) for _ in range(dims)]
        tensor = torch.zeros(size)
        return tensor

    def _random_float(self, a, b):
        return (b - a) * random.random() + a

    def test_calculate_gain_linear(self):
        for fn in ['linear', 'conv1d', 'conv2d', 'conv3d', 'conv_transpose2d', 'conv_transpose2d', 'conv_transpose3d']:
            gain = init.calculate_gain(fn)
            self.assertEqual(gain, 1)

    def test_calculate_gain_nonlinear(self):
        for fn in ['sigmoid', 'tanh', 'relu', 'leaky_relu']:
            gain = init.calculate_gain(fn)
            if fn == 'sigmoid':
                self.assertEqual(gain, 1)
            elif fn == 'tanh':  # 5 / 3
                self.assertEqual(gain, 1.6666666666666667)
            elif fn == 'relu':  # sqrt(2)
                self.assertEqual(gain, 1.4142135623730951)
            elif fn == 'leaky_relu':  # sqrt(2 / 1 + slope^2))
                self.assertEqual(gain, 1.4141428569978354)
            elif fn == 'selu':
                self.assertEqual(gain, 0.75)

    def test_calculate_gain_leaky_relu(self):
        for param in [None, 0, 0.01, 10]:
            gain = init.calculate_gain('leaky_relu', param)
            if param is None:  # Default slope is 0.01
                self.assertEqual(gain, 1.4141428569978354)
            elif param == 0:  # No slope = same gain as normal ReLU
                self.assertEqual(gain, 1.4142135623730951)
            elif param == 0.01:
                self.assertEqual(gain, 1.4141428569978354)
            elif param == 10:
                self.assertEqual(gain, 0.14071950894605836)

    def test_calculate_gain_leaky_relu_only_accepts_numbers(self):
        for param in [True, [1], {'a': 'b'}]:
            with self.assertRaises(ValueError):
                init.calculate_gain('leaky_relu', param)

    def test_calculate_gain_only_accepts_valid_nonlinearities(self):
        for n in [2, 5, 25]:
            # Generate random strings of lengths that definitely aren't supported
            random_string = ''.join([random.choice(string.ascii_lowercase) for i in range(n)])
            with self.assertRaises(ValueError):
                init.calculate_gain(random_string)

    @unittest.skipIf(not TEST_SCIPY, "Scipy not found.")
    def test_uniform(self):
        for dims in [1, 2, 4]:
            input_tensor = self._create_random_nd_tensor(dims, size_min=30, size_max=50)
            a = self._random_float(-3, 3)
            b = a + self._random_float(1, 5)
            init.uniform_(input_tensor, a=a, b=b)
            assert self._is_uniform(input_tensor, a, b)

    @unittest.skipIf(not TEST_SCIPY, "Scipy not found.")
    @skipIfTorchDynamo("scipy.kstest is failing under dynamo")
    def test_normal(self):
        for dims in [1, 2, 4]:
            input_tensor = self._create_random_nd_tensor(dims, size_min=30, size_max=50)
            mean = self._random_float(-3, 3)
            std = self._random_float(1, 5)
            init.normal_(input_tensor, mean=mean, std=std)

            assert self._is_normal(input_tensor, mean, std)

    @unittest.skipIf(not TEST_SCIPY, "Scipy not found.")
    @skipIfTorchDynamo("scipy.kstest is failing under dynamo")
    def test_trunc_normal(self):
        for dims in [1, 2, 4]:
            input_tensor = self._create_random_nd_tensor(dims, size_min=30, size_max=50)
            mean = self._random_float(-3, 3)
            std = self._random_float(.01, 1)
            a = self._random_float(mean - 2 * std, mean)
            b = self._random_float(mean, mean + 2 * std)
            init.trunc_normal_(input_tensor, mean=mean, std=std, a=a, b=b)

            assert self._is_trunc_normal(input_tensor, mean, std, a, b)

    @unittest.skipIf(not TEST_SCIPY, "Scipy not found.")
    @skipIfTorchDynamo("scipy.kstest is failing under dynamo")
    def test_trunc_normal_generator(self):
        gen = torch.Generator()
        gen.manual_seed(42)
        input_tensor = torch.empty(5)
        init.trunc_normal_(input_tensor, generator=gen)

        ref = torch.empty(5)
        torch.manual_seed(42)
        init.trunc_normal_(ref)

        self.assertEqual(input_tensor, ref)
        assert self._is_trunc_normal(input_tensor, mean=0, std=1, a=0, b=1)

    def test_constant(self):
        for dims in [1, 2, 4]:
            input_tensor = self._create_random_nd_tensor(dims, size_min=1, size_max=5)
            val = self._random_float(1, 10)
            init.constant_(input_tensor, val)

            self.assertEqual(input_tensor, input_tensor.clone().fill_(val))

    def test_ones_and_zeros(self):
        for init_fn_, val in zip([init.ones_, init.zeros_], [1, 0]):
            for dims in [1, 2, 4]:
                input_tensor = self._create_random_nd_tensor(dims, size_min=1, size_max=5)
                init_fn_(input_tensor)

                self.assertEqual(input_tensor, input_tensor.clone().fill_(val))

    def test_eye(self):
        input_tensor = self._create_random_nd_tensor(2, size_min=1, size_max=5)
        init.eye_(input_tensor)

        # Check every single element
        for i in range(input_tensor.size(0)):
            for j in range(input_tensor.size(1)):
                if i == j:
                    assert input_tensor[i][j] == 1
                else:
                    assert input_tensor[i][j] == 0

    def test_eye_only_works_on_2d_inputs(self):
        for dims in [1, 3]:
            with self.assertRaises(ValueError):
                tensor = self._create_random_nd_tensor(dims, size_min=1, size_max=3)
                init.eye_(tensor)

    def test_dirac_properties(self):
        for dims in [3, 4, 5]:
            for groups in [1, 2, 3]:
                # prepare random tensor with random sizes, but fits groups
                a, c, d, e = (random.randint(1, 5) for _ in range(4))
                b = random.randint(1, 5 * groups)  # same range as a*groups but all range allowed
                # make sure first dim divides by groups
                input_tensor = torch.randn((a * groups, b, c, d, e)[:dims])

                init.dirac_(input_tensor, groups)

                c_out, c_in = input_tensor.size(0) // groups, input_tensor.size(1)
                min_d = min(c_out, c_in)
                # Check number of nonzeros is equivalent to smallest dim (for each group)
                assert torch.nonzero(input_tensor).size(0) == min_d * groups
                # Check sum of values (can have precision issues, hence assertEqual) is also equivalent
                self.assertEqual(input_tensor.sum(), min_d * groups)

    def test_dirac_identity(self):
        for groups in [1, 3]:
            batch, in_c, out_c, size, kernel_size = 8, 3, 9, 5, 3  # in_c, out_c must divide by groups
            eff_out_c = out_c // groups

            # Test 1D
            input_var = torch.randn(batch, in_c, size)
            filter_var = torch.zeros(eff_out_c, in_c, kernel_size)
            filter_var = torch.cat([filter_var] * groups)
            init.dirac_(filter_var, groups)
            output_var = F.conv1d(input_var, filter_var)
            input_tensor, output_tensor = input_var.data, output_var.data  # Variables do not support nonzero
            for g in range(groups):
                # Assert in_c outputs are preserved (per each group)
                self.assertEqual(input_tensor[:, :, 1:-1],
                                 output_tensor[:, eff_out_c * g:eff_out_c * g + in_c, :])
                # Assert extra outputs are 0
                assert torch.nonzero(output_tensor[:, eff_out_c * g + in_c:eff_out_c * (g + 1), :]).numel() == 0

            # Test 2D
            input_var = torch.randn(batch, in_c, size, size)
            filter_var = torch.zeros(eff_out_c, in_c, kernel_size, kernel_size)
            filter_var = torch.cat([filter_var] * groups)
            init.dirac_(filter_var, groups)
            output_var = F.conv2d(input_var, filter_var)
            input_tensor, output_tensor = input_var.data, output_var.data  # Variables do not support nonzero
            for g in range(groups):
                # Assert in_c outputs are preserved (per each group)
                self.assertEqual(input_tensor[:, :, 1:-1, 1:-1],
                                 output_tensor[:, eff_out_c * g:eff_out_c * g + in_c, :, :])
                # Assert extra outputs are 0
                assert torch.nonzero(output_tensor[:, eff_out_c * g + in_c:eff_out_c * (g + 1), :, :]).numel() == 0

            # Test 3D
            input_var = torch.randn(batch, in_c, size, size, size)
            filter_var = torch.zeros(eff_out_c, in_c, kernel_size, kernel_size, kernel_size)
            filter_var = torch.cat([filter_var] * groups)
            init.dirac_(filter_var, groups)
            output_var = F.conv3d(input_var, filter_var)
            input_tensor, output_tensor = input_var.data, output_var.data
            for g in range(groups):
                # Assert in_c outputs are preserved (per each group)
                self.assertEqual(input_tensor[:, :, 1:-1, 1:-1, 1:-1],
                                 output_tensor[:, eff_out_c * g:eff_out_c * g + in_c, :, :, :])
                # Assert extra outputs are 0
                assert torch.nonzero(output_tensor[:, eff_out_c * g + in_c:eff_out_c * (g + 1), :, :, :]).numel() == 0

    def test_dirac_only_works_on_3_4_5d_inputs(self):
        for dims in [1, 2, 6]:
            with self.assertRaises(ValueError):
                tensor = self._create_random_nd_tensor(dims, size_min=1, size_max=3)
                init.dirac_(tensor)

    def test_xavier_uniform_errors_on_inputs_smaller_than_2d(self):
        for dims in [0, 1]:
            tensor = self._create_random_nd_tensor(dims, size_min=1, size_max=1)
            with self.assertRaises(ValueError):
                init.xavier_uniform_(tensor)

    def test_xavier_normal_errors_on_inputs_smaller_than_2d(self):
        for dims in [0, 1]:
            tensor = self._create_random_nd_tensor(dims, size_min=1, size_max=1)
            with self.assertRaises(ValueError):
                init.xavier_normal_(tensor)

    @unittest.skipIf(not TEST_SCIPY, "Scipy not found.")
    def test_xavier_uniform(self):
        for use_gain in [True, False]:
            for dims in [2, 4]:
                input_tensor = self._create_random_nd_tensor(dims, size_min=20, size_max=25)
                gain = 1

                if use_gain:
                    gain = self._random_float(0.1, 2)
                    init.xavier_uniform_(input_tensor, gain=gain)
                else:
                    init.xavier_uniform_(input_tensor)

                fan_in = input_tensor.size(1)
                fan_out = input_tensor.size(0)
                if input_tensor.dim() > 2:
                    fan_in *= input_tensor[0, 0].numel()
                    fan_out *= input_tensor[0, 0].numel()

                expected_std = gain * math.sqrt(2.0 / (fan_in + fan_out))
                bounds = expected_std * math.sqrt(3)
                assert self._is_uniform(input_tensor, -bounds, bounds)

    @unittest.skipIf(not TEST_SCIPY, "Scipy not found.")
    @skipIfTorchDynamo("scipy.kstest is failing under dynamo")
    def test_xavier_normal(self):
        for use_gain in [True, False]:
            for dims in [2, 4]:
                input_tensor = self._create_random_nd_tensor(dims, size_min=20, size_max=25)
                gain = 1

                if use_gain:
                    gain = self._random_float(0.1, 2)
                    init.xavier_normal_(input_tensor, gain=gain)
                else:
                    init.xavier_normal_(input_tensor)

                fan_in = input_tensor.size(1)
                fan_out = input_tensor.size(0)
                if input_tensor.dim() > 2:
                    fan_in *= input_tensor[0, 0].numel()
                    fan_out *= input_tensor[0, 0].numel()

                expected_std = gain * math.sqrt(2.0 / (fan_in + fan_out))
                assert self._is_normal(input_tensor, 0, expected_std)

    def test_kaiming_uniform_errors_on_inputs_smaller_than_2d(self):
        for dims in [0, 1]:
            with self.assertRaises(ValueError):
                tensor = self._create_random_nd_tensor(dims, size_min=1, size_max=1)
                init.kaiming_uniform_(tensor)

    def test_kaiming_normal_errors_on_inputs_smaller_than_2d(self):
        for dims in [0, 1]:
            with self.assertRaises(ValueError):
                tensor = self._create_random_nd_tensor(dims, size_min=1, size_max=1)
                init.kaiming_normal_(tensor)

    def test_kaiming_uniform_warning_on_0element_tensor(self):
        tensor = torch.empty(0, 1)
        with self.assertWarnsRegex(UserWarning, "Initializing zero-element tensors is a no-op"):
            _ = init.kaiming_uniform_(tensor)

    def test_kaiming_normal_warning_on_0element_tensor(self):
        tensor = torch.empty(0, 1)
        with self.assertWarnsRegex(UserWarning, "Initializing zero-element tensors is a no-op"):
            _ = init.kaiming_normal_(tensor)

    @unittest.skipIf(not TEST_SCIPY, "Scipy not found.")
    @skipIfTorchDynamo("scipy.kstest is failing under dynamo")
    def test_kaiming_uniform(self):
        for use_a in [True, False]:
            for dims in [2, 4]:
                for mode in ['fan_in', 'fan_out']:
                    input_tensor = self._create_random_nd_tensor(dims, size_min=20, size_max=25)
                    if use_a:
                        a = self._random_float(0.1, 2)
                        init.kaiming_uniform_(input_tensor, a=a, mode=mode)
                    else:
                        a = 0
                        init.kaiming_uniform_(input_tensor, mode=mode)

                    fan_in = input_tensor.size(1)
                    fan_out = input_tensor.size(0)
                    if input_tensor.dim() > 2:
                        fan_in *= input_tensor[0, 0].numel()
                        fan_out *= input_tensor[0, 0].numel()

                    if mode == 'fan_in':
                        n = fan_in
                    else:
                        n = fan_out

                    expected_std = math.sqrt(2.0 / ((1 + a**2) * n))
                    bounds = expected_std * math.sqrt(3.0)
                    assert self._is_uniform(input_tensor, -bounds, bounds)

    @unittest.skipIf(not TEST_SCIPY, "Scipy not found.")
    @skipIfTorchDynamo("scipy.kstest is failing under dynamo")
    def test_kaiming_normal(self):
        for use_a in [True, False]:
            for dims in [2, 4]:
                for mode in ['fan_in', 'fan_out']:
                    input_tensor = self._create_random_nd_tensor(dims, size_min=20, size_max=25)
                    if use_a:
                        a = self._random_float(0.1, 2)
                        init.kaiming_normal_(input_tensor, a=a, mode=mode)
                    else:
                        a = 0
                        init.kaiming_normal_(input_tensor, mode=mode)

                    fan_in = input_tensor.size(1)
                    fan_out = input_tensor.size(0)
                    if input_tensor.dim() > 2:
                        fan_in *= input_tensor[0, 0].numel()
                        fan_out *= input_tensor[0, 0].numel()

                    if mode == 'fan_in':
                        n = fan_in
                    else:
                        n = fan_out

                    expected_std = math.sqrt(2.0 / ((1 + a**2) * n))
                    assert self._is_normal(input_tensor, 0, expected_std)

    def test_sparse_only_works_on_2d_inputs(self):
        for dims in [1, 3]:
            with self.assertRaises(ValueError):
                sparsity = self._random_float(0.1, 0.9)
                tensor = self._create_random_nd_tensor(dims, size_min=1, size_max=3)
                init.sparse_(tensor, sparsity)

    @unittest.skipIf(not TEST_SCIPY, "Scipy not found.")
    @skipIfTorchDynamo("scipy.kstest is failing under dynamo")
    def test_sparse_default_std(self):
        for use_random_std in [True, False]:
            input_tensor = self._create_random_nd_tensor(2, size_min=30, size_max=35)
            rows, cols = input_tensor.size(0), input_tensor.size(1)
            sparsity = self._random_float(0.1, 0.2)

            std = 0.01  # default std
            if use_random_std:
                std = self._random_float(0.01, 0.2)
                init.sparse_(input_tensor, sparsity=sparsity, std=std)
            else:
                init.sparse_(input_tensor, sparsity=sparsity)

            for col_idx in range(input_tensor.size(1)):
                column = input_tensor[:, col_idx]
                assert column[column == 0].nelement() >= math.ceil(sparsity * rows)

            assert self._is_normal(input_tensor[input_tensor != 0], 0, std)

    @skipIfNoLapack
    def test_orthogonal(self):
        for use_gain in [True, False]:
            for tensor_size in [[3, 4], [4, 3], [20, 2, 3, 4], [2, 3, 4, 5]]:
                input_tensor = torch.zeros(tensor_size)
                gain = 1.0

                if use_gain:
                    gain = self._random_float(0.1, 2)
                    init.orthogonal_(input_tensor, gain=gain)
                else:
                    init.orthogonal_(input_tensor)

                rows, cols = tensor_size[0], reduce(mul, tensor_size[1:])
                flattened_tensor = input_tensor.view(rows, cols)
                if rows > cols:
                    self.assertEqual(torch.mm(flattened_tensor.t(), flattened_tensor),
                                     torch.eye(cols) * gain ** 2, atol=1e-6, rtol=0)
                else:
                    self.assertEqual(torch.mm(flattened_tensor, flattened_tensor.t()),
                                     torch.eye(rows) * gain ** 2, atol=1e-6, rtol=0)

    def test_deprecation(self):
        x = torch.randn(3, 3)

        def fn():
            init.normal(x)

        with self.assertWarnsRegex(
            FutureWarning,
            "deprecated",
            msg="methods not suffixed with underscore should be deprecated",
        ):
            fn()



class TestDiracNPU(TestCase):
    """Validate NPU values, parameter boundaries, storage and convolution behavior."""

    def _device(self):
        acc = torch.accelerator.current_accelerator()
        if acc is None or acc.type != "npu" or not torch.accelerator.is_available():
            self.skipTest("NPU is required")
        return torch.device(acc.type, 0)

    def _expected(self, shape, groups, dtype, device):
        channels_per_group = shape[0] // groups
        center = tuple(size // 2 for size in shape[2:])
        values = [
            int(index[0] % channels_per_group == index[1] and index[2:] == center)
            for index in product(*(range(size) for size in shape))
        ]
        return torch.tensor(values, dtype=dtype, device=device).reshape(shape)

    def _check(self, tensor, groups=1, omit_groups=False):
        self.assertEqual(tensor.device.type, "npu")
        pointer = tensor.data_ptr()
        metadata = (tensor.shape, tensor.stride(), tensor.dtype, tensor.device, tensor.requires_grad)
        expected = self._expected(tuple(tensor.shape), groups, tensor.dtype, tensor.device)
        result = init.dirac_(tensor) if omit_groups else init.dirac_(tensor, groups=groups)
        self.assertIs(result, tensor)
        self.assertEqual(result.data_ptr(), pointer)
        self.assertEqual(
            (result.shape, result.stride(), result.dtype, result.device, result.requires_grad),
            metadata,
        )
        self.assertEqual(expected.device.type, "npu")
        if result.is_complex():
            # NPU isclose accepts real components, not complex tensors.
            self.assertEqual(torch.view_as_real(result), torch.view_as_real(expected), atol=0, rtol=0)
        else:
            self.assertEqual(result, expected, atol=0, rtol=0)

    def test_dirac_dtype_groups_and_inplace(self):
        device = self._device()
        dtypes = (
            torch.float16, torch.bfloat16, torch.float32, torch.float64,
            torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64,
            torch.bool, torch.complex64, torch.complex128,
        )
        for dims, groups, dtype in product((3, 4, 5), (1, 2, 3), dtypes):
            with self.subTest(dims=dims, groups=groups, dtype=dtype):
                shape = (groups * 3, 2) + (3,) * (dims - 2)
                tensor = torch.tensor([2] * math.prod(shape), dtype=dtype, device=device).reshape(shape)
                self._check(tensor, groups)

    def test_dirac_default_and_kernel_channel_boundaries(self):
        device = self._device()
        for dims in (3, 4, 5):
            shape = (3, 2) + (1,) * (dims - 2)
            default = torch.full(shape, 2.0, device=device)
            explicit = torch.full_like(default, 2.0)
            self._check(default, omit_groups=True)
            self._check(explicit, groups=1)
            self.assertEqual(default, explicit, atol=0, rtol=0)
            kernels = [(size,) * (dims - 2) for size in (1, 2, 3)]
            if dims > 3:
                kernels.append(tuple(2 + axis for axis in range(dims - 2)))
            for groups, channels, kernel in product(
                (1, 2, 3), ((1, 1), (1, 3), (3, 1), (2, 2)), kernels
            ):
                with self.subTest(dims=dims, groups=groups, channels=channels, kernel=kernel):
                    shape = (groups * channels[0], channels[1]) + kernel
                    self._check(torch.full(shape, 2.0, device=device), groups)
        self._check(torch.full((4, 2, 3), 2.0, device=device), groups=True)

    def test_dirac_noncontiguous_requires_grad(self):
        device = self._device()
        for dims, groups, noncontiguous, requires_grad in product(
            (3, 4, 5), (1, 2, 3), (False, True), (False, True)
        ):
            with self.subTest(
                dims=dims, groups=groups,
                noncontiguous=noncontiguous, requires_grad=requires_grad,
            ):
                shape = (6, 3) + (3,) * (dims - 2)
                if noncontiguous:
                    base_shape = shape[:-1] + (shape[-1] * 2,)
                    base = torch.full(base_shape, 2.0, device=device, requires_grad=requires_grad)
                    tensor = base[..., ::2]
                    self.assertFalse(tensor.is_contiguous())
                else:
                    tensor = torch.full(shape, 2.0, device=device, requires_grad=requires_grad)
                    self.assertTrue(tensor.is_leaf)
                self._check(tensor, groups)

    def test_dirac_empty_channels(self):
        device = self._device()
        for dims, groups, kernel in product((3, 4, 5), (1, 2, 3), (0, 3)):
            for channels in ((0, 2), (groups * 2, 0), (0, 0)):
                shape = channels + (kernel,) * (dims - 2)
                with self.subTest(shape=shape, groups=groups):
                    tensor = torch.empty(shape, device=device)
                    self._check(tensor, groups)
                    self.assertEqual(tensor.numel(), 0)

    def test_dirac_invalid_groups_match_cpu(self):
        device = self._device()
        shape = (4, 2, 3)
        for groups in (3, 8, 2**64, 0, False, None, 1.5, "2", [], {}, (), 1 + 0j, -1, -2, -3):
            with self.subTest(groups=groups):
                reference = torch.full(shape, 2.0, device="cpu")
                tensor = torch.full(shape, 2.0, device=device)
                try:
                    init.dirac_(reference, groups)
                except (TypeError, ValueError, ZeroDivisionError) as error:
                    expected_error = type(error)
                else:
                    self.assertIn(groups, (-1, -2))
                    self.assertEqual(reference, torch.zeros_like(reference), atol=0, rtol=0)
                    self.assertIs(init.dirac_(tensor, groups), tensor)
                    self.assertEqual(tensor, torch.zeros_like(tensor), atol=0, rtol=0)
                    continue
                with self.assertRaises(expected_error):
                    init.dirac_(tensor, groups)

    def test_dirac_invalid_tensor_arguments(self):
        device = self._device()
        shapes = [
            (), (2,), (2, 2), (2, 2, 2, 2, 2, 2),
            (2, 2, 0), (2, 2, 3, 0), (2, 2, 3, 0, 3),
        ]
        for shape in shapes:
            with self.subTest(shape=shape):
                reference = torch.empty(shape, device="cpu")
                tensor = torch.empty(shape, device=device)
                try:
                    init.dirac_(reference)
                except (ValueError, IndexError) as error:
                    expected_error = type(error)
                else:
                    self.fail("Invalid tensor unexpectedly succeeded on CPU")
                with self.assertRaises(expected_error):
                    init.dirac_(tensor)
        for value in (None, 1, 1.5, True, [], (), {}, "tensor"):
            with self.subTest(value=value):
                with self.assertRaises(AttributeError):
                    init.dirac_(value)
        tensor = torch.empty(3, 2, 3, device=device)
        for call in (
            lambda: init.dirac_(), lambda: init.dirac_(tensor, 1, 1),
            lambda: init.dirac_(tensor, unknown=1),
        ):
            with self.assertRaises(TypeError):
                call()

    def test_dirac_convolution_identity_and_input_grad(self):
        device = self._device()
        previous_hf32 = torch.npu.conv.allow_hf32
        try:
            torch.npu.conv.allow_hf32 = False
            for dims, groups, channels in product(
                (3, 4, 5), (1, 2, 3), ((1, 2), (2, 2), (3, 2))
            ):
                with self.subTest(dims=dims, groups=groups, channels=channels):
                    out_channels, in_channels = channels
                    shape = (1, groups * in_channels) + (4,) * (dims - 2)
                    size = math.prod(shape)
                    inputs = (
                        torch.arange(size, device=device, dtype=torch.float32).reshape(shape) / 100
                    ).requires_grad_()
                    weight_shape = (groups * out_channels, in_channels) + (3,) * (dims - 2)
                    weights = torch.full(weight_shape, 2.0, device=device)
                    self._check(weights, groups)
                    conv = (F.conv1d, F.conv2d, F.conv3d)[dims - 3]
                    output = conv(inputs, weights, padding=1, groups=groups)
                    preserved = min(out_channels, in_channels)
                    parts = []
                    for group in range(groups):
                        part = inputs[:, group * in_channels:group * in_channels + preserved]
                        if out_channels > preserved:
                            zeros = inputs.new_zeros((1, out_channels - preserved) + shape[2:])
                            part = torch.cat((part, zeros), dim=1)
                        parts.append(part)
                    expected = torch.cat(parts, dim=1)
                    self.assertEqual(output.device.type, "npu")
                    self.assertEqual(output, expected, atol=1e-5, rtol=1e-5)
                    output.sum().backward()
                    expected_grad = torch.zeros_like(inputs)
                    for group in range(groups):
                        expected_grad[:, group * in_channels:group * in_channels + preserved].fill_(1)
                    self.assertEqual(inputs.grad.device.type, "npu")
                    self.assertEqual(inputs.grad, expected_grad, atol=0, rtol=0)
        finally:
            torch.npu.conv.allow_hf32 = previous_hf32

if __name__ == '__main__':
    run_tests()
