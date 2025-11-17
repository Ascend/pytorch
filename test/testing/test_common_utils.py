import os
from unittest.mock import patch

import numpy as np
import torch

import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import (
    freeze_rng_state,
    iter_indices,
    is_iterable,
    get_npu_device,
    create_common_tensor,
    test_2args_broadcast,
    create_dtype_tensor,
    check_operators_in_prof,
    _create_scaling_case
)


class TestCommonUtils(TestCase):

    def test_iter_indices_zero_dim(self):
        zero_dim_tensor = torch.tensor(5)
        indices = list(iter_indices(zero_dim_tensor))
        self.assertEqual(indices, [])

    def test_create_scaling_case_dtype(self):
        mod_control, mod_scaling, opt_control, opt_scaling, data, loss_fn, skip_iter = _create_scaling_case(
            device="npu", dtype=torch.float
        )
        self.assertIsNotNone(mod_control)
        self.assertIsNotNone(mod_scaling)
        self.assertIsNotNone(opt_control)
        self.assertIsNotNone(opt_scaling)
        self.assertIsNotNone(data)
        self.assertIsNotNone(loss_fn)
        self.assertEqual(skip_iter, 2)

        for input_data, target_data in data:
            self.assertEqual(input_data.dtype, torch.float)
            self.assertEqual(target_data.dtype, torch.float)

    def test_check_operators_in_prof(self):
        class MockProf:
            class MockItem:
                def __init__(self, key):
                    self.key = key

            def key_averages(self):
                return[self.MockItem("add"), self.MockItem("mul")]

        expected = ["add", "mul"]
        result = check_operators_in_prof(expected, MockProf())
        self.assertTrue(result)

        unexpected = ["add", "mul"]
        result = check_operators_in_prof(["add"], MockProf(), unexpected)
        self.assertFalse(result)

    def test_create_dtype_tensor_no_zero(self):
        cpu_input, npu_input = create_dtype_tensor((2, 3), torch.int32, no_zero=True)
        self.assertEqual(cpu_input.shape, (2, 3))
        self.assertEqual(npu_input.shape, (2, 3))

        self.assertFalse(torch.any(cpu_input == 0))
        self.assertFalse(torch.any(npu_input == 0))

    def test_iter_indices_2d(self):
        tensor = torch.tensor([[1, 2], [3, 4]])
        indices = list(iter_indices(tensor))
        self.assertEqual(indices, [(0, 0), (0, 1), (1, 0), (1, 1)])

    def test_get_npu_device_with_env_vat(self):
        with patch.dict(os.environ, {"SET_NPU_DEVICE": "1"}, clear=True):
            device = get_npu_device()
            self.assertEqual(device, "npu:1")

    def test_create_common_tensor(self):
        item = (np.float32, -1, (2, 3))
        cpu_input, npu_input = create_common_tensor(item, -5, 5)

        self.assertIsInstance(cpu_input, torch.Tensor)
        self.assertIsInstance(npu_input, torch.Tensor)
        self.assertEqual(cpu_input.shape, (2, 3))
        self.assertEqual(npu_input.shape, (2, 3))
        self.assertEqual(cpu_input.dtype, torch.float32)
        self.assertEqual(npu_input.dtype, torch.float32)

    def test_is_iterable(self):
        self.assertTrue(is_iterable([1, 2, 3]))
        self.assertTrue(is_iterable((1, 2, 3)))
        self.assertTrue(is_iterable({1, 2, 3}))
        self.assertTrue(is_iterable("hello"))
        self.assertFalse(is_iterable(42))
        self.assertFalse(is_iterable(None))

    def test_iter_indices_1d(self):
        tensor_1d = torch.tensor([1, 2, 3, 4])
        indices = list(iter_indices(tensor_1d))
        self.assertEqual(indices, [0, 1, 2, 3])

    def test_torch_manual_seed_seeds_npu_devices(self):
        with freeze_rng_state():
            x = torch.zeros(4, 4).float()
            torch.manual_seed(2)
            self.assertEqual(torch_npu.npu.initial_seed(), 2)
            x.uniform_()
            torch.manual_seed(2)
            y = x.clone().uniform_()
            self.assertEqual(x, y)
            self.assertEqual((torch_npu.npu.initial_seed()), 2)

    def test_manual_seed(self):
        with freeze_rng_state():
            x = torch.zeros(4, 4).float()
            torch_npu.npu.manual_seed(2)
            torch.manual_seed(2)
            self.assertEqual(torch_npu.npu.initial_seed(), 2)
            x.uniform_()
            a = torch.bernoulli(torch.full_like(x, 0.5))
            torch.manual_seed(2)
            y = x.clone().uniform_()
            b = torch.bernoulli(torch.full_like(x, 0.5))
            self.assertEqual(x, y)
            self.assertEqual(a, b)
            self.assertEqual(torch_npu.npu.initial_seed(), 2)

    def test_get_set_rng_state(self):
        with freeze_rng_state():
            torch.manual_seed(3)
            cpu_state = torch.get_rng_state()
            npu_state = torch_npu.npu.get_rng_state()
            self.assertEqual(int(cpu_state[0]), 3)
            self.assertEqual(cpu_state[0], npu_state[0])
            torch_npu.npu.manual_seed(2)
            cpu_state_new = torch.get_rng_state()
            npu_state = torch_npu.npu.get_rng_state()
            self.assertEqual(cpu_state, cpu_state_new)
            self.assertEqual(int(npu_state[0]), 2)

    def test_create_dtype_tensor_with_format(self):
        cpu_input, npu_input = create_dtype_tensor((2, 3), torch.float, npu_format=2)
        self.assertEqual(cpu_input.shape, (2, 3))
        self.assertEqual(npu_input.shape, (2, 3))
        self.assertEqual(cpu_input.dtype, torch.float)
        self.assertEqual(npu_input.dtype, torch.float)

    def test_create_dtype_tensor_bool(self):
        cpu_input, npu_input = create_dtype_tensor((2, 3), torch.bool)
        self.assertEqual(cpu_input.shape, (2, 3))
        self.assertEqual(npu_input.shape, (2, 3))
        self.assertEqual(cpu_input.dtype, torch.bool)
        self.assertEqual(npu_input.dtype, torch.bool)
        self.assertTrue(torch.isfinite(cpu_input).all())

    def test_check_operators_in_prof_empty_prof(self):
        class MockEmptyProf:
            def key_averages(self):
                return []
        expected = ["add"]
        result = check_operators_in_prof(expected, MockEmptyProf())
        self.assertFalse(result)

    def test_create_dtype_tensor_different_dtypes(self):
        cpu_input, npu_input = create_dtype_tensor((2, 3), torch.half)
        self.assertEqual(cpu_input.dtype, torch.float16)
        self.assertEqual(npu_input.dtype, torch.float16)

        cpu_input, npu_input = create_dtype_tensor((2, 3), torch.int32)
        self.assertEqual(cpu_input.dtype, torch.int32)
        self.assertEqual(npu_input.dtype, torch.int32)

        cpu_input, npu_input = create_dtype_tensor((2, 3), torch.float32)
        self.assertEqual(cpu_input.dtype, torch.float32)
        self.assertEqual(npu_input.dtype, torch.float32)

    def test_2args_broadcast(self):
        def add_fn(x, y):
            result = test_2args_broadcast()
            self.assertEqual(len(result), 2)

            for cpu_out, npu_out in result:
                self.assertIsNotNone(cpu_out)
                self.assertIsNotNone(npu_out)


if __name__ == "__main__":
    run_tests()