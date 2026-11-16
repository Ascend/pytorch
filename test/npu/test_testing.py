import itertools
import torch
import numpy as np

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_dtype_tensor
from torch_npu.testing.decorator import Dtypes, Formats, instantiate_tests

device_type = (
    acc.type
    if (acc := torch.accelerator.current_accelerator())
    else "cpu"
)


# For testing TestCase methods and torch_npu.testing functions
@instantiate_tests
class TestTesting(TestCase):

    # Ensure that assertTensorSlowEqual handles npu arrays properly
    @Dtypes(torch.int32, torch.bool, torch.half, torch.float)
    @Formats(0, 3, 4)
    def test_assert_tensor_slow_equal(self, device, dtype, npu_format):
        test_sizes = [
            (),
            (0,),
            (5,),
            (5, 5),
            (0, 5),
            (5, 0),
        ]
        for test_size in test_sizes:
            a_cpu, a_npu = create_dtype_tensor(test_size, dtype, npu_format, device=device)
            msg = f'Device: {device} Size: {test_size} Dtype: {dtype} Npu_format: {npu_format}'
            self.assertTensorsSlowEqual(a_cpu, a_npu, prec=1e-3, message=msg)
            self.assertTensorsSlowEqual(a_npu, a_cpu, prec=1e-3, message=msg)
            self.assertTensorsSlowEqual(a_cpu, a_cpu, prec=1e-3, message=msg)

    # Ensure that assertRtolEqual handles npu arrays properly
    @Dtypes(torch.int32, torch.bool, torch.half, torch.float)
    @Formats(0, 3, 4)
    def test_assert_rtol_equal(self, device, dtype, npu_format):
        test_sizes = [
            (),
            (0,),
            (6,),
            (6, 6),
            (0, 6),
            (6, 0),
        ]
        for test_size in test_sizes:
            a_cpu, a_npu = create_dtype_tensor(test_size, dtype, npu_format, device=device)
            msg = f'Device: {device} Size: {test_size} Dtype: {dtype} Npu_format: {npu_format}'
            self.assertRtolEqual(a_cpu, a_npu.cpu())
            self.assertRtolEqual(a_npu.cpu(), a_cpu)
            self.assertRtolEqual(a_cpu, a_cpu)

    # Ensure that assertEqual handles npu arrays properly
    @Dtypes(torch.int32, torch.bool, torch.half, torch.float)
    @Formats(0, 3, 4)
    def test_assert_equal(self, device, dtype, npu_format):
        test_sizes = [
            (),
            (0,),
            (7,),
            (7, 7),
            (0, 7),
            (7, 0),
        ]
        for test_size in test_sizes:
            a_cpu, a_npu = create_dtype_tensor(test_size, dtype, npu_format, device=device)
            msg = f'Device: {device} Size: {test_size} Dtype: {dtype} Npu_format: {npu_format}'
            self.assertEqual(a_cpu, a_npu, message=msg)
            self.assertEqual(a_npu, a_cpu, message=msg)
            self.assertEqual(a_cpu, a_cpu, message=msg)

    # Ensure that assertAlmostEqual handles npu arrays properly
    @Dtypes(torch.int32, torch.bool, torch.half, torch.float)
    @Formats(0, 3, 4)
    def test_assert_almost_equal(self, device, dtype, npu_format):
        test_sizes = [
            (),
            (0,),
            (8,),
            (8, 8),
            (0, 8),
            (8, 0),
        ]
        for test_size in test_sizes:
            a_cpu, a_npu = create_dtype_tensor(test_size, dtype, npu_format, device=device)
            msg = f'Device: {device} Size: {test_size} Dtype: {dtype} Npu_format: {npu_format}'
            self.assertAlmostEqual(a_cpu, a_npu, msg=msg)
            self.assertAlmostEqual(a_npu, a_cpu, msg=msg)
            self.assertAlmostEqual(a_cpu, a_cpu, msg=msg)

    # Ensure that assertNotEqual handles npu arrays properly
    @Dtypes(torch.int32, torch.bool, torch.float)
    @Formats(0, 3, 4)
    def test_assert_not_equal(self, device, dtype, npu_format):
        test_sizes = [
            (),
            (9,),
            (9, 9),
        ]
        for test_size in test_sizes:
            if dtype == torch.bool:
                a_cpu = torch.from_numpy(np.zeros(test_size, bool))
                a_npu = a_cpu.to(device)
                b_cpu = torch.from_numpy(np.ones(test_size, bool))
                b_npu = b_cpu.to(device)
            else:
                a_cpu, a_npu = create_dtype_tensor(test_size, dtype, npu_format,
                                                   min_value=5, max_value=10, device=device)
                b_cpu, b_npu = create_dtype_tensor(test_size, dtype, npu_format,
                                                   min_value=-10, max_value=-5, device=device)
            msg = f'Device: {device} Size: {test_size} Dtype: {dtype} Npu_format: {npu_format}'
            self.assertNotEqual(a_cpu, b_cpu, message=msg)
            self.assertNotEqual(a_cpu, b_npu, message=msg)
            self.assertNotEqual(a_npu, b_cpu, message=msg)
            self.assertNotEqual(a_npu, b_npu, message=msg)

    # Validate torch.testing.assert_allclose directly on NPU tensors.
    def test_assert_allclose_default(self):
        actual = torch.tensor([1.0, 2.0, 3.0]).to(device_type)
        expected = torch.tensor([1.0, 2.0, 3.0]).to(device_type)

        torch.testing.assert_allclose(actual, expected)
        torch.testing.assert_allclose(
            actual,
            expected,
            rtol=None,
            atol=None,
        )

        torch.testing.assert_allclose(
            torch.tensor([1.0]).to(device_type),
            torch.tensor([1.00005]).to(device_type),
        )

        with self.assertRaises(AssertionError):
            torch.testing.assert_allclose(
                torch.tensor([1.0]).to(device_type),
                torch.tensor([1.01]).to(device_type),
            )

    def test_assert_allclose_tolerance(self):
        actual = torch.tensor([1.0, 2.0, 3.0]).to(device_type)
        expected = torch.tensor([1.001, 2.001, 3.001]).to(device_type)

        torch.testing.assert_allclose(
            actual,
            expected,
            rtol=1e-2,
            atol=1e-3,
        )

    def test_assert_allclose_equal_nan(self):
        actual = torch.tensor([1.0, float("nan"), 3.0]).to(device_type)
        expected = torch.tensor([1.0, float("nan"), 3.0]).to(device_type)

        torch.testing.assert_allclose(actual, expected)
        torch.testing.assert_allclose(
            actual,
            expected,
            equal_nan=True,
        )

        with self.assertRaises(AssertionError):
            torch.testing.assert_allclose(
                actual,
                expected,
                equal_nan=False,
            )

    def test_assert_allclose_dtype(self):
        actual = torch.tensor(
            [1.0, 2.0, 3.0],
            dtype=torch.float16,
        ).to(device_type)
        expected = torch.tensor(
            [1.0, 2.0, 3.0],
            dtype=torch.float32,
        ).to(device_type)

        torch.testing.assert_allclose(
            actual,
            expected,
            rtol=0.0,
            atol=0.0,
        )

    def test_assert_allclose_failure_message(self):
        actual = torch.tensor([1.0, 2.0, 3.0]).to(device_type)
        expected = torch.tensor([1.0, 2.0, 4.0]).to(device_type)

        with self.assertRaisesRegex(
            AssertionError,
            "assert_allclose mismatch",
        ):
            torch.testing.assert_allclose(
                actual,
                expected,
                rtol=0.0,
                atol=0.0,
                msg="assert_allclose mismatch",
            )



    def test_assert_allclose_non_tensor_inputs(self):
        # Validate the compatibility conversion path for non-Tensor inputs.
        torch.testing.assert_allclose(
            [1.0, 2.0, 3.0],
            [1.0, 2.0, 3.0],
        )
        torch.testing.assert_allclose(1, 1)

    def test_assert_allclose_partial_tolerance(self):
        actual = torch.tensor(
            [1.0, 2.0, 3.0]
        ).to(device_type)
        expected = torch.tensor(
            [1.0, 2.0, 3.0]
        ).to(device_type)

        with self.assertRaises(ValueError):
            torch.testing.assert_allclose(
                actual,
                expected,
                rtol=1e-5,
                atol=None,
            )

        with self.assertRaises(ValueError):
            torch.testing.assert_allclose(
                actual,
                expected,
                rtol=None,
                atol=1e-8,
            )

    def test_assert_allclose_shape_mismatch(self):
        actual = torch.tensor([1.0, 2.0]).to(device_type)
        expected = torch.tensor([[1.0, 2.0]]).to(device_type)

        with self.assertRaisesRegex(
            AssertionError,
            "shape",
        ):
            torch.testing.assert_allclose(
                actual,
                expected,
            )

    def test_assert_allclose_device_mismatch(self):
        actual = torch.tensor(
            [1.0, 2.0, 3.0]
        ).to(device_type)

        # Keep expected on CPU to validate device mismatch handling.
        expected = torch.tensor(
            [1.0, 2.0, 3.0]
        )

        with self.assertRaises(AssertionError):
            torch.testing.assert_allclose(
                actual,
                expected,
                rtol=0.0,
                atol=0.0,
            )


if __name__ == '__main__':
    run_tests()
