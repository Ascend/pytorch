from unittest import skipIf
import torch

from torch.testing._internal.common_device_type import dtypes, instantiate_device_type_tests
from torch.testing._internal.common_dtype import all_types_and_complex_and, all_types_and, custom_types
from torch.testing._internal.common_utils import DeterministicGuard, TestCase, run_tests

import torch_npu


DEVICE_NAME = torch_npu.npu.get_device_name(0)

device_is_910A = False
if "Ascend910A" in DEVICE_NAME or "Ascend910P" in DEVICE_NAME:
    device_is_910A = True


class TorchFillUninitializedMemoryTestCase(TestCase):
    @dtypes(*(all_types_and_complex_and(torch.half, torch.bool, torch.bfloat16) if not device_is_910A
              else all_types_and(torch.half, torch.bool)))
    def test_deterministic_resize(self, device, dtype):
        test_cases = [
            # size, stride, resize_size
            ((10,), (1,), (5,)),
            ((10,), (0,), (10,)),
            ((10,), (1,), (20,)),
            ((2, 3, 4), None, (2, 3, 4)),
            ((2, 3, 4), None, (6, 3, 4)),
            ((2, 3, 4), None, (2, 5, 4)),
            ((2, 3, 4), None, (2, 3, 6)),
            ((2, 3, 4), None, (3, 4, 5)),
            ((2, 3, 4), (1, 4, 12), (2, 3, 4)),
            ((2, 3, 4), (1, 4, 12), (4, 3, 4)),
            ((2, 3, 4), (1, 4, 12), (2, 4, 4)),
            ((2, 3, 4), (1, 4, 12), (2, 3, 5)),
            ((2, 3, 4), (1, 4, 12), (3, 4, 5)),
            ((2, 3, 4), (1, 0, 1), (2, 4, 5)),
        ]

        for size, stride, resize_size in test_cases:
            if stride is None:
                a = torch.zeros(size, dtype=dtype, device=device)
            else:
                a = torch.empty_strided(size, stride, dtype=dtype, device=device).fill_(0)
            old_storage = a.untyped_storage().clone()
            with DeterministicGuard(True, fill_uninitialized_memory=True):
                a.resize_(resize_size)

            new_storage = a.untyped_storage()

            # If storage size was increased, check that the new section is
            # filled with NaN/MAX_INT. Otherwise, check that the storages are
            # equal.
            old_tensor = torch.tensor(old_storage, dtype=dtype)
            old_numel = old_tensor.numel()
            new_tensor = torch.tensor(new_storage, dtype=dtype)
            new_numel = new_tensor.numel()

            if new_numel > old_numel:
                self.assertEqual(new_tensor[:old_numel], old_tensor)
                fill_section = new_tensor[old_numel:]

                if dtype.is_floating_point or dtype.is_complex:
                    self.assertTrue(fill_section.isnan().all())
                else:
                    if dtype == torch.bool:
                        max_val = True
                    else:
                        max_val = torch.iinfo(dtype).max
                    self.assertTrue(fill_section.eq(max_val).all())
            else:
                self.assertEqual(old_tensor, new_tensor)

    # When deterministic algorithms are enabled, `torch.empty` should fill floating
    # point tensors with NaN and integer tensors with MAX_INT
    @dtypes(*(all_types_and_complex_and(torch.half, torch.bool, torch.bfloat16) if not device_is_910A
              else all_types_and(torch.half, torch.bool)))
    def test_deterministic_empty(self, device, dtype):
        gen_fns = [
            lambda: torch.empty(10, 9, device=device, dtype=dtype),
            lambda: torch.empty(10, 9, out=torch.zeros(1, device=device, dtype=dtype)),
            lambda: torch.empty_like(torch.zeros(10, 9, device=device, dtype=dtype)),
            lambda: torch.empty_like(torch.zeros(10, 9, device=device, dtype=dtype), memory_format=torch.contiguous_format),
            lambda: torch.empty_strided((10, 9), (1, 5), device=device, dtype=dtype),
            lambda: torch.empty_permuted((2, 3, 5), (1, 0, 2), device=device, dtype=dtype),
            lambda: torch_npu.empty_with_format((10, 9), device=device, dtype=dtype),
        ]

        for gen_fn in gen_fns:
            with DeterministicGuard(True, fill_uninitialized_memory=True):
                res = gen_fn()

            if dtype.is_floating_point or dtype.is_complex:
                self.assertTrue(res.isnan().all())
            else:
                if dtype == torch.bool:
                    max_val = True
                else:
                    max_val = torch.iinfo(dtype).max
                self.assertTrue(res.eq(max_val).all())

    # When deterministic algorithms are enabled, `torch.empty` should fill floating
    # point tensors with NaN and integer tensors with MAX_INT
    @skipIf(device_is_910A, "This soc is not supported")
    @dtypes(*(custom_types(torch.half, torch.bfloat16, torch.float32, torch.bool, torch.int8, torch.int32, torch.int64)))
    def test_deterministic_empty_with_swapped_memory(self, device, dtype):
        gen_fns = [
            lambda: torch_npu.empty_with_swapped_memory((10, 9), device=device, dtype=dtype),
        ]

        for gen_fn in gen_fns:
            with DeterministicGuard(True, fill_uninitialized_memory=True):
                res = gen_fn()

            if dtype.is_floating_point or dtype.is_complex:
                self.assertTrue(res.isnan().all())
            else:
                if dtype == torch.bool:
                    max_val = True
                else:
                    max_val = torch.iinfo(dtype).max
                self.assertTrue(res.eq(max_val).all())


instantiate_device_type_tests(TorchFillUninitializedMemoryTestCase, globals(), only_for='privateuse1')


if __name__ == '__main__':
    run_tests()
