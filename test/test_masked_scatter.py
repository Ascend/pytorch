# Owner(s): ["module: tests"]
import torch
import torch_npu.testing
import torch.utils.data
from torch.testing._internal.common_utils import run_tests, TestCase, skipIfMPS
from torch.testing._internal.common_device_type import (
    instantiate_device_type_tests, onlyNativeDeviceTypes, dtypes,
    onlyPRIVATEUSE1, largeTensorTest, expectedFailureMeta)
from torch.testing._internal.common_dtype import all_types_and_complex_and, all_types_and

DEVICE_NAME = torch_npu.npu.get_device_name(0)

device_is_910A = False
if "Ascend910A" in DEVICE_NAME or "Ascend910P" in DEVICE_NAME:
    device_is_910A = True

if device_is_910A:
    all_types_and_complex_and = all_types_and

class TestTorchDeviceType(TestCase):
    @onlyNativeDeviceTypes
    @dtypes(*(all_types_and_complex_and(torch.half, torch.bfloat16) if not device_is_910A
            else all_types_and(torch.half)))
    def test_masked_scatter(self, device, dtype):
        dt = dtype
        num_copy, num_dest = 3, 10
        dest = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=dt, device=device)
        dest2 = dest.clone()
        dest_ones = dest.clone()
        dest_ones_expected = dest.clone()
        src = torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=dt, device=device)
        src_ones = torch.tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=dt, device=device)
        mask = torch.tensor((0, 0, 0, 0, 1, 0, 1, 0, 1, 0), dtype=torch.bool, device=device)

        dest.masked_scatter_(mask, src)
        j = 0
        for i in range(num_dest):
            if mask[i]:
                dest2[i] = src[j]
                dest_ones_expected[i] = src_ones[j]
                j += 1
        self.assertEqual(dest, dest2, atol=0, rtol=0)

        dest_ones.masked_scatter_(mask, src_ones)
        self.assertEqual(dest_ones, dest_ones_expected, atol=0, rtol=0)

        # Bound checking in NPU is done inside a kernel
        # in order to avoid synchronization, but this means
        # we can not clear the failures. So there is no way
        # to test it then recover.
        if self.device_type != 'npu':
            # make src smaller. this should fail
            src = torch.zeros(num_copy - 1, dtype=dt, device=device)
            with self.assertRaises(RuntimeError):
                dest.masked_scatter_(mask, src)

        # empty tensor
        dest = torch.empty((5, 0, 5), dtype=dt, device=device)
        mask = torch.ones_like(dest, dtype=torch.bool, device=device)
        src = torch.empty((0,), dtype=dt, device=device)
        dest.masked_scatter_(mask, src)

        dest = torch.empty((5, 0, 5), dtype=dt, device=device)
        mask = torch.ones((5, 1, 5), dtype=torch.bool, device=device)
        src = torch.empty((0,), dtype=dt, device=device)
        dest.masked_scatter_(mask, src)

    @skipIfMPS
    def test_masked_scatter_bool_tensor(self, device):
        src = torch.tensor([True, True, True], device=device)
        dst = torch.tensor([False, False, False], device=device)
        mask = torch.tensor([False, True, False], device=device)

        dst.masked_scatter_(mask, src)
        self.assertEqual(dst, torch.tensor([False, True, False], device=device))

        mask = torch.tensor([True, False, True], device=device)
        dst = dst.masked_scatter(mask, src)
        self.assertEqual(dst, torch.tensor([True, True, True], device=device))

    @onlyPRIVATEUSE1
    @largeTensorTest('30GB')
    def test_masked_scatter_large_tensor(self, device):
        t_cpu = torch.empty(2**31 + 1, dtype=torch.bool).random_()
        t = t_cpu.to(device)
        result_cpu = t_cpu.masked_scatter(t_cpu, t_cpu)
        result = t.masked_scatter(t, t)
        self.assertEqual(result, result_cpu)

    # (but have to extend ErrorInputs to handle inplace-only errors!)
    @expectedFailureMeta  # RuntimeError not raised
    @onlyNativeDeviceTypes
    def test_masked_scatter_mem_overlap(self, device):
        x = torch.rand((1,), device=device).expand((6,))
        src = torch.rand((3,), device=device)
        mask = torch.tensor([True, False, True, True, False, False], device=device)

        with self.assertRaisesRegex(RuntimeError, 'unsupported operation'):
            x.masked_scatter_(mask, src)

    @onlyNativeDeviceTypes
    def test_masked_scatter_inplace_noncontiguous(self, device):
        t = torch.zeros(5, 2, dtype=torch.long, device=device)
        t_non_contig = t.transpose(0, 1)
        t_contig = t_non_contig.contiguous()

        assert t_contig.is_contiguous()
        assert not t_non_contig.is_contiguous()

        mask = torch.tensor([[False, True], [False, True], [False, False], [True, True], [True, True]], device=device)
        mask_non_contig = mask.transpose(0, 1)
        mask_contig = mask_non_contig.contiguous()

        assert mask_contig.is_contiguous()
        assert not mask_non_contig.is_contiguous()

        # source is always converted to contiguous by the op.
        source = torch.tensor([[1, 2, 3, 4, 5], [6, 7, 8, 9, 9]], device=device)

        # t: contig, mask: contig
        expected = t_contig.masked_scatter_(mask_contig, source)

        # t: non-contig, mask: non-contig
        actual = t_non_contig.masked_scatter_(mask_non_contig, source)
        self.assertEqual(actual, expected)

        # t: contig, mask: non-contig
        actual = t_contig.masked_scatter_(mask_non_contig, source)
        self.assertEqual(actual, expected)

        # t: non-contig, mask: contig
        actual = t_non_contig.masked_scatter_(mask_contig, source)
        self.assertEqual(actual, expected)

instantiate_device_type_tests(TestTorchDeviceType, globals(), only_for='privateuse1')

if __name__ == "__main__":
    torch.npu.config.allow_internal_format = False
    torch.npu.set_compile_mode(jit_compile=False)
    TestCase._default_dtype_check_enabled = True
    run_tests()