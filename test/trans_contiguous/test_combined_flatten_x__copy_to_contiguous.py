import os
import torch
import numpy as np
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor, check_operators_in_prof

os.environ["COMBINED_ENABLE"] = "1"  # Open combined-view cases optimization


# Note: NPU only support trans-contiguous with base format, so format_list uses -1


class CombinedFlattenXCopyToContiguous(TestCase):
    def test_flatten_select_copy_contiguous(self, device="npu"):
        dtype_list1 = [np.float16, np.float32]
        format_list1 = [-1]
        shape_list1 = [
                      [20, 30, 40, 16],
        ]
        shape_format1 = [
            [i, j, k] for i in dtype_list1 for j in format_list1 for k in shape_list1
        ]

        for item in shape_format1:
            cpu_input, npu_input = create_common_tensor(item, 0, 100)
            # case 1: flatten+select
            with torch.autograd.profiler.profile(use_device='npu') as prof:
                npu_out1 = npu_input.flatten(2).select(1, 1).contiguous()
            self.assertEqual(check_operators_in_prof(['contiguous_h_match', 'contiguous_d_StridedSlice'], prof)
                             or check_operators_in_prof(['aclnnInplaceCopy'], prof),
                             True, message="Error operators called!")
            cpu_out1 = cpu_input.flatten(2).select(1, 1).contiguous()
            self.assertRtolEqual(npu_out1.to("cpu").numpy(), cpu_out1.numpy())
            # case 2: select+flatten == can be optimized as single select(contiguous_h_combined should not be called)
            with torch.autograd.profiler.profile(use_device='npu') as prof:
                npu_out2 = npu_input.select(2, 1).flatten(1).contiguous()
            self.assertEqual(check_operators_in_prof(['contiguous_d_StridedSlice'], prof, ['contiguous_h_combined'])
                             or check_operators_in_prof(['aclnnInplaceCopy'], prof, ['contiguous_h_combined']),
                             True, message="Error operators called!")
            cpu_out2 = cpu_input.select(2, 1).flatten(1).contiguous()
            self.assertRtolEqual(npu_out2.to("cpu").numpy(), cpu_out2.numpy())

    def test_flatten_strideslice_copy_contiguous(self, device="npu"):
        dtype_list2 = [np.float16, np.float32]
        format_list2 = [-1]
        shape_list2 = [
                      [20, 30, 40, 16],
        ]
        shape_format2 = [
            [i, j, k] for i in dtype_list2 for j in format_list2 for k in shape_list2
        ]

        for item in shape_format2:
            cpu_input, npu_input = create_common_tensor(item, 0, 100)
            # case 1: flatten+strideslice ==> can be optimized as slice(contiguous with offset) + select
            with torch.autograd.profiler.profile(use_device='npu') as prof:
                npu_out1 = npu_input.flatten()[2:100:10].contiguous()
            self.assertEqual(check_operators_in_prof(['contiguous_d_Reshape', 'contiguous_d_StridedSlice'], prof)
                             or check_operators_in_prof(['aclnnInplaceCopy'], prof),
                             True, message="Error operators called!")
            cpu_out1 = cpu_input.flatten()[2:100:10].contiguous()
            self.assertRtolEqual(npu_out1.to("cpu").numpy(), cpu_out1.numpy())
            # case 2: strideslice+flatten==> can be optimized as single strideslice
            # (contiguous_h_combined should not be called)
            with torch.autograd.profiler.profile(use_device='npu') as prof:
                npu_out2 = npu_input[:, 2:20:3].flatten().contiguous()
            self.assertEqual(check_operators_in_prof(['contiguous_d_StridedSlice'], prof, ['contiguous_h_combined'])
                             or check_operators_in_prof(['aclnnInplaceCopy'], prof, ['contiguous_h_combined']),
                             True, message="Error operators called!")
            cpu_out2 = cpu_input[:, 2:20:3].flatten().contiguous()
            self.assertRtolEqual(npu_out2.to("cpu").numpy(), cpu_out2.numpy())


if __name__ == "__main__":
    run_tests()
