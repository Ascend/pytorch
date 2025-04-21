import os
import torch
import numpy as np
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor, check_operators_in_prof

os.environ["COMBINED_ENABLE"] = "1"  # Open combined-view cases optimization


# Note: NPU only support trans-contiguous with base format, so format_list uses -1


class CombinedReshapeXCopyToContiguous(TestCase):
    def test_view_permute_copy_contiguous(self, device="npu"):
        dtype_list1 = [np.float16, np.float32]
        format_list1 = [-1]
        shape_list1 = [
                      [200, 30, 40, 16],
        ]
        shape_format1 = [
            [i, j, k] for i in dtype_list1 for j in format_list1 for k in shape_list1
        ]

        for item in shape_format1:
            cpu_input, npu_input = create_common_tensor(item, 0, 100)
            # case 1: view+permute
            with torch.autograd.profiler.profile(use_device='npu') as prof:
                npu_out1 = npu_input \
                    .view(npu_input.size(0) * npu_input.size(1), npu_input.size(2), npu_input.size(3)) \
                    .transpose(0, 1) \
                    .contiguous()
            self.assertEqual(check_operators_in_prof(['contiguous_d_Transpose'], prof) or
                             check_operators_in_prof(['aclnnInplaceCopy'], prof),
                             True, message="Error operators called!")
            cpu_out1 = cpu_input \
                .view(cpu_input.size(0) * cpu_input.size(1), cpu_input.size(2), cpu_input.size(3)) \
                .transpose(0, 1) \
                .contiguous()
            self.assertRtolEqual(npu_out1.to("cpu").numpy(), cpu_out1.numpy())

            # case 2: permute+view
            with torch.autograd.profiler.profile(use_device='npu') as prof:
                npu_out2 = npu_input \
                    .permute(1, 0, 2, 3) \
                    .view(npu_input.size(1), npu_input.size(0), npu_input.size(2) * npu_input.size(3)) \
                    .contiguous()
            self.assertEqual(check_operators_in_prof(['contiguous_d_Transpose'], prof) or
                             check_operators_in_prof(['aclnnInplaceCopy'], prof),
                             True, message="Error operators called!")
            cpu_out2 = cpu_input \
                .permute(1, 0, 2, 3) \
                .view(cpu_input.size(1), cpu_input.size(0), cpu_input.size(2) * cpu_input.size(3)) \
                .contiguous()
            self.assertRtolEqual(npu_out2.to("cpu").numpy(), cpu_out2.numpy())

    def test_view_select_copy_contiguous(self, device="npu"):
        dtype_list2 = [np.float16, np.float32]
        format_list2 = [-1]
        shape_list2 = [
                      [2, 3, 4, 5],
        ]
        shape_format2 = [
            [i, j, k] for i in dtype_list2 for j in format_list2 for k in shape_list2
        ]

        for item in shape_format2:
            cpu_input, npu_input = create_common_tensor(item, 0, 100)
            # case 1: view+select
            with torch.autograd.profiler.profile(use_device='npu') as prof:
                npu_out1 = npu_input \
                    .view(npu_input.size(0), npu_input.size(1) * npu_input.size(2), npu_input.size(3)) \
                    .select(2, 1) \
                    .contiguous()
            self.assertEqual(check_operators_in_prof(['contiguous_h_match', 'contiguous_d_StridedSlice'], prof) or
                             check_operators_in_prof(['aclnnInplaceCopy'], prof),
                             True, message="Error operators called!")
            cpu_out1 = cpu_input \
                .view(npu_input.size(0), npu_input.size(1) * npu_input.size(2), npu_input.size(3)) \
                .select(2, 1) \
                .contiguous()
            self.assertRtolEqual(npu_out1.to("cpu").numpy(), cpu_out1.numpy())
            # case 2: select+view ==> can be optimized as reshape+narrow
            with torch.autograd.profiler.profile(use_device='npu') as prof:
                npu_out2 = npu_input.select(2, 1).view(npu_input.size(1), npu_input.size(0), -1).contiguous()
            self.assertEqual(check_operators_in_prof(['contiguous_h_match', 'contiguous_d_Slice'], prof) or
                             check_operators_in_prof(['aclnnInplaceCopy'], prof),
                             True, message="Error operators called!")
            cpu_out2 = cpu_input.select(2, 1).view(npu_input.size(1), npu_input.size(0), -1).contiguous()
            self.assertRtolEqual(npu_out2.to("cpu").numpy(), cpu_out2.numpy())

    def test_view_narrow_copy_contiguous(self, device="npu"):
        dtype_list3 = [np.float16, np.float32]
        format_list3 = [-1]
        shape_list3 = [
                      [20, 30, 40, 16],
        ]
        shape_format3 = [
            [i, j, k] for i in dtype_list3 for j in format_list3 for k in shape_list3
        ]

        for item in shape_format3:
            cpu_input, npu_input = create_common_tensor(item, 0, 100)
            # case 1: view + narrow
            with torch.autograd.profiler.profile(use_device='npu') as prof:
                npu_out1 = npu_input.view(20, 1200, 16)[:, 20:150, :].contiguous()
            self.assertEqual(check_operators_in_prof(['contiguous_h_match', 'contiguous_d_Slice'], prof) or
                             check_operators_in_prof(['aclnnInplaceCopy'], prof),
                             True, message="Error operators called!")
            cpu_out1 = cpu_input.view(20, 1200, 16)[:, 20:150, :].contiguous()
            self.assertRtolEqual(npu_out1.to("cpu").numpy(), cpu_out1.numpy())
            # case 2: narrow + view
            with torch.autograd.profiler.profile(use_device='npu') as prof:
                npu_out2 = npu_input[:, 10:19, :, :].view(20, 360, 16).contiguous()
            self.assertEqual(check_operators_in_prof(['contiguous_h_match', 'contiguous_d_Slice'], prof) or
                             check_operators_in_prof(['aclnnInplaceCopy'], prof),
                             True, message="Error operators called!")
            cpu_out2 = cpu_input[:, 10:19, :, :].view(20, 360, 16).contiguous()
            self.assertRtolEqual(npu_out2.to("cpu").numpy(), cpu_out2.numpy())

    def test_view_strideslice_copy_contiguous(self, device="npu"):
        dtype_list4 = [np.float16, np.float32]
        format_list4 = [-1]
        shape_list4 = [
                      [20, 30, 40, 10],
        ]
        shape_format4 = [
            [i, j, k] for i in dtype_list4 for j in format_list4 for k in shape_list4
        ]

        for item in shape_format4:
            cpu_input, npu_input = create_common_tensor(item, 0, 100)
            # case 1: view + strideslice
            with torch.autograd.profiler.profile(use_device='npu') as prof:
                npu_out1 = npu_input.view(20, 1200, 10)[:, 20:150:3, :].contiguous()
            self.assertEqual(check_operators_in_prof(['contiguous_h_match', 'contiguous_d_Slice'], prof) or
                             check_operators_in_prof(['aclnnInplaceCopy'], prof),
                             True, message="Error operators called!")
            cpu_out1 = cpu_input.view(20, 1200, 10)[:, 20:150:3, :].contiguous()
            self.assertRtolEqual(npu_out1.to("cpu").numpy(), cpu_out1.numpy())
            # case 2: strideslice + view
            with torch.autograd.profiler.profile(use_device='npu') as prof:
                npu_out2 = npu_input[10:19:3, :, :].view(3, 2400, 5).contiguous()
            self.assertEqual(check_operators_in_prof(['contiguous_d_AsStrided'], prof) or
                             check_operators_in_prof(['aclnnInplaceCopy'], prof),
                             True, message="Error operators called!")
            cpu_out2 = cpu_input[10:19:3, :, :].view(3, 2400, 5).contiguous()
            self.assertRtolEqual(npu_out2.to("cpu").numpy(), cpu_out2.numpy())


if __name__ == "__main__":
    run_tests()
