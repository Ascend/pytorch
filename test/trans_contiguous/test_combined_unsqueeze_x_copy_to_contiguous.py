import os
import torch
import numpy as np
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor, check_operators_in_prof

os.environ["COMBINED_ENABLE"] = "1"  # Open combined-view cases optimization


# Note: NPU only support trans-contiguous with base format, so format_list uses -1


class CombinedUnsqueezeXCopyToContiguous(TestCase):
    def test_unsqueeze_permute_copy_contiguous(self, device="npu"):
        dtype_list1 = [np.float16, np.float32]
        format_list1 = [-1]
        shape_list1 = [
                      [2, 3, 4, 5],
        ]
        shape_format1 = [
            [i, j, k] for i in dtype_list1 for j in format_list1 for k in shape_list1
        ]

        for item in shape_format1:
            cpu_input, npu_input = create_common_tensor(item, 0, 100)
            # case 1: unsqueeze+permute ==> can be optimized as single permute
            # (contiguous_h_combined should not be called)
            with torch.autograd.profiler.profile(use_device='npu') as prof:
                npu_out1 = npu_input.unsqueeze(1).transpose(2, 3).contiguous()
            self.assertEqual(check_operators_in_prof(['contiguous_d_Transpose'], prof, ['contiguous_h_combined']) or
                             check_operators_in_prof(['aclnnInplaceCopy'], prof, ['contiguous_h_combined']),
                             True, message="Error operators called!")
            cpu_out1 = cpu_input.unsqueeze(1).transpose(2, 3).contiguous()
            self.assertRtolEqual(npu_out1.to("cpu").numpy(), cpu_out1.numpy())

            # case 2: permute+unsqueeze ==> can be optimized as single permute
            # (contiguous_h_combined should not be called)
            with torch.autograd.profiler.profile(use_device='npu') as prof:
                npu_out2 = npu_input.permute(1, 0, 2, 3).unsqueeze(0).contiguous()
            self.assertEqual(check_operators_in_prof(['contiguous_d_Transpose'], prof, ['contiguous_h_combined']) or
                             check_operators_in_prof(['aclnnInplaceCopy'], prof, ['contiguous_h_combined']),
                             True, message="Error operators called!")
            cpu_out2 = cpu_input.permute(1, 0, 2, 3).unsqueeze(0).contiguous()
            self.assertRtolEqual(npu_out2.to("cpu").numpy(), cpu_out2.numpy())

    def test_unsqueeze_narrow_copy_contiguous(self, device="npu"):
        dtype_list2 = [np.float16, np.float32]
        format_list2 = [-1]
        shape_list2 = [
            # 3D-->4D-->3D
                      [20, 30, 40],
            # 4D-->5D-->4D test memory allocation
                      [2, 300, 400, 500],
            # 5D-->6D-->5D
                      [20, 30, 40, 50, 60]
        ]
        shape_format2 = [
            [i, j, k] for i in dtype_list2 for j in format_list2 for k in shape_list2
        ]

        for item in shape_format2:
            cpu_input, npu_input = create_common_tensor(item, 0, 100)
            # case 1: unsqueeze+narrow
            with torch.autograd.profiler.profile(use_device='npu') as prof:
                npu_out1 = npu_input.unsqueeze(0)[:, :, 1:10].contiguous()
            self.assertEqual(check_operators_in_prof(['contiguous_h_match', 'contiguous_d_Slice'], prof) or
                             check_operators_in_prof(['aclnnInplaceCopy'], prof),
                             True, message="Error operators called!")
            cpu_out1 = cpu_input.unsqueeze(0)[:, :, 1:10].contiguous()
            self.assertRtolEqual(npu_out1.to("cpu").numpy(), cpu_out1.numpy())
            # case 2: narrow+unsqueeze
            with torch.autograd.profiler.profile(use_device='npu') as prof:
                npu_out2 = npu_input[:, 1:10].unsqueeze(2).contiguous()
            self.assertEqual(check_operators_in_prof(['contiguous_h_match', 'contiguous_d_Slice'], prof) or
                             check_operators_in_prof(['aclnnInplaceCopy'], prof),
                             True, message="Error operators called!")
            cpu_out2 = cpu_input[:, 1:10].unsqueeze(2).contiguous()
            self.assertRtolEqual(npu_out2.to("cpu").numpy(), cpu_out2.numpy())

    def test_unsqueeze_select_copy_contiguous(self, device="npu"):
        dtype_list3 = [np.float16, np.float32]
        format_list3 = [-1]
        shape_list3 = [
                      [2, 3, 4],
                      [2, 300, 400, 500],
                      [2, 3, 4, 5, 6]
        ]
        shape_format3 = [
            [i, j, k] for i in dtype_list3 for j in format_list3 for k in shape_list3
        ]

        for item in shape_format3:
            cpu_input, npu_input = create_common_tensor(item, 0, 100)
            # case 1: unsqueeze+select
            with torch.autograd.profiler.profile(use_device='npu') as prof:
                npu_out1 = npu_input.unsqueeze(0).select(2, 1).contiguous()
            cpu_out1 = cpu_input.unsqueeze(0).select(2, 1).contiguous()
            self.assertEqual(check_operators_in_prof(['contiguous_h_match', 'contiguous_d_Slice'], prof) or
                             check_operators_in_prof(['aclnnInplaceCopy'], prof),
                             True, message="Error operators called!")
            self.assertRtolEqual(npu_out1.to("cpu").numpy(), cpu_out1.numpy())
            # case 2: select+unsqueeze
            with torch.autograd.profiler.profile(use_device='npu') as prof:
                npu_out2 = npu_input.select(1, 1).unsqueeze(0).contiguous()
            self.assertEqual(check_operators_in_prof(['contiguous_h_match', 'contiguous_d_Slice'], prof) or
                             check_operators_in_prof(['aclnnInplaceCopy'], prof),
                             True, message="Error operators called!")
            cpu_out2 = cpu_input.select(1, 1).unsqueeze(0).contiguous()
            self.assertRtolEqual(npu_out2.to("cpu").numpy(), cpu_out2.numpy())

    def test_unsqueeze_unfold_copy_contiguous(self, device="npu"):
        dtype_list4 = [np.float16, np.float32]
        format_list4 = [-1]
        shape_list4 = [
                      [4, 2, 4],
        ]
        shape_format = [
            [i, j, k] for i in dtype_list4 for j in format_list4 for k in shape_list4
        ]

        for item in shape_format:
            cpu_input, npu_input = create_common_tensor(item, 0, 100)
            # case 1: unsqueeze+unfold:size==step ==> can be optimized as reshape+permute
            with torch.autograd.profiler.profile(use_device='npu') as prof:
                npu_out1 = npu_input.unsqueeze(1).unfold(0, 2, 2).contiguous()
            self.assertEqual(check_operators_in_prof(['contiguous_d_Transpose'], prof) or
                             check_operators_in_prof(['aclnnInplaceCopy'], prof),
                             True, message="Error operators called!")
            self.assertEqual(check_operators_in_prof(['contiguous_d_Transpose'], prof) or
                             check_operators_in_prof(['aclnnInplaceCopy'], prof),
                             True, message="Error operators called!")
            cpu_out1 = cpu_input.unsqueeze(1).unfold(0, 2, 2).contiguous()
            self.assertRtolEqual(npu_out1.to("cpu").numpy(), cpu_out1.numpy())
            # case 2: unfold+unsqueeze: size!=step ==> cannot be optimized
            with torch.autograd.profiler.profile(use_device='npu') as prof:
                npu_out2 = npu_input.unfold(2, 2, 3).unsqueeze(1).contiguous()
            self.assertEqual(check_operators_in_prof(['contiguous_d_AsStrided'], prof, ['contiguous_h_combined']) or
                             check_operators_in_prof(['aclnnInplaceCopy'], prof, ['contiguous_h_combined']),
                             True, message="Error operators called!")
            cpu_out2 = cpu_input.unfold(2, 2, 3).unsqueeze(1).contiguous()
            self.assertRtolEqual(npu_out2.to("cpu").numpy(), cpu_out2.numpy())

    def test_unsqueeze_strideslice_copy_contiguous(self, device="npu"):
        dtype_list5 = [np.float16, np.float32]
        format_list5 = [-1]
        shape_list5 = [
                      [20, 200, 40],
                      [20, 200, 40, 10]
        ]
        shape_format5 = [
            [i, j, k] for i in dtype_list5 for j in format_list5 for k in shape_list5
        ]

        for item in shape_format5:
            cpu_input, npu_input = create_common_tensor(item, 0, 100)
            # case 1: squeeze + strideslice ==> cannot be optimized(contiguous_h_combined should not be called)
            with torch.autograd.profiler.profile(use_device='npu') as prof:
                npu_out1 = npu_input.unsqueeze(1)[:, :, 20:150:3].contiguous()
            self.assertEqual(check_operators_in_prof(['contiguous_d_AsStrided'], prof, ['contiguous_h_combined']) or
                             check_operators_in_prof(['aclnnInplaceCopy'], prof, ['contiguous_h_combined']),
                             True, message="Error operators called!")
            cpu_out1 = cpu_input.unsqueeze(1)[:, :, 20:150:3].contiguous()
            self.assertRtolEqual(npu_out1.to("cpu").numpy(), cpu_out1.numpy())
            # case 2: strideslice + squeeze ==> cannot be optimized(contiguous_h_combined should not be called)
            with torch.autograd.profiler.profile(use_device='npu') as prof:
                npu_out2 = npu_input[:, :, 10:19:3].unsqueeze(0).contiguous()
            self.assertEqual(check_operators_in_prof(['contiguous_d_AsStrided'], prof, ['contiguous_h_combined']) or
                             check_operators_in_prof(['aclnnInplaceCopy'], prof, ['contiguous_h_combined']),
                             True, message="Error operators called!")
            cpu_out2 = cpu_input[:, :, 10:19:3].unsqueeze(0).contiguous()
            self.assertRtolEqual(npu_out2.to("cpu").numpy(), cpu_out2.numpy())


if __name__ == "__main__":
    run_tests()
