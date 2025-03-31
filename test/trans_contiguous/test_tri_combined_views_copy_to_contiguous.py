import os
import torch
import numpy as np
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor, check_operators_in_prof

os.environ["COMBINED_ENABLE"] = "1"  # Open combined-view cases optimization


class TestTriCombinedViewsCopyToContiguous(TestCase):
    def test_view_narrow_permute_copy_contiguous(self, device="npu"):
        dtype_list1 = [np.float16, np.float32]
        format_list1 = [-1]
        shape_list1 = [
                      [200, 30, 40, 16],
        ]
        shape_format = [
            [i, j, k] for i in dtype_list1 for j in format_list1 for k in shape_list1
        ]

        for item in shape_format:
            cpu_input, npu_input = create_common_tensor(item, 0, 100)
            # case 1: view+narrow+permute ==> cannot be optimized
            with torch.autograd.profiler.profile(use_device='npu') as prof:
                npu_out1 = npu_input.view(npu_input.size(0) * npu_input.size(1), npu_input.size(2),
                                          npu_input.size(3))[:, 1:10].transpose(0, 1).contiguous()
            self.assertEqual(check_operators_in_prof(['contiguous_d_AsStrided'], prof, ['contiguous_h_combined'])
                             or check_operators_in_prof(['aclnnInplaceCopy'], prof, ['contiguous_h_combined']),
                             True, message="Error operators called!")
            cpu_out1 = cpu_input.view(cpu_input.size(0) * cpu_input.size(1), cpu_input.size(2),
                                      cpu_input.size(3))[:, 1:10].transpose(0, 1).contiguous()
            self.assertRtolEqual(npu_out1.to("cpu").numpy(), cpu_out1.numpy())

            # case 2: permute+view+narrow ==> cannot be optimized
            with torch.autograd.profiler.profile(use_device='npu') as prof:
                npu_out2 = npu_input.permute(1, 0, 2, 3). \
                    view(npu_input.size(1), npu_input.size(0), npu_input.size(
                        2) * npu_input.size(3))[:, :, 1:10].contiguous()
            self.assertEqual(check_operators_in_prof(['contiguous_d_AsStrided'], prof, ['contiguous_h_combined'])
                             or check_operators_in_prof(['aclnnInplaceCopy'], prof, ['contiguous_h_combined']),
                             True, message="Error operators called!")
            cpu_out2 = cpu_input.permute(1, 0, 2, 3). \
                view(
                cpu_input.size(1),
                cpu_input.size(0),
                cpu_input.size(2) *
                cpu_input.size(3))[
                :,
                :,
                1:10].contiguous()
            self.assertRtolEqual(npu_out2.to("cpu").numpy(), cpu_out2.numpy())

    def test_view_select_permute_copy_contiguous(self, device="npu"):
        dtype_list2 = [np.float16, np.float32]
        format_list2 = [-1]
        shape_list2 = [
                      [200, 30, 40, 16],
        ]
        shape_format = [
            [i, j, k] for i in dtype_list2 for j in format_list2 for k in shape_list2
        ]

        for item in shape_format:
            cpu_input, npu_input = create_common_tensor(item, 0, 100)
            # case 1: view+select+permute ==> cannot be optimized
            with torch.autograd.profiler.profile(use_device='npu') as prof:
                npu_out1 = npu_input.view(npu_input.size(0) * npu_input.size(1), npu_input.size(2),
                                          npu_input.size(3))[:, 1].transpose(0, 1).contiguous()
            self.assertEqual(check_operators_in_prof(['contiguous_d_AsStrided'], prof, ['contiguous_h_combined'])
                             or check_operators_in_prof(['aclnnInplaceCopy'], prof, ['contiguous_h_combined']),
                             True, message="Error operators called!")
            cpu_out1 = cpu_input.view(cpu_input.size(0) * cpu_input.size(1), cpu_input.size(2),
                                      cpu_input.size(3))[:, 1].transpose(0, 1).contiguous()
            self.assertRtolEqual(npu_out1.to("cpu").numpy(), cpu_out1.numpy())

            # case 2: permute+view+select ==> cannot be optimized
            with torch.autograd.profiler.profile(use_device='npu') as prof:
                npu_out2 = npu_input.permute(1, 0, 2, 3). \
                    view(npu_input.size(1), npu_input.size(0), npu_input.size(
                        2) * npu_input.size(3))[:, :, 2].contiguous()
            self.assertEqual(check_operators_in_prof(['contiguous_d_AsStrided'], prof, ['contiguous_h_combined'])
                             or check_operators_in_prof(['aclnnInplaceCopy'], prof, ['contiguous_h_combined']),
                             True, message="Error operators called!")
            cpu_out2 = cpu_input.permute(1, 0, 2, 3). \
                view(cpu_input.size(1), cpu_input.size(0), cpu_input.size(2) * cpu_input.size(3))[:, :, 2].contiguous()
            self.assertRtolEqual(npu_out2.to("cpu").numpy(), cpu_out2.numpy())


if __name__ == "__main__":
    run_tests()
