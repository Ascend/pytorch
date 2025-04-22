import os
import torch
import numpy as np
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor, check_operators_in_prof

os.environ["COMBINED_ENABLE"] = "1"  # Open combined-view cases optimization

# Optimized view Ops contains Transpose, permute, narrow, strideslice, select, unfold


class SingleViewCopyToContiguous(TestCase):
    def test_view_copy(self, device="npu"):
        dtype_list1 = [np.float16, np.float32]
        format_list1 = [0, 3, 29]
        shape_list1 = [
            # No padding for NZ format
                      [2, 3, 16, 16],
            # Padding for NZ format
                      [2, 3, 15, 16],
        ]
        shape_format1 = [
            [i, j, k] for i in dtype_list1 for j in format_list1 for k in shape_list1
        ]

        for item in shape_format1:
            cpu_input, npu_input = create_common_tensor(item, 0, 100)
            # Directly using d2dcopy without transdata(contiguous_d_Reshape)
            # case1. base format
            match_case1 = (item[1] == 0)
            with torch.autograd.profiler.profile(use_device='npu') as prof:
                npu_out1 = npu_input.view(1, 6, npu_input.size(2), npu_input.size(3)).clone()
            # case2. The key axis remains unchanged for NZ format
            match_case2 = (item[1] == 29)
            if match_case1 or match_case2:
                self.assertEqual(check_operators_in_prof(['contiguous_d_Reshape'], prof) or
                                 check_operators_in_prof(['aclnnInplaceCopy'], prof),
                                 True, message="contiguous_d_Reshape or aclnnInplaceCopy is not called!")
            else:
                self.assertEqual(check_operators_in_prof(['d2dCopyAsync'], prof) or
                                 check_operators_in_prof(['aclnnInplaceCopy'], prof),
                                 True, message="d2dCopyAsync or aclnnInplaceCopy is not called!")
            cpu_out1 = cpu_input.view(1, 6, cpu_input.size(2), cpu_input.size(3)).clone()
            self.assertRtolEqual(npu_out1.to("cpu").numpy(), cpu_out1.numpy())

            # The key axis changes for NZ format
            with torch.autograd.profiler.profile(use_device='npu') as prof:
                npu_out2 = npu_input.view(1, 6, npu_input.size(2) * npu_input.size(3), 1).clone()
            if match_case1:
                self.assertEqual(check_operators_in_prof(['contiguous_d_Reshape'], prof) or
                                 check_operators_in_prof(['aclnnInplaceCopy'], prof),
                                 True, message="contiguous_d_Reshape or aclnnInplaceCopy is not called!")
            else:
                self.assertEqual(check_operators_in_prof(['d2dCopyAsync'], prof) or
                                 check_operators_in_prof(['aclnnInplaceCopy'], prof),
                                 True, message="d2dCopyAsync or aclnnInplaceCopy is not called!")
            cpu_out2 = cpu_input.view(1, 6, cpu_input.size(2) * cpu_input.size(3), 1).clone()
            self.assertRtolEqual(npu_out2.to("cpu").numpy(), cpu_out2.numpy())

    def test_unsqueeze_copy(self, device="npu"):
        dtype_list2 = [np.float16, np.float32]
        format_list2 = [2, 3, 29]
        shape_list2 = [
                      [3, 16, 16],
                      [3, 15, 16],
        ]
        shape_format2 = [
            [i, j, k] for i in dtype_list2 for j in format_list2 for k in shape_list2
        ]

        for i in range(3):
            for item in shape_format2:
                cpu_input, npu_input = create_common_tensor(item, 0, 100)
                # Directly using d2dcopy without transdata(contiguous_d_Reshape)
                # case1. base format
                match_case1 = (item[1] == 2)
                # case2. The key axis remains unchanged for NZ format
                match_case2 = (item[1] == 29 and i < 2)
                with torch.autograd.profiler.profile(use_device='npu') as prof:
                    npu_out = npu_input.unsqueeze(i).clone()
                if match_case1 or match_case2:
                    self.assertEqual(check_operators_in_prof(['contiguous_d_Reshape'], prof) or
                                 check_operators_in_prof(['aclnnInplaceCopy'], prof),
                                     True, message="contiguous_d_Reshape or aclnnInplaceCopy is not called!")
                else:
                    self.assertEqual(check_operators_in_prof(['d2dCopyAsync'], prof) or
                                 check_operators_in_prof(['aclnnInplaceCopy'], prof),
                                     True, message="d2dCopyAsync or aclnnInplaceCopy is not called!")
                cpu_out = cpu_input.unsqueeze(i).clone()
                self.assertRtolEqual(npu_out.to("cpu").numpy(), cpu_out.numpy())

    def test_flatten_copy(self, device="npu"):
        dtype_list3 = [np.float16, np.float32]
        format_list3 = [0, 3, 29]
        shape_list3 = [
                      [2, 3, 16, 16],
                      [2, 3, 16, 15],
        ]
        shape_format3 = [
            [i, j, k] for i in dtype_list3 for j in format_list3 for k in shape_list3
        ]

        for item in shape_format3:
            cpu_input, npu_input = create_common_tensor(item, 0, 100)
            with torch.autograd.profiler.profile(use_device='npu') as prof:
                npu_out = torch.flatten(npu_input, 0, 1).clone()
            if item[1] == 3:
                # Using d2dcopy with transdata(d2dCopyAsync)
                self.assertEqual(check_operators_in_prof(['d2dCopyAsync'], prof) or
                                 check_operators_in_prof(['aclnnInplaceCopy'], prof),
                                 True, message="d2dCopyAsync or aclnnInplaceCopy is not called!")
            else:
                # Directly using d2dcopy without transdata(contiguous_d_Reshape)
                self.assertEqual(check_operators_in_prof(['contiguous_d_Reshape'], prof) or
                                 check_operators_in_prof(['aclnnInplaceCopy'], prof),
                                 True, message="contiguous_d_Reshape or aclnnInplaceCopy is not called!")

            cpu_out = torch.flatten(cpu_input, 0, 1).clone()
            self.assertRtolEqual(npu_out.to("cpu").numpy(), cpu_out.numpy())

    def test_narrow_at_first_axis_copy(self, device="npu"):
        # this case: slice at the first dim, tensor with offset remains contiguous
        dtype_list4 = [np.float16, np.float32]
        format_list4 = [2, 3, 29]
        shape_list4 = [
                      [20, 16, 16],
                      [20, 16, 15],
        ]
        shape_format4 = [
            [i, j, k] for i in dtype_list4 for j in format_list4 for k in shape_list4
        ]

        for item in shape_format4:
            cpu_input, npu_input = create_common_tensor(item, 0, 100)
            # Directly using d2dcopy without transdata(contiguous_d_Reshape)
            # The key axis remains unchanged for NZ format in all cases.
            # case1. base format
            match_case1 = (item[1] == 2)
            # case2. NZ format but no padding
            match_case2 = (item[1] == 29 and item[2] == [20, 16, 16])

            # contiguous and no offset
            with torch.autograd.profiler.profile(use_device='npu') as prof:
                npu_out1 = npu_input[:10, :, :].clone()
            # case3. NZ format with padding but no offset
            match_case3 = (item[1] == 29 and True)
            if match_case1 or match_case2 or match_case3:
                self.assertEqual(check_operators_in_prof(['contiguous_d_Reshape'], prof) or
                                 check_operators_in_prof(['aclnnInplaceCopy'], prof),
                                 True, message="contiguous_d_Reshape or aclnnInplaceCopy is not called!")
            else:
                self.assertEqual(check_operators_in_prof(['contiguous_d_Slice'], prof) or
                                 check_operators_in_prof(['aclnnInplaceCopy'], prof),
                                 True, message="contiguous_d_Slice or aclnnInplaceCopy is not called!")
            cpu_out1 = cpu_input[:10, :, :].clone()
            self.assertRtolEqual(npu_out1.to("cpu").numpy(), cpu_out1.numpy())

            # contiguous but has offset
            with torch.autograd.profiler.profile(use_device='npu') as prof:
                npu_out2 = npu_input[1:10, :, :].clone()
            match_case3 = False
            if match_case1 or match_case2 or match_case3:
                self.assertEqual(check_operators_in_prof(['contiguous_d_Reshape'], prof) or
                                 check_operators_in_prof(['aclnnInplaceCopy'], prof),
                                 True, message="contiguous_d_Reshape or aclnnInplaceCopy is not called!")
            else:
                self.assertEqual(check_operators_in_prof(['contiguous_d_Slice'], prof) or
                                 check_operators_in_prof(['aclnnInplaceCopy'], prof),
                                 True, message="contiguous_d_Slice or aclnnInplaceCopy is not called!")
            cpu_out2 = cpu_input[1:10, :, :].clone()
            self.assertRtolEqual(npu_out2.to("cpu").numpy(), cpu_out2.numpy())

    def test_select_at_first_axis_to_single_element_tensor_copy(self, device="npu"):
        dtype_list5 = [torch.float32]
        format_list5 = [2, 3, 29]
        shape_format5 = [
            [i, j] for i in dtype_list5 for j in format_list5
        ]

        for item in shape_format5:
            cpu_input = torch.tensor([1.0]).to(item[0])
            npu_input = torch_npu.npu_format_cast(cpu_input.npu(), item[1])

            match_case = (item[1] == 2 or item[1] == 29)
            with torch.autograd.profiler.profile(use_device='npu') as prof:
                npu_out1 = npu_input[0].clone()
            if match_case:
                self.assertEqual(check_operators_in_prof(['contiguous_d_Reshape'], prof) or
                                 check_operators_in_prof(['aclnnInplaceCopy'], prof),
                                 True, message="contiguous_d_Reshape or aclnnInplaceCopy is not called!")
            else:
                self.assertEqual(check_operators_in_prof(['contiguous_d_Slice'], prof) or
                                 check_operators_in_prof(['aclnnInplaceCopy'], prof),
                                 True, message="contiguous_d_Slice or aclnnInplaceCopy is not called!")
            cpu_out1 = cpu_input[0].clone()
            self.assertRtolEqual(npu_out1.to("cpu").numpy(), cpu_out1.numpy())

            with torch.autograd.profiler.profile(use_device='npu') as prof:
                npu_out2 = npu_input[0] + 1
            if match_case:
                self.assertEqual(check_operators_in_prof(['contiguous_h_memRepoint'], prof) or
                                 check_operators_in_prof(['aclnnInplaceCopy'], prof),
                                 True, message="contiguous_h_memRepoint or aclnnInplaceCopy is not called!")
            else:
                # refresh storage desc after transdata
                self.assertEqual(check_operators_in_prof(['Identity'], prof) or
                                 check_operators_in_prof(['aclnnInplaceCopy'], prof),
                                 True, message="Identity or aclnnInplaceCopy is not called!")
            cpu_out2 = cpu_input[0] + 1
            self.assertRtolEqual(npu_out2.to("cpu").numpy(), cpu_out2.numpy())


if __name__ == "__main__":
    run_tests()
