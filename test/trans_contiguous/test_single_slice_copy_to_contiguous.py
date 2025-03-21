import os
import torch
import numpy as np
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor, check_operators_in_prof

os.environ["COMBINED_ENABLE"] = "1"  # Open combined-view cases optimization

# Optimized view Ops contains Transpose, permute, narrow, strideslice, select, unfold


class SingleViewCopyToContiguous(TestCase):
    def test_narrow_copy_contiguous(self, device="npu"):
        # AssertionError: required dtype in [np.bool, np.int32, np.float16, np.float32, np.int8, np.uint8, np.int64]
        # However, considering the dtypes that Transdata supports, only np.float16, np.float32 are tested.
        dtype_list1 = [np.float16, np.float32]
        format_list_4D = [0, 3, 29, 4]
        shape_list_4D = [[2, 32, 16, 20]]
        format_list_5D = [30, 32, 33]
        shape_list_5D = [[2, 32, 16, 20, 15]]
        shape_format_4D = [
            [i, j, k] for i in dtype_list1 for j in format_list_4D for k in shape_list_4D
        ]
        shape_format_5D = [
            [i, j, k] for i in dtype_list1 for j in format_list_5D for k in shape_list_5D
        ]
        shape_format1 = shape_format_4D + shape_format_5D

        for item in shape_format1:
            cpu_input, npu_input = create_common_tensor(item, 0, 100)
            # for narrow with step=1, if narrow at the first axis, it will generate a contiguous tensor
            with torch.autograd.profiler.profile(use_device='npu') as prof:
                npu_out1 = npu_input[:, :16, :, :].contiguous()
            self.assertEqual(check_operators_in_prof(['contiguous_d_Slice'], prof)
                             or check_operators_in_prof(['aclnnInplaceCopy'], prof),
                             True, "contiguous_d_Slice or aclnnInplaceCopy is not called!")
            with torch.autograd.profiler.profile(use_device='npu') as prof:
                npu_out2 = npu_input[:, :, 1:16, :].contiguous()
            self.assertEqual(check_operators_in_prof(['contiguous_d_Slice'], prof)
                             or check_operators_in_prof(['aclnnInplaceCopy'], prof),
                             True, "contiguous_d_Slice or aclnnInplaceCopy is not called!")
            with torch.autograd.profiler.profile(use_device='npu') as prof:
                npu_out3 = npu_input[:, :, :, 2:16].contiguous()
            self.assertEqual(check_operators_in_prof(['contiguous_d_Slice'], prof)
                             or check_operators_in_prof(['aclnnInplaceCopy'], prof),
                             True, "contiguous_d_Slice or aclnnInplaceCopy is not called!")

            cpu_out1 = cpu_input[:, :16, :, :].contiguous()
            cpu_out2 = cpu_input[:, :, 1:16, :].contiguous()
            cpu_out3 = cpu_input[:, :, :, 2:16].contiguous()
            if npu_input.dim() == 5:
                cpu_out4 = cpu_input[:, :, :, :, 3:10].contiguous()
                npu_out4 = npu_input[:, :, :, :, 3:10].contiguous()
                self.assertRtolEqual(npu_out4.to("cpu").numpy(), cpu_out4.numpy())

            self.assertRtolEqual(npu_out1.to("cpu").numpy(), cpu_out1.numpy())
            self.assertRtolEqual(npu_out2.to("cpu").numpy(), cpu_out2.numpy())
            self.assertRtolEqual(npu_out3.to("cpu").numpy(), cpu_out3.numpy())

    def test_strideslice_copy_contiguous(self, device="npu"):
        dtype_list2 = [np.float16, np.float32, np.int8, np.int32, np.uint8, np.bool_]
        format_list2 = [-1]
        shape_list2 = [[10, 32, 16, 9], [10, 32, 16, 9, 10]]
        shape_format2 = [
            [i, j, k] for i in dtype_list2 for j in format_list2 for k in shape_list2
        ]

        for item in shape_format2:
            cpu_input, npu_input = create_common_tensor(item, 0, 100)
            # for indexing with step>1 -- stridedSlice
            if cpu_input.dim() == 4:
                with torch.autograd.profiler.profile(use_device='npu') as prof:
                    npu_out1 = npu_input[::2].contiguous()
                self.assertEqual(check_operators_in_prof(['contiguous_d_StridedSlice'], prof)
                                 or check_operators_in_prof(['aclnnInplaceCopy'], prof),
                                 True, "Error operators called!")
                with torch.autograd.profiler.profile(use_device='npu') as prof:
                    npu_out2 = npu_input[:, 1:17:4].contiguous()
                self.assertEqual(check_operators_in_prof(['contiguous_d_StridedSlice'], prof)
                                 or check_operators_in_prof(['aclnnInplaceCopy'], prof),
                                 True, "Error operators called!")
                with torch.autograd.profiler.profile(use_device='npu') as prof:
                    npu_out3 = npu_input[:, :, 2:16:5].contiguous()
                self.assertEqual(check_operators_in_prof(['contiguous_d_StridedSlice'], prof)
                                 or check_operators_in_prof(['aclnnInplaceCopy'], prof),
                                 True, "Error operators called!")
                with torch.autograd.profiler.profile(use_device='npu') as prof:
                    # stridedSlice do not support slice at last dim
                    npu_out4 = npu_input[:, :, :, 3:9:2].contiguous()
                self.assertEqual(check_operators_in_prof(['contiguous_d_AsStrided'], prof)
                                 or check_operators_in_prof(['aclnnInplaceCopy'], prof),
                                 True, "Error operators called!")
                with torch.autograd.profiler.profile(use_device='npu') as prof:
                    npu_out5 = npu_input[::2, 1:17:4, 2:16:5, :].contiguous()
                self.assertEqual(check_operators_in_prof(['contiguous_d_StridedSlice'], prof)
                                 or check_operators_in_prof(['aclnnInplaceCopy'], prof),
                                 True, "Error operators called!")

                cpu_out1 = cpu_input[::2].contiguous()
                cpu_out2 = cpu_input[:, 1:17:4].contiguous()
                cpu_out3 = cpu_input[:, :, 2:16:5].contiguous()
                cpu_out4 = cpu_input[:, :, :, 3:9:2].contiguous()
                # strideslice at each axis
                cpu_out5 = cpu_input[::2, 1:17:4, 2:16:5, :].contiguous()

                self.assertRtolEqual(npu_out1.to("cpu").numpy(), cpu_out1.numpy())
                self.assertRtolEqual(npu_out2.to("cpu").numpy(), cpu_out2.numpy())
                self.assertRtolEqual(npu_out3.to("cpu").numpy(), cpu_out3.numpy())
                self.assertRtolEqual(npu_out4.to("cpu").numpy(), cpu_out4.numpy())
                self.assertRtolEqual(npu_out5.to("cpu").numpy(), cpu_out5.numpy())
            if cpu_input.dim() == 5:
                cpu_out6 = cpu_input[:, :, :, :, 1:7:3].contiguous()
                npu_out6 = npu_input[:, :, :, :, 1:7:3].contiguous()
                self.assertRtolEqual(npu_out6.to("cpu").numpy(), cpu_out6.numpy())

    def test_select_copy_contiguous(self, device="npu"):
        dtype_list = [np.float16, np.float32]
        format_list = [-1]
        shape_list = [[2, 32, 16, 9], [2, 32, 16, 9, 10]]
        shape_format = [
            [i, j, k] for i in dtype_list for j in format_list for k in shape_list
        ]

        for item in shape_format:
            cpu_input, npu_input = create_common_tensor(item, 0, 100)
            for dim in range(1, len(item[2])):
                with torch.autograd.profiler.profile(use_device='npu') as prof:
                    npu_out = npu_input.select(dim, 1).contiguous()
                self.assertEqual(check_operators_in_prof(['contiguous_d_StridedSlice'], prof)
                                 or check_operators_in_prof(['aclnnInplaceCopy'], prof),
                                 True, "contiguous_d_StridedSlice or aclnnInplaceCopy is not called!")
                cpu_out = cpu_input.select(dim, 1).contiguous()
                self.assertRtolEqual(npu_out.to("cpu").numpy(), cpu_out.numpy())

    def test_span_axis_strideslice_contiguous(self, device="npu"):
        dtype_list = [np.float16, np.float32]
        format_list = [-1]
        shape_list = [[32, 8, 2], [(8, 6, 2), (5, 4, 1), 1]]
        shape_format = [
            [i, j, shape_list[0]] for i in dtype_list for j in format_list
        ]

        for item in shape_format:
            cpu_input, npu_input = create_common_tensor(item, 0, 100)
            # npuStrideSlice do not support span-axis strideslice, can not be optimized
            with torch.autograd.profiler.profile(use_device='npu') as prof:
                npu_out = torch.as_strided(npu_input,
                                           shape_list[1][0], shape_list[1][1], shape_list[1][2]).contiguous()
            self.assertEqual(check_operators_in_prof(['contiguous_d_AsStrided'], prof, ['contiguous_d_StridedSlice'])
                             or check_operators_in_prof(['aclnnInplaceCopy'], prof, ['contiguous_d_StridedSlice']),
                             True, "Error operators called!")
            cpu_out = torch.as_strided(cpu_input,
                                       shape_list[1][0], shape_list[1][1], shape_list[1][2]).contiguous()
            self.assertRtolEqual(npu_out.to("cpu").numpy(), cpu_out.numpy())


if __name__ == "__main__":
    run_tests()
