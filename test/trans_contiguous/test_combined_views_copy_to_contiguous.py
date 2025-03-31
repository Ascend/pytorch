import os
import numpy as np
import torch
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor, check_operators_in_prof

os.environ["COMBINED_ENABLE"] = "1"  # Open combined-view cases optimization


# Note: NPU only support trans-contiguous with base format, so format_list uses -1


class CombinedViewsCopyToContiguous(TestCase):
    def test_permute_narrow_copy_contiguous(self):
        dtype_list1 = [np.float16]
        format_list1 = [-1]
        shape_list1 = [[20, 30, 40, 50]]
        shape_format1 = [
            [i, j, k] for i in dtype_list1 for j in format_list1 for k in shape_list1
        ]

        for item in shape_format1:
            cpu_input, npu_input = create_common_tensor(item, 0, 100)
            # case 1: permute+narrow
            with torch.autograd.profiler.profile(use_device='npu') as prof:
                npu_out1 = npu_input.permute(1, 3, 2, 0)[:10].contiguous()
            self.assertEqual(check_operators_in_prof(['contiguous_d_Slice', 'contiguous_d_Transpose'], prof)
                             or check_operators_in_prof(['aclnnInplaceCopy'], prof),
                             True, message="Error operators called!")
            cpu_out1 = cpu_input.permute(1, 3, 2, 0)[:10].contiguous()
            self.assertRtolEqual(npu_out1.to("cpu").numpy(), cpu_out1.numpy())

            # case 2: narrow+permute
            with torch.autograd.profiler.profile(use_device='npu') as prof:
                npu_out2 = npu_input[:, 1:10].permute(1, 0, 3, 2).contiguous()
            self.assertEqual(check_operators_in_prof(['contiguous_d_Slice', 'contiguous_d_Transpose'], prof)
                             or check_operators_in_prof(['aclnnInplaceCopy'], prof),
                             True, message="Error operators called!")
            cpu_out2 = cpu_input[:, 1:10].permute(1, 0, 3, 2).contiguous()
            self.assertRtolEqual(npu_out2.to("cpu").numpy(), cpu_out2.numpy())

    def test_permute_select_copy_contiguous(self):
        dtype_list2 = [np.float32]
        format_list2 = [-1]
        shape_list2 = [[20, 30, 40, 50], ]
        shape_format2 = [
            [i, j, k] for i in dtype_list2 for j in format_list2 for k in shape_list2
        ]

        for item in shape_format2:
            cpu_input, npu_input = create_common_tensor(item, 0, 100)
            # case 1: permute+select
            with torch.autograd.profiler.profile(use_device='npu') as prof:
                npu_out1 = npu_input.permute(1, 3, 2, 0).select(1, 2).contiguous()
            self.assertEqual(check_operators_in_prof(['contiguous_d_StridedSlice', 'contiguous_d_Transpose'], prof)
                             or check_operators_in_prof(['aclnnInplaceCopy'], prof),
                             True, message="Error operators called!")
            cpu_out1 = cpu_input.permute(1, 3, 2, 0).select(1, 2).contiguous()
            self.assertRtolEqual(npu_out1.to("cpu").numpy(), cpu_out1.numpy())

            # case 2: select+permute
            with torch.autograd.profiler.profile(use_device='npu') as prof:
                npu_out2 = npu_input.select(1, 0).permute(1, 0, 2).contiguous()
            self.assertEqual(check_operators_in_prof(['contiguous_d_StridedSlice', 'contiguous_d_Transpose'], prof)
                             or check_operators_in_prof(['aclnnInplaceCopy'], prof),
                             True, message="Error operators called!")
            cpu_out2 = cpu_input.select(1, 0).permute(1, 0, 2).contiguous()
            self.assertRtolEqual(npu_out2.to("cpu").numpy(), cpu_out2.numpy())

    def test_permute_strideslice_copy_contiguous(self):
        dtype_list3 = [np.float16]
        format_list3 = [-1]
        shape_list3 = [[20, 30, 40, 50]]
        shape_format3 = [
            [i, j, k] for i in dtype_list3 for j in format_list3 for k in shape_list3
        ]

        for item in shape_format3:
            cpu_input, npu_input = create_common_tensor(item, 0, 100)
            # case 1: permute+strideslice-no offset ==> all cannot be optimized
            # (contiguous_h_combined should not be called)
            with torch.autograd.profiler.profile(use_device='npu') as prof:
                npu_out1 = npu_input.permute(1, 3, 2, 0)[::2].contiguous()
            self.assertEqual(check_operators_in_prof(['contiguous_d_AsStrided'], prof, ['contiguous_h_combined'])
                             or check_operators_in_prof(['aclnnInplaceCopy'], prof, ['contiguous_h_combined']),
                             True, message="Error operators called!")
            cpu_out1 = cpu_input.permute(1, 3, 2, 0)[::2].contiguous()
            self.assertRtolEqual(npu_out1.to("cpu").numpy(), cpu_out1.numpy())

            # case 2: strideslice+permute-with offset ==> all cannot be optimized
            # (contiguous_h_combined should not be called)
            with torch.autograd.profiler.profile(use_device='npu') as prof:
                npu_out2 = npu_input[:, 1:10:3].permute(1, 3, 0, 2).contiguous()
            self.assertEqual(check_operators_in_prof(['contiguous_d_AsStrided'], prof, ['contiguous_h_combined'])
                             or check_operators_in_prof(['aclnnInplaceCopy'], prof, ['contiguous_h_combined']),
                             True, message="Error operators called!")
            cpu_out2 = cpu_input[:, 1:10:3].permute(1, 3, 0, 2).contiguous()
            self.assertRtolEqual(npu_out2.to("cpu").numpy(), cpu_out2.numpy())

    def test_narrow_select_copy_contiguous(self):
        dtype_list4 = [np.float16, np.float32]
        format_list4 = [0, 3, 29]
        shape_list4 = [[20, 30, 40, 16]]
        shape_format4 = [
            [i, j, k] for i in dtype_list4 for j in format_list4 for k in shape_list4
        ]

        for item in shape_format4:
            cpu_input, npu_input = create_common_tensor(item, 0, 100)
            # case 1: narrow+select
            # narrow at any dim + select the last dim ==> narrow
            with torch.autograd.profiler.profile(use_device='npu') as prof:
                npu_out1 = npu_input[:, 2:4].select(3, 1).contiguous()
            self.assertEqual(check_operators_in_prof(['contiguous_d_Slice'], prof)
                             or check_operators_in_prof(['aclnnInplaceCopy'], prof),
                             True, message="Error operators called!")
            cpu_out1 = cpu_input[:, 2:4].select(3, 1).contiguous()
            # narrow at 0 dim + select the any dim ==> common copy
            with torch.autograd.profiler.profile(use_device='npu') as prof:
                npu_out2 = npu_input[2:4].select(2, 2).contiguous()
            self.assertEqual(check_operators_in_prof(['contiguous_d_AsStrided'], prof)
                             or check_operators_in_prof(['aclnnInplaceCopy'], prof),
                             True, message="Error operators called!")
            cpu_out2 = cpu_input[2:4].select(2, 2).contiguous()
            self.assertRtolEqual(npu_out1.to("cpu").numpy(), cpu_out1.numpy())
            self.assertRtolEqual(npu_out2.to("cpu").numpy(), cpu_out2.numpy())
            # case 2: select+narrow
            # select the 0 dim + narrow at the 1 dim ==> reshape + select
            with torch.autograd.profiler.profile(use_device='npu') as prof:
                npu_out3 = npu_input.select(0, 2)[:, 1:2].contiguous()
            self.assertEqual(check_operators_in_prof(['contiguous_h_match', 'contiguous_d_Slice'], prof)
                             or check_operators_in_prof(['aclnnInplaceCopy'], prof),
                             True, message="Error operators called!")
            cpu_out3 = cpu_input.select(0, 2)[:, 1:2].contiguous()
            # select the 0 dim + narrow at the last dim ==> reshape + select
            with torch.autograd.profiler.profile(use_device='npu') as prof:
                npu_out4 = npu_input.select(0, 1)[:, :, 1:2].contiguous()
            self.assertEqual(check_operators_in_prof(['contiguous_h_match', 'contiguous_d_Slice'], prof)
                             or check_operators_in_prof(['aclnnInplaceCopy'], prof),
                             True, message="Error operators called!")
            cpu_out4 = cpu_input.select(0, 1)[:, :, 1:2].contiguous()

            self.assertRtolEqual(npu_out3.to("cpu").numpy(), cpu_out3.numpy())
            self.assertRtolEqual(npu_out4.to("cpu").numpy(), cpu_out4.numpy())

    def test_narrow_strideslice_copy_contiguous(self):
        dtype_list5 = [np.float32]
        format_list5 = [-1]
        shape_list5 = [[20, 30, 40, 16]]
        shape_format5 = [
            [i, j, k] for i in dtype_list5 for j in format_list5 for k in shape_list5
        ]

        for item in shape_format5:
            cpu_input, npu_input = create_common_tensor(item, 0, 100)
            # case 1: narrow+strideslice
            # slice at adjacent axes + strideslice at lower dim ==> cannot be optimized(contiguous_h_combined is called)
            with torch.autograd.profiler.profile(use_device='npu') as prof:
                npu_out1 = npu_input[2:4, ::2].contiguous()
            self.assertEqual(check_operators_in_prof(['contiguous_d_AsStrided'], prof)
                             or check_operators_in_prof(['aclnnInplaceCopy'], prof),
                             True, message="Error operators called!")
            cpu_out1 = cpu_input[2:4, ::2].contiguous()
            # strideslice at last dim ==> cannot be optimized(contiguous_h_combined should not be called)
            with torch.autograd.profiler.profile(use_device='npu') as prof:
                npu_out2 = npu_input[:, 2:4, :, 1:10:2].contiguous()
            self.assertEqual(check_operators_in_prof(['contiguous_d_AsStrided'], prof, ['contiguous_h_combined'])
                             or check_operators_in_prof(['aclnnInplaceCopy'], prof, ['contiguous_h_combined']),
                             True, message="Error operators called!")
            cpu_out2 = cpu_input[:, 2:4, :, 1:10:2].contiguous()
            # narrow at 0 dim and strideslice at last dim==> can be optimized as slice(contiguous)+select
            with torch.autograd.profiler.profile(use_device='npu') as prof:
                npu_out3 = npu_input[2:4, :, :, ::2].contiguous()
            self.assertEqual(check_operators_in_prof(['contiguous_d_Reshape', 'contiguous_d_StridedSlice'], prof)
                             or check_operators_in_prof(['aclnnInplaceCopy'], prof),
                             True, message="Error operators called!")
            cpu_out3 = cpu_input[2:4, :, :, ::2].contiguous()
            self.assertRtolEqual(npu_out1.to("cpu").numpy(), cpu_out1.numpy())
            self.assertRtolEqual(npu_out2.to("cpu").numpy(), cpu_out2.numpy())
            self.assertRtolEqual(npu_out3.to("cpu").numpy(), cpu_out3.numpy())

            # case 2: strideslice+narrow
            # slice at adjacent axes + strideslice at higher dim ==> reshape+narrow
            with torch.autograd.profiler.profile(use_device='npu') as prof:
                npu_out4 = npu_input[1:10:2, 1:10].contiguous()
            self.assertEqual(check_operators_in_prof(['contiguous_h_match', 'contiguous_d_Slice'], prof)
                             or check_operators_in_prof(['aclnnInplaceCopy'], prof),
                             True, message="Error operators called!")
            cpu_out4 = cpu_input[1:10:2, 1:10].contiguous()
            # slice at non-adjacent axes
            with torch.autograd.profiler.profile(use_device='npu') as prof:
                npu_out5 = npu_input[::2, :, 1:10].contiguous()
            self.assertEqual(check_operators_in_prof(['contiguous_h_match', 'contiguous_d_Slice'], prof)
                             or check_operators_in_prof(['aclnnInplaceCopy'], prof),
                             True, message="Error operators called!")
            cpu_out5 = cpu_input[::2, :, 1:10].contiguous()
            self.assertRtolEqual(npu_out4.to("cpu").numpy(), cpu_out4.numpy())
            self.assertRtolEqual(npu_out5.to("cpu").numpy(), cpu_out5.numpy())

    def test_strideslice_select_contiguous(self):
        dtype_list6 = [np.float16]
        format_list6 = [-1]
        shape_list6 = [[20, 30, 40, 16]]
        shape_format6 = [
            [i, j, k] for i in dtype_list6 for j in format_list6 for k in shape_list6
        ]

        for item in shape_format6:
            cpu_input, npu_input = create_common_tensor(item, 0, 100)
            # case 1: strideslice+select
            # select at last dim ==> cannot be optimized(contiguous_h_combined is called)
            with torch.autograd.profiler.profile(use_device='npu') as prof:
                npu_out1 = npu_input[:10:2].select(3, 1).contiguous()
            self.assertEqual(check_operators_in_prof(['contiguous_d_AsStrided'], prof)
                             or check_operators_in_prof(['aclnnInplaceCopy'], prof),
                             True, message="Error operators called!")
            cpu_out1 = cpu_input[:10:2].select(3, 1).contiguous()
            # select at lower dims except last dim ==> reshape+narrow
            with torch.autograd.profiler.profile(use_device='npu') as prof:
                npu_out2 = npu_input[1:10:2].select(2, 1).contiguous()
            cpu_out2 = cpu_input[1:10:2].select(2, 1).contiguous()
            self.assertEqual(check_operators_in_prof(['contiguous_h_match', 'contiguous_d_Slice'], prof)
                             or check_operators_in_prof(['aclnnInplaceCopy'], prof),
                             True, message="Error operators called!")
            self.assertRtolEqual(npu_out1.to("cpu").numpy(), cpu_out1.numpy())
            self.assertRtolEqual(npu_out2.to("cpu").numpy(), cpu_out2.numpy())
            # case 2: select+strideslice
            # strideslice at lower dims except last dim ==> reshape+narrow
            with torch.autograd.profiler.profile(use_device='npu') as prof:
                npu_out3 = npu_input.select(0, 1)[1:10:2].contiguous()
            self.assertEqual(check_operators_in_prof(['contiguous_h_match', 'contiguous_d_Slice'], prof)
                             or check_operators_in_prof(['aclnnInplaceCopy'], prof),
                             True, message="Error operators called!")
            cpu_out3 = cpu_input.select(0, 1)[1:10:2].contiguous()
            # strideslice at the last dim ==> cannot be optimized(contiguous_h_combined should not be called)
            with torch.autograd.profiler.profile(use_device='npu') as prof:
                npu_out4 = npu_input.select(0, 1)[:, :, ::3].contiguous()
            self.assertEqual(check_operators_in_prof(['contiguous_d_AsStrided'], prof, ['contiguous_h_combined'])
                             or check_operators_in_prof(['aclnnInplaceCopy'], prof, ['contiguous_h_combined']),
                             True, message="Error operators called!")
            cpu_out4 = cpu_input.select(0, 1)[:, :, ::3].contiguous()
            self.assertRtolEqual(npu_out3.to("cpu").numpy(), cpu_out3.numpy())
            self.assertRtolEqual(npu_out4.to("cpu").numpy(), cpu_out4.numpy())

    def test_broadcast_permute_contiguous(self):
        dtype_list7 = [np.float16, np.float32]
        format_list7 = [-1]
        shape_list7 = [
            [[2, 1, 3], [1, 2, 4, 3]],
            [[2, 1, 3], [5, 2, 4, 3]],
        ]
        shape_format7 = [
            [i, j, k] for i in dtype_list7 for j in format_list7 for k in shape_list7
        ]

        for item in shape_format7:
            item_broadcast = [item[0], item[1], item[2][0]]
            cpu_input, npu_input = create_common_tensor(item_broadcast, 0, 100)
            # Broadcast + permute all cannot be optimized(contiguous_h_combined should not be called)
            with torch.autograd.profiler.profile(use_device='npu') as prof:
                npu_out1 = npu_input.expand(item[2][1]).transpose(1, 3).contiguous()
            self.assertEqual(check_operators_in_prof(['contiguous_d_AsStrided'], prof, ['contiguous_h_combined'])
                             or check_operators_in_prof(['aclnnInplaceCopy'], prof, ['contiguous_h_combined']),
                             True, message="Error operators called!")
            cpu_out1 = cpu_input.expand(item[2][1]).transpose(1, 3).contiguous()
            self.assertRtolEqual(npu_out1.to("cpu").numpy(), cpu_out1.numpy())


if __name__ == "__main__":
    run_tests()
