import os
import torch
import numpy as np
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor, check_operators_in_prof

os.environ["COMBINED_ENABLE"] = "1"  # Open combined-view cases optimization

# Optimized view Ops contains Transpose, permute, narrow, strideslice, select, unfold


class SingleViewCopyToContiguous(TestCase):
    def test_broadcast_copy_contiguous(self, device="npu"):
        dtype_list = [np.float16, np.float32, np.int32, np.int8, np.uint8]
        format_list = [-1]
        shape_list = [
            [[1], [5]],
            [[1, 2], [3, 2]],
            [[1, 2, 1], [1, 2, 3]],
            [[1, 2, 1, 3], [4, 2, 5, 3]],
            [[1, 3], [1, 1, 4, 3]],
            [[1, 3], [2, 1, 4, 3]],
            [[1, 3], [1, 2, 4, 3]],
            [[3, 1], [2, 1, 3, 1]],
            [[3, 1], [1, 2, 3, 1]],
        ]
        shape_format = [
            [i, j, k] for i in dtype_list for j in format_list for k in shape_list
        ]

        for item in shape_format:
            item_broadcast = [item[0], item[1], item[2][0]]
            cpu_input, npu_input = create_common_tensor(item_broadcast, 0, 100)
            with torch.autograd.profiler.profile(use_device='npu') as prof:
                npu_out1 = npu_input.expand(item[2][1]).contiguous()
            self.assertEqual(check_operators_in_prof(['contiguous_d_BroadcastTo'], prof) or
                             check_operators_in_prof(['aclnnInplaceCopy'], prof),
                             True, message="contiguous_d_BroadcastTo or aclnnInplaceCopy is not called!")
            cpu_out1 = cpu_input.expand(item[2][1]).contiguous()
            self.assertRtolEqual(npu_out1.to("cpu").numpy(), cpu_out1.numpy())


if __name__ == "__main__":
    run_tests()
