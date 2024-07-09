import numpy as np
import torch

import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor


class TestMax(TestCase):

    def supported_op_exec(self, input1, dim, keepdim):
        if isinstance(dim, str):
            input1.names = ('N', 'C', 'H', 'W')
            dim = input1.names.index(dim)
        values, indices = torch.max(input1, dim, keepdim)
        indices = indices.to(torch.int32)
        return values.cpu().detach(), indices.cpu().detach()

    def custom_op_exec(self, input1, dim, keepdim):
        values, indices = torch_npu.npu_max(input1, dim, keepdim)
        return values.cpu().detach(), indices.cpu().detach()

    def test_npu_max(self, device="npu"):
        item = [np.float32, 0, (2, 2, 2, 2)]
        _, npu_input = create_common_tensor(item, -1, 1)
        dims = (2,)
        keepdim = False

        for dim in dims:
            supported_values, supported_indices = self.supported_op_exec(npu_input, dim, keepdim)
            custom_values, custom_indices = self.custom_op_exec(npu_input, dim, keepdim)
            self.assertRtolEqual(supported_values, custom_values)
            self.assertRtolEqual(supported_indices, custom_indices)


if __name__ == "__main__":
    run_tests()
