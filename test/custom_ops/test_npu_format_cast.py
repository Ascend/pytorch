import numpy as np
import torch

import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor


class TestFormatCast(TestCase):

    def supported_op_exec(self, input1):
        m = torch.nn.Identity(54, unused_argument1=0.1, unused_argument2=False)
        output = m(input1)
        return output.cpu().detach()

    def custom_op_exec(self, input1, acl_format):
        output = torch_npu.npu_format_cast(input1, acl_format)
        return output.cpu().detach()

    def test_npu_format_cast(self, device="npu"):
        item = [np.float16, 0, (2, 2, 4, 4)]
        _, npu_input = create_common_tensor(item, -1, 1)
        acl_format = 3

        supported_output = self.supported_op_exec(npu_input)
        custom_output = self.custom_op_exec(npu_input, acl_format)
        self.assertRtolEqual(supported_output, custom_output)


if __name__ == "__main__":
    run_tests()
