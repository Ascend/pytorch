import torch
import numpy as np
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor


class TestChannelShuffle(TestCase):
    def cpu_op_exec(self, input1, group):
        output = torch.channel_shuffle(input1, group)
        output = output.numpy()
        return output

    def npu_op_exec(self, input1, group):
        output = torch.channel_shuffle(input1, group)
        output = output.to("cpu")
        output = output.numpy()
        return output

    def test_channel_shuffle_fp32(self):
        format_list = [0, 3, 4]
        shape_format = [[np.float32, i, [16, 640, 640]] for i in format_list]
        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item, 0, 100)
            if cpu_input1.dtype == torch.float16:
                cpu_input1 = cpu_input1.to(torch.float32)
            group = 2
            cpu_output = self.cpu_op_exec(cpu_input1, group)
            npu_output = self.npu_op_exec(npu_input1, group)

            self.assertRtolEqual(cpu_output, npu_output)

    def test_channel_shuffle_fp16(self):
        format_list = [0, 3, 4]
        shape_format = [[np.float16, i, [16, 640, 640]] for i in format_list]
        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item, 0, 100)
            if cpu_input1.dtype == torch.float16:
                cpu_input1 = cpu_input1.to(torch.float32)
            group = 2
            cpu_output = self.cpu_op_exec(cpu_input1, group)
            npu_output = self.npu_op_exec(npu_input1, group)
            cpu_output = cpu_output.astype(npu_output.dtype)
            self.assertRtolEqual(cpu_output, npu_output)


if __name__ == '__main__':
    run_tests()
