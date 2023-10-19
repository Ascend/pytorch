import torch
import numpy as np
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor


class TestAtan2(TestCase):
    def cpu_op_exec(self, input1, input2):
        output = torch.atan2(input1, input2)
        output = output.numpy()
        return output

    def npu_op_exec(self, input1, input2):
        output = torch.atan2(input1, input2)
        output = output.to("cpu")
        output = output.numpy()
        return output

    def npu_op_exec_out(self, input1, input2, out):
        torch.atan2(input1, input2, out=out)
        output = out.to("cpu")
        output = output.numpy()
        return output

    def test_atan2_common_shape_format(self):
        shape_format = [
            [[np.float16, 0, [4, 12, 12, 128]], [np.float16, 0, [4]]],
            [[np.float16, 0, [4, 128]], [np.float16, 0, [4, 256, 12]]],
            [[np.float32, 0, [4, 12, 12, 128]], [np.float32, 0, [4]]],
            [[np.float32, 0, [4, 128]], [np.float32, 0, [4, 256, 12]]],
            [[np.float16, 2, [4, 12, 12, 128]], [np.float16, 0, [4]]],
            [[np.float16, 3, [4, 128]], [np.float16, 0, [4, 256, 12]]],
            [[np.float32, 2, [4, 12, 12, 128]], [np.float32, 0, [4]]],
            [[np.float32, 3, [4, 128]], [np.float32, 0, [4, 256, 12]]],
        ]
        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item[0], -1, 1)
            cpu_input2, npu_input2 = create_common_tensor(item[0], -1, 1)
            cpu_out, npu_out = create_common_tensor(item[1], -1, 1)
            if cpu_input1.dtype == torch.float16:
                cpu_input1 = cpu_input1.to(torch.float32)
            if cpu_input2.dtype == torch.float16:
                cpu_input2 = cpu_input2.to(torch.float32)
            cpu_output = self.cpu_op_exec(cpu_input1, cpu_input2)
            npu_output = self.npu_op_exec(npu_input1, npu_input2)
            npu_output_out = self.npu_op_exec_out(npu_input1, npu_input2, npu_out)
            cpu_output = cpu_output.astype(npu_output.dtype)
            self.assertRtolEqual(cpu_output, npu_output)
            self.assertRtolEqual(cpu_output, npu_output_out)

    def test_atan2_mix_dtype(self):
        npu_input1, npu_input2 = create_common_tensor([np.float32, 0, (2, 3)], 1, 100)
        npu_input3, npu_input4 = create_common_tensor([np.float16, 0, (2, 3)], 1, 100)
        cpu_output = self.cpu_op_exec(npu_input1, npu_input3)
        npu_output = self.npu_op_exec(npu_input2, npu_input4)
        self.assertRtolEqual(cpu_output, npu_output)


if __name__ == "__main__":
    run_tests()
