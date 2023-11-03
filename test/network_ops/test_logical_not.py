import torch
import numpy as np
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor


class TestLogicalNot(TestCase):

    def cpu_op_exec(self, input1):
        output = torch.logical_not(input1)
        output = output.numpy()
        return output

    def npu_op_exec(self, input1):
        output = torch.logical_not(input1)
        output = output.to("cpu")
        output = output.numpy()
        return output

    def npu_op_exec_out(self, input1, output):
        output = torch.logical_not(input1, out=output)
        output = output.to("cpu")
        output = output.numpy()
        return output

    def npu_op_exec_inplace(self, input1):
        input1.logical_not_()
        output = input1.cpu()
        output = output.numpy()
        return output

    def test_logical_not_common_shape_format(self):
        shape_format = [
            [[np.int8, -1, 1]],
            [[np.int8, -1, (64, 10)]],
            [[np.int8, -1, (256, 2048, 7, 7)]],
            [[np.int8, -1, (32, 1, 3, 3)]],
            [[np.int32, -1, (64, 10)]],
            [[np.int32, -1, (256, 2048, 7, 7)]],
            [[np.int32, -1, (32, 1, 3, 3)]],
            [[np.uint8, -1, (64, 10)]],
            [[np.uint8, -1, (256, 2048, 7, 7)]],
            [[np.uint8, -1, (32, 1, 3, 3)]],
            [[np.float16, -1, (64, 10)]],
            [[np.float16, -1, (256, 2048, 7, 7)]],
            [[np.float16, -1, (32, 1, 3, 3)]],
            [[np.float32, -1, (64, 10)]],
            [[np.float32, -1, (256, 2048, 7, 7)]],
            [[np.float32, -1, (32, 1, 3, 3)]],
            [[np.bool_, -1, (64, 10)]],
            [[np.bool_, -1, (256, 2048, 7, 7)]]
        ]
        for item in shape_format:
            cpu_input, npu_input = create_common_tensor(item[0], 1, 10)
            cpu_output = self.cpu_op_exec(cpu_input)
            npu_output = self.npu_op_exec(npu_input)
            self.assertRtolEqual(cpu_output, npu_output)

    def test_logical_not_out_common_shape_format(self):
        shape_format = [
            [[np.float16, -1, (64, 10)], [np.float16, -1, (64, 1)]],
            [[np.float16, -1, (256, 2048, 7, 7)], [np.float16, -1, (256, 2048, 7)]],
            [[np.float16, -1, (32, 1, 3, 3)], [np.float16, -1, (32, 1, 3, 3)]],
            [[np.float32, -1, (64, 10)], [np.float32, -1, (64, 1)]],
            [[np.float32, -1, (256, 2048, 7, 7)], [np.float32, -1, (256, 2048, 7)]],
            [[np.float32, -1, (32, 1, 3, 3)], [np.float32, -1, (32, 1, 3, 3)]],
            [[np.bool_, -1, (64, 10)], [np.bool_, -1, (64, 10)]],
            [[np.bool_, -1, (256, 2048, 7, 7)], [np.bool_, -1, (256, 2048, 7, 7)]]
        ]
        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item[0], 1, 10)
            cpu_input2, npu_input2 = create_common_tensor(item[1], 1, 10)
            cpu_output = self.cpu_op_exec(cpu_input1)
            npu_output = self.npu_op_exec_out(npu_input1, npu_input2)
            cpu_output = cpu_output.astype(npu_output.dtype)
            self.assertRtolEqual(cpu_output, npu_output)

    def test_logical_not_inplace_common_shape_format(self):
        shape_format = [
            [[np.int8, -1, 1]],
            [[np.int8, -1, (64, 10)]],
            [[np.int8, -1, (256, 2048, 7, 7)]],
            [[np.int32, -1, (64, 10)]],
            [[np.int32, -1, (256, 2048, 7, 7)]],
            [[np.int32, -1, (32, 1, 3, 3)]],
            [[np.uint8, -1, (64, 10)]],
            [[np.uint8, -1, (256, 2048, 7, 7)]],
            [[np.uint8, -1, (32, 1, 3, 3)]],
            [[np.float16, -1, (64, 10)]],
            [[np.float16, 2, (256, 2048, 7, 7)]],
            [[np.float16, -1, (32, 1, 3, 3)]],
            [[np.float32, 0, (64, 10)]],
            [[np.float32, -1, (256, 2048, 7, 7)]],
            [[np.float32, -1, (32, 1, 3, 3)]],
            [[np.bool_, -1, (64, 10)]],
            [[np.bool_, -1, (256, 2048, 7, 7)]]
        ]
        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item[0], 1, 10)
            cpu_output = self.cpu_op_exec(cpu_input1)
            npu_output = self.npu_op_exec_inplace(npu_input1)
            cpu_output = cpu_output.astype(npu_output.dtype)
            self.assertRtolEqual(cpu_output, npu_output)


if __name__ == "__main__":
    run_tests()
