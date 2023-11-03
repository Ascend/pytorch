import torch
import numpy as np

import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor


class TestSub(TestCase):

    def cpu_op_exec(self, input1, input2):
        output = torch.sub(input1, input2)
        output = output.numpy()
        return output

    def cpu_op_exec_fp16(self, input1, input2):
        input1 = input1.to(torch.float32)
        input2 = input2.to(torch.float32)
        output = torch.sub(input1, input2)
        output = output.numpy()
        output = output.astype(np.float16)
        return output

    def npu_op_exec(self, input1, input2):
        output = torch.sub(input1, input2)
        output = output.to("cpu")
        output = output.numpy()
        return output

    def test_sub_common_shape_format(self):
        shape_format = [
            [[np.int32, -1, (2, 3)], [np.int32, -1, (2, 3)]],
            [[np.int32, -1, (500, 100)], [np.int32, -1, (500, 100)]],
            [[np.float32, -1, (4, 3)], [np.float32, -1, (4, 3)]],
            [[np.float32, -1, (4, 3, 5, 1)], [np.float32, -1, (4, 3, 5, 1)]],
            [[np.int32, -1, (4, 3)], [np.float32, -1, (4, 3)]],
            [[np.float32, -1, (4, 3)], [np.int32, -1, (4, 1)]]
        ]

        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item[0], -100, 100)
            cpu_input2, npu_input2 = create_common_tensor(item[1], -100, 100)
            cpu_output = self.cpu_op_exec(cpu_input1, cpu_input2)
            npu_output = self.npu_op_exec(npu_input1, npu_input2)
            self.assertRtolEqual(cpu_output, npu_output)

    def test_sub_float16_shape_format(self):
        shape_format = [
            [[np.float16, -1, (2, 3)], [np.float16, -1, (2, 3)]],
            [[np.float16, -1, (500, 100)], [np.float16, -1, (500, 100)]],
            [[np.float16, -1, (4, 3, 5, 1)], [np.float16, -1, (4, 3, 5, 1)]],
            [[np.float16, -1, (4, 3, 5, 1)], [np.float16, -1, (4, 3, 5, 1)]]
        ]

        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item[0], -100, 100)
            cpu_input2, npu_input2 = create_common_tensor(item[1], -100, 100)
            cpu_output = self.cpu_op_exec_fp16(cpu_input1, cpu_input2)
            npu_output = self.npu_op_exec(npu_input1, npu_input2)
            self.assertRtolEqual(cpu_output, npu_output)

    def test_sub_mix_dtype(self):
        dtype_list = [
            [np.int32, np.int64],
            [np.int64, np.int32],
            [np.float32, np.float16],
            [np.float16, np.float32],
            [np.int64, np.float32]
        ]
        for item in dtype_list:
            cpu_input1, npu_input1 = create_common_tensor([item[0], 0, (2, 3, 4)], -100, 100)
            cpu_input2, npu_input2 = create_common_tensor([item[1], 0, (2, 3, 4)], -100, 100)
            cpu_output = self.cpu_op_exec(cpu_input1, cpu_input2)
            npu_output = self.npu_op_exec(npu_input1, npu_input2)
            self.assertRtolEqual(cpu_output, npu_output)

    def test_sub_inplace_and_out_mix_dtype(self):
        dtype_list = [
            [np.int32, np.int64, np.int64],
            [np.int64, np.int32, np.int64],
            [np.float32, np.float16, np.float32],
            [np.float16, np.float32, np.float32],
            [np.int64, np.float32, np.float32]
        ]
        for item in dtype_list:
            cpu_input1, npu_input1 = create_common_tensor([item[0], 0, (2, 3, 4)], -100, 100)
            cpu_input2, npu_input2 = create_common_tensor([item[1], 0, (2, 3, 4)], -100, 100)
            _, npu_output = create_common_tensor([item[2], 0, (2, 3, 4)], 1, 100)

            if item[0] == np.int64 and item[1] == np.float32:
                try:
                    npu_input1.sub_(npu_input2)
                except RuntimeError as e:
                    self.assertRegex(
                        str(e), "result type Float can't be cast to the desired output type Long")
            else:
                cpu_input1.sub_(cpu_input2)
                npu_input1.sub_(npu_input2)
                self.assertRtolEqual(cpu_input1, npu_input1.cpu())

            cpu_output = torch.sub(cpu_input1, cpu_input2)
            torch.sub(npu_input1, npu_input2, out=npu_output)
            self.assertRtolEqual(cpu_output, npu_output.cpu())


if __name__ == "__main__":
    run_tests()
