import torch
import numpy as np

import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor


def cpu_op_exec(input1, func):
    output = func(input1)
    output = output.numpy()
    return output


def npu_op_exec(input1, func):
    input1 = input1.to("npu")
    if func == torch._cast_Byte:
        output = func(input1.int())
    else:
        output = func(input1)
    output = output.to("cpu")
    output = output.numpy()
    return output


shape_format = [
    [[np.bool_, -1, (4, 3, 1)]],
    [[np.int64, -1, (4, 3)]],
    [[np.int32, -1, (4, 3, 1)]],
    [[np.int8, -1, (2, 3)]],
    [[np.float32, -1, (4, 3, 1)]],
    [[np.float16, -1, (4, 3, 1)]],
    [[np.uint8, -1, (4, 3, 1)]]
]


class TestCast(TestCase):
    def test__cast_Byte_common_shape_format(self):
        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item[0], -100, 100)
            cpu_output = cpu_op_exec(cpu_input1, torch._cast_Byte)
            npu_output = npu_op_exec(npu_input1, torch._cast_Byte)
            self.assertEqual(cpu_output, npu_output)

    def test_cast_Char_common_shape_format(self):
        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item[0], -100, 100)
            cpu_output = cpu_op_exec(cpu_input1, torch._cast_Char)
            npu_output = npu_op_exec(npu_input1, torch._cast_Char)
            self.assertEqual(cpu_output, npu_output)

    def test_cast_Float_common_shape_format(self):
        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item[0], -100, 100)
            cpu_output = cpu_op_exec(cpu_input1, torch._cast_Float)
            npu_output = npu_op_exec(npu_input1, torch._cast_Float)
            self.assertRtolEqual(cpu_output, npu_output)

    def test_cast_Half(self):
        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item[0], -100, 100)
            if cpu_input1.dtype == torch.float16:
                cpu_input1 = cpu_input1.to(torch.float32)
            cpu_output = cpu_op_exec(cpu_input1, torch._cast_Half)
            npu_output = npu_op_exec(npu_input1, torch._cast_Half)
            self.assertRtolEqual(cpu_output, npu_output)

    def test_cast_Int(self):
        for item in shape_format:
            cpu_input1, _ = create_common_tensor(item[0], -100, 100)
            if cpu_input1.dtype == torch.float16:
                cpu_input1 = cpu_input1.to(torch.float32)
            cpu_output = cpu_op_exec(cpu_input1, torch._cast_Int)
            npu_output = npu_op_exec(cpu_input1, torch._cast_Int)
            self.assertRtolEqual(cpu_output, npu_output)

    def test_cast_Long_common_shape_format(self):
        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item[0], -100, 100)
            cpu_output = cpu_op_exec(cpu_input1, torch._cast_Long)
            npu_output = npu_op_exec(npu_input1, torch._cast_Long)
            self.assertRtolEqual(cpu_output, npu_output)

    def test_cast_Short_common_shape_format(self):
        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item[0], -100, 100)
            cpu_output = cpu_op_exec(cpu_input1, torch._cast_Short)
            npu_output = npu_op_exec(npu_input1, torch._cast_Short)
            self.assertRtolEqual(cpu_output, npu_output)


if __name__ == "__main__":
    run_tests()
