import unittest
import torch
import numpy as np
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor


class TestEye(TestCase):

    def cpu_op_exec(self, shapes):
        if shapes[0] == shapes[1]:
            output = torch.eye(shapes[0])
        else:
            output = torch.eye(shapes[0], shapes[1])
        output = output.numpy()
        return output

    def npu_op_exec(self, shapes):
        if shapes[0] == shapes[1]:
            output = torch.eye(shapes[0], device="npu")
        else:
            output = torch.eye(shapes[0], shapes[1], device="npu")
        output = output.to("cpu")
        output = output.numpy()
        return output

    def cpu_op_out_exec(self, shapes, out):
        if shapes[0] == shapes[1]:
            torch.eye(shapes[0], out=out)
        else:
            torch.eye(shapes[0], shapes[1], out=out)
        output = out.numpy()
        return output

    def npu_op_out_exec(self, shapes, out):
        out = out.to("npu")
        if shapes[0] == shapes[1]:
            torch.eye(shapes[0], out=out)
        else:
            torch.eye(shapes[0], shapes[1], out=out)
        output = out.to("cpu")
        output = output.numpy()
        return output

    @unittest.skip("skip test_eye_int32_common_shape_format now")
    def test_eye_int32_common_shape_format(self):
        shape_format = [
            [np.int32, 0, (3563, 4000)],
            [np.int32, 0, (1350, 1762)],
        ]
        for item in shape_format:
            cpu_output = self.cpu_op_exec(item[2])
            npu_output = self.npu_op_exec(item[2])
            self.assertRtolEqual(cpu_output, npu_output)

    def test_eye_bool_common_shape_format(self):
        shape_format = [
            [torch.bool, 3, 3],
            [torch.bool, 5, 6]
        ]
        for item in shape_format:
            cpu_output = torch.eye(item[1], item[2], dtype=item[0], device="cpu")
            npu_output = torch.eye(item[1], item[2], dtype=item[0], device="npu")
            self.assertRtolEqual(cpu_output, npu_output.cpu())

    @unittest.skip("skip test_eye_float32_common_shape_format now")
    def test_eye_float32_common_shape_format(self):
        shape_format = [
            [np.float32, 0, (5, 5)],
            [np.float32, 0, (15, 15)],
            [np.float32, 0, (3, 5)],
            [np.float32, 0, (40, 5)],
            [np.float32, 0, (16480, 25890)],
            [np.float32, 0, (1350, 1762)],
            [np.float32, 0, (352, 4000)],
            [np.float32, 0, (3563, 4000)],
            [np.float32, 0, (1, 51)],
            [np.float32, 0, (1, 173)],
            [np.float32, 0, (1, 45000)],
            [np.float32, 0, (1, 100000)],
        ]
        for item in shape_format:
            cpu_output = self.cpu_op_exec(item[2])
            npu_output = self.npu_op_exec(item[2])
            self.assertRtolEqual(cpu_output, npu_output)

    @unittest.skip("skip test_eye_out_float32_common_shape_format now")
    def test_eye_out_float32_common_shape_format(self):
        shape_format = [
            [np.float32, 0, (5, 5)],
            [np.float32, 0, (3, 5)],
            [np.float32, 0, (1350, 1762)],
            [np.float32, 0, (352, 4000)],
            [np.float32, 0, (3563, 4000)],
            [np.float32, 0, (40000, 40000)]
        ]
        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item, 1, 100)
            cpu_output = self.cpu_op_out_exec(item[2], cpu_input1)
            npu_output = self.npu_op_out_exec(item[2], npu_input1)
            self.assertRtolEqual(cpu_output, npu_output)

    @unittest.skip("skip test_eye_out_float32_different_shape_format now")
    def test_eye_out_float32_different_shape_format(self):
        shape_1 = [np.float32, 0, (4000, 400)]
        shape_2 = [np.float32, 0, (4000, 4000)]
        cpu_input1 = torch.randn(shape_1[2][0], shape_1[2][1], dtype=torch.float32)
        cpu_output = self.cpu_op_out_exec(shape_2[2], cpu_input1)
        npu_input1 = torch.randn(shape_2[2][0], shape_2[2][1], dtype=torch.float32)
        npu_output = self.npu_op_out_exec(shape_2[2], npu_input1)
        self.assertRtolEqual(cpu_output, npu_output)

    def test_eye_float16_shape_format(self):
        def cpu_op_exec_fp16(shapes):
            output = torch.eye(shapes[0], shapes[1])
            output = output.numpy()
            output = output.astype(np.float16)
            return output

        def npu_op_exec_fp16(shapes):
            output = torch.eye(shapes[0], shapes[1], device="npu", dtype=torch.float16)
            output = output.to("cpu")
            output = output.numpy()
            return output

        shape_format = [
            [np.float16, 0, (5, 5)],
            [np.float16, 0, (3, 5)],
            [np.float32, 0, (1350, 1762)],
            [np.float32, 0, (352, 4000)],
            [np.float32, 0, (3563, 4000)]
        ]

        for item in shape_format:
            cpu_output = cpu_op_exec_fp16(item[2])
            npu_output = npu_op_exec_fp16(item[2])
            self.assertRtolEqual(cpu_output, npu_output)


if __name__ == "__main__":
    run_tests()
