import torch
import numpy as np
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor


class TestCat(TestCase):

    def cpu_op_exec(self, input1, input2, n):
        output = torch.cat(input1 + input2, n)
        if not(output.is_contiguous()):
            output = output.contiguous()
        output = output.numpy()
        return output

    def npu_op_exec(self, input1, input2, n):
        output = torch.cat(input1 + input2, n)
        if not(output.is_contiguous()):
            output = output.contiguous()
        output = output.to("cpu")
        output = output.numpy()
        return output

    def test_cat_shape_format_fp16_3d(self):
        format_list = [0, 3, 29]
        shape_list = [(256, 32, 56)]
        shape_format = [
            [np.float16, i, j] for i in format_list for j in shape_list
        ]
        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item, 0, 100)
            cpu_input2, npu_input2 = create_common_tensor(item, 0, 100)
            if cpu_input1.dtype == torch.float16:
                cpu_input1 = cpu_input1.to(torch.float32)
                cpu_input2 = cpu_input2.to(torch.float32)
            cpu_output = self.cpu_op_exec([cpu_input1], [cpu_input2], 1)
            npu_output = self.npu_op_exec([npu_input1], [npu_input2], 1)
            cpu_output = cpu_output.astype(npu_output.dtype)
            self.assertRtolEqual(cpu_output, npu_output)

    def test_cat_shape_format_fp32_3d(self):
        format_list = [0, 3, 29]
        shape_list = [(256, 32, 56)]
        shape_format = [
            [np.float32, i, j] for i in format_list for j in shape_list
        ]
        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item, 0, 100)
            cpu_input2, npu_input2 = create_common_tensor(item, 0, 100)
            if cpu_input1.dtype == torch.float16:
                cpu_input1 = cpu_input1.to(torch.float32)
                cpu_input2 = cpu_input2.to(torch.float32)
            cpu_output = self.cpu_op_exec([cpu_input1], [cpu_input2], 1)
            npu_output = self.npu_op_exec([npu_input1], [npu_input2], 1)
            cpu_output = cpu_output.astype(npu_output.dtype)
            self.assertRtolEqual(cpu_output, npu_output)

    def test_cat_shape_format_fp16_4d(self):
        format_list = [0, 3, 29]
        shape_list = [(256, 32, 56, 56)]
        shape_format = [
            [np.float16, i, j] for i in format_list for j in shape_list
        ]
        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item, 0, 100)
            cpu_input2, npu_input2 = create_common_tensor(item, 0, 100)
            if cpu_input1.dtype == torch.float16:
                cpu_input1 = cpu_input1.to(torch.float32)
                cpu_input2 = cpu_input2.to(torch.float32)
            cpu_output = self.cpu_op_exec([cpu_input1], [cpu_input2], 1)
            npu_output = self.npu_op_exec([npu_input1], [npu_input2], 1)
            cpu_output = cpu_output.astype(npu_output.dtype)
            self.assertRtolEqual(cpu_output, npu_output)

    def test_cat_shape_format_fp32_4d(self):
        format_list = [0, 3, 29]
        shape_list = [(256, 32, 56, 56)]
        shape_format = [
            [np.float32, i, j] for i in format_list for j in shape_list
        ]
        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item, 0, 100)
            cpu_input2, npu_input2 = create_common_tensor(item, 0, 100)
            if cpu_input1.dtype == torch.float16:
                cpu_input1 = cpu_input1.to(torch.float32)
                cpu_input2 = cpu_input2.to(torch.float32)
            cpu_output = self.cpu_op_exec([cpu_input1], [cpu_input2], 1)
            npu_output = self.npu_op_exec([npu_input1], [npu_input2], 1)
            cpu_output = cpu_output.astype(npu_output.dtype)
            self.assertRtolEqual(cpu_output, npu_output)

    def test_cat_shape_format_fp16_2d(self):
        format_list = [0, 3, 29]
        shape_list = [(56, 56)]
        shape_format = [
            [np.float16, i, j] for i in format_list for j in shape_list
        ]
        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item, 0, 100)
            cpu_input2, npu_input2 = create_common_tensor(item, 0, 100)
            if cpu_input1.dtype == torch.float16:
                cpu_input1 = cpu_input1.to(torch.float32)
                cpu_input2 = cpu_input2.to(torch.float32)
            cpu_output = self.cpu_op_exec([cpu_input1], [cpu_input2], 1)
            npu_output = self.npu_op_exec([npu_input1], [npu_input2], 1)
            cpu_output = cpu_output.astype(npu_output.dtype)
            self.assertRtolEqual(cpu_output, npu_output)

    def test_cat_shape_format_fp32_2d(self):
        format_list = [0, 3, 29]
        shape_list = [(56, 56)]
        shape_format = [
            [np.float32, i, j] for i in format_list for j in shape_list
        ]
        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item, 0, 100)
            cpu_input2, npu_input2 = create_common_tensor(item, 0, 100)
            if cpu_input1.dtype == torch.float16:
                cpu_input1 = cpu_input1.to(torch.float32)
                cpu_input2 = cpu_input2.to(torch.float32)
            cpu_output = self.cpu_op_exec([cpu_input1], [cpu_input2], 1)
            npu_output = self.npu_op_exec([npu_input1], [npu_input2], 1)
            cpu_output = cpu_output.astype(npu_output.dtype)
            self.assertRtolEqual(cpu_output, npu_output)

    def test_cat_null_tensor(self):
        x1 = torch.randn(15, 2, 1, 1)
        x2 = torch.randn(0, 2, 1, 1)
        x3 = torch.randn(0, 2, 3, 1)
        y1_cpu = torch.cat([x1, x2], dim=0)
        y2_cpu = torch.cat([x2, x3], dim=2)
        y3_cpu = torch.cat([x2, x2, x2], dim=1)
        x1 = x1.npu()
        x2 = x2.npu()
        x3 = x3.npu()
        y1_npu = torch.cat([x1, x2], dim=0)
        y2_npu = torch.cat([x2, x3], dim=2)
        y3_npu = torch.cat([x2, x2, x2], dim=1)
        self.assertRtolEqual(y1_cpu, y1_npu.cpu())
        self.assertRtolEqual(y2_cpu, y2_npu.cpu())
        self.assertRtolEqual(y3_cpu, y3_npu.cpu())

    def test_cat_different_dtype(self):
        dtype_list = [np.float16, np.float32, np.int32, np.int64]
        shape_format = [
            [[i, -1, (56, 56)], [j, -1, (56, 56)]]
            for i in dtype_list
            for j in dtype_list
        ]
        dim = 1
        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item[0], 0, 100)
            cpu_input2, npu_input2 = create_common_tensor(item[1], 0, 100)
            cpu_output = self.cpu_op_exec([cpu_input1], [cpu_input2], dim)
            npu_output = self.npu_op_exec([npu_input1], [npu_input2], dim)
            self.assertRtolEqual(cpu_output, npu_output)

    def test_cat_null(self):
        cpu_x = torch.tensor([])
        npu_x = cpu_x.npu()
        cpu_y = torch.cat([cpu_x, cpu_x], 0)
        npu_y = torch.cat([npu_x, npu_x], 0)

        cpu_out = torch.rand(2, 3, 4)
        npu_out = cpu_out.npu()
        torch.cat([cpu_x, cpu_x], 0, out=cpu_out)
        torch.cat([npu_x, npu_x], 0, out=npu_out)

        self.assertRtolEqual(cpu_y, npu_y.cpu())
        self.assertRtolEqual(cpu_out, npu_out.cpu())


class TestConCat(TestCase):

    def cpu_op_exec(self, input1, input2, n):
        output = torch.concat(input1 + input2, n)
        if not(output.is_contiguous()):
            output = output.contiguous()
        output = output.numpy()
        return output

    def npu_op_exec(self, input1, input2, n):
        output = torch.concat(input1 + input2, n)
        if not(output.is_contiguous()):
            output = output.contiguous()
        output = output.to("cpu")
        output = output.numpy()
        return output

    def test_cat_shape_format_fp16_3d(self):
        format_list = [0, 3, 29]
        shape_list = [(256, 32, 56)]
        shape_format = [
            [np.float16, i, j] for i in format_list for j in shape_list
        ]
        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item, 0, 100)
            cpu_input2, npu_input2 = create_common_tensor(item, 0, 100)
            if cpu_input1.dtype == torch.float16:
                cpu_input1 = cpu_input1.to(torch.float32)
                cpu_input2 = cpu_input2.to(torch.float32)
            cpu_output = self.cpu_op_exec([cpu_input1], [cpu_input2], 1)
            npu_output = self.npu_op_exec([npu_input1], [npu_input2], 1)
            cpu_output = cpu_output.astype(npu_output.dtype)
            self.assertRtolEqual(cpu_output, npu_output)

    def test_cat_shape_format_fp32_3d(self):
        format_list = [0, 3, 29]
        shape_list = [(256, 32, 56)]
        shape_format = [
            [np.float32, i, j] for i in format_list for j in shape_list
        ]
        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item, 0, 100)
            cpu_input2, npu_input2 = create_common_tensor(item, 0, 100)
            if cpu_input1.dtype == torch.float16:
                cpu_input1 = cpu_input1.to(torch.float32)
                cpu_input2 = cpu_input2.to(torch.float32)
            cpu_output = self.cpu_op_exec([cpu_input1], [cpu_input2], 1)
            npu_output = self.npu_op_exec([npu_input1], [npu_input2], 1)
            cpu_output = cpu_output.astype(npu_output.dtype)
            self.assertRtolEqual(cpu_output, npu_output)

    def test_cat_shape_format_fp16_4d(self):
        format_list = [0, 3, 29]
        shape_list = [(256, 32, 56, 56)]
        shape_format = [
            [np.float16, i, j] for i in format_list for j in shape_list
        ]
        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item, 0, 100)
            cpu_input2, npu_input2 = create_common_tensor(item, 0, 100)
            if cpu_input1.dtype == torch.float16:
                cpu_input1 = cpu_input1.to(torch.float32)
                cpu_input2 = cpu_input2.to(torch.float32)
            cpu_output = self.cpu_op_exec([cpu_input1], [cpu_input2], 1)
            npu_output = self.npu_op_exec([npu_input1], [npu_input2], 1)
            cpu_output = cpu_output.astype(npu_output.dtype)
            self.assertRtolEqual(cpu_output, npu_output)

    def test_cat_shape_format_fp32_4d(self):
        format_list = [0, 3, 29]
        shape_list = [(256, 32, 56, 56)]
        shape_format = [
            [np.float32, i, j] for i in format_list for j in shape_list
        ]
        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item, 0, 100)
            cpu_input2, npu_input2 = create_common_tensor(item, 0, 100)
            if cpu_input1.dtype == torch.float16:
                cpu_input1 = cpu_input1.to(torch.float32)
                cpu_input2 = cpu_input2.to(torch.float32)
            cpu_output = self.cpu_op_exec([cpu_input1], [cpu_input2], 1)
            npu_output = self.npu_op_exec([npu_input1], [npu_input2], 1)
            cpu_output = cpu_output.astype(npu_output.dtype)
            self.assertRtolEqual(cpu_output, npu_output)

    def test_cat_shape_format_fp16_2d(self):
        format_list = [0, 3, 29]
        shape_list = [(56, 56)]
        shape_format = [
            [np.float16, i, j] for i in format_list for j in shape_list
        ]
        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item, 0, 100)
            cpu_input2, npu_input2 = create_common_tensor(item, 0, 100)
            if cpu_input1.dtype == torch.float16:
                cpu_input1 = cpu_input1.to(torch.float32)
                cpu_input2 = cpu_input2.to(torch.float32)
            cpu_output = self.cpu_op_exec([cpu_input1], [cpu_input2], 1)
            npu_output = self.npu_op_exec([npu_input1], [npu_input2], 1)
            cpu_output = cpu_output.astype(npu_output.dtype)
            self.assertRtolEqual(cpu_output, npu_output)

    def test_cat_shape_format_fp32_2d(self):
        format_list = [0, 3, 29]
        shape_list = [(56, 56)]
        shape_format = [
            [np.float32, i, j] for i in format_list for j in shape_list
        ]
        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item, 0, 100)
            cpu_input2, npu_input2 = create_common_tensor(item, 0, 100)
            if cpu_input1.dtype == torch.float16:
                cpu_input1 = cpu_input1.to(torch.float32)
                cpu_input2 = cpu_input2.to(torch.float32)
            cpu_output = self.cpu_op_exec([cpu_input1], [cpu_input2], 1)
            npu_output = self.npu_op_exec([npu_input1], [npu_input2], 1)
            cpu_output = cpu_output.astype(npu_output.dtype)
            self.assertRtolEqual(cpu_output, npu_output)

    def test_cat_null_tensor(self):
        x1 = torch.randn(15, 2, 1, 1)
        x2 = torch.randn(0, 2, 1, 1)
        x3 = torch.randn(0, 2, 3, 1)
        y1_cpu = torch.concat([x1, x2], dim=0)
        y2_cpu = torch.concat([x2, x3], dim=2)
        y3_cpu = torch.concat([x2, x2, x2], dim=1)
        x1 = x1.npu()
        x2 = x2.npu()
        x3 = x3.npu()
        y1_npu = torch.concat([x1, x2], dim=0)
        y2_npu = torch.concat([x2, x3], dim=2)
        y3_npu = torch.concat([x2, x2, x2], dim=1)
        self.assertRtolEqual(y1_cpu, y1_npu.cpu())
        self.assertRtolEqual(y2_cpu, y2_npu.cpu())
        self.assertRtolEqual(y3_cpu, y3_npu.cpu())

    def test_cat_different_dtype(self):
        dtype_list = [np.float16, np.float32, np.int32, np.int64]
        shape_format = [
            [[i, -1, (56, 56)], [j, -1, (56, 56)]]
            for i in dtype_list
            for j in dtype_list
        ]
        dim = 1
        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item[0], 0, 100)
            cpu_input2, npu_input2 = create_common_tensor(item[1], 0, 100)
            cpu_output = self.cpu_op_exec([cpu_input1], [cpu_input2], dim)
            npu_output = self.npu_op_exec([npu_input1], [npu_input2], dim)
            self.assertRtolEqual(cpu_output, npu_output)

    def test_cat_null(self):
        cpu_x = torch.tensor([])
        npu_x = cpu_x.npu()
        cpu_y = torch.concat([cpu_x, cpu_x], 0)
        npu_y = torch.concat([npu_x, npu_x], 0)

        cpu_out = torch.rand(2, 3, 4)
        npu_out = cpu_out.npu()
        torch.concat([cpu_x, cpu_x], 0, out=cpu_out)
        torch.concat([npu_x, npu_x], 0, out=npu_out)

        self.assertRtolEqual(cpu_y, npu_y.cpu())
        self.assertRtolEqual(cpu_out, npu_out.cpu())

if __name__ == "__main__":
    run_tests()
