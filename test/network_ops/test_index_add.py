import torch
import numpy as np
import torch_npu


from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor


class TestIndexAdd(TestCase):

    def cpu_op_exec(self, var, index, source, dim):
        output = var.index_add(dim=dim, index=index.long(), source=source)
        output = output.numpy()
        return output

    def npu_op_exec(self, var, index, source, dim):
        output = torch.index_add(var, dim, index.int(), source)
        output = output.to("cpu")
        output = output.numpy()
        return output

    def cpu_op_inter_exec(self, var, index, source, dim):
        output = var.index_add_(dim=dim, index=index.long(), source=source)
        output = output.numpy()
        return output

    def npu_op_inter_exec(self, var, index, source, dim):
        output = var.index_add_(dim, index.int(), source)
        output = output.to("cpu")
        output = output.numpy()
        return output

    def test_index_add_float32(self):
        shape_format = [
            [[np.float32, -1, (5, 3)], [np.int32, -1, (3, )], [np.float32, -1, (3, 3)], 0],
            [[np.float32, -1, (6, 4)], [np.int32, -1, (5, )], [np.float32, -1, (5, 4)], 0],
            [[np.float32, -1, (3, 2)], [np.int32, -1, (2, )], [np.float32, -1, (2, 2)], 0],
            [[np.float32, -1, (8, 6)], [np.int32, -1, (4, )], [np.float32, -1, (4, 6)], 0],
            [[np.float32, -1, (3, 5)], [np.int32, -1, (2, )], [np.float32, -1, (3, 2)], 1],
            [[np.float32, 4, (4, 6)], [np.int32, -1, (5, )], [np.float32, 4, (4, 5)], 1],
            [[np.float32, 3, (2, 3)], [np.int32, -1, (2, )], [np.float32, 3, (2, 2)], 1],
            [[np.float32, -1, (16, 7, 5, 9, 11)], [np.int32, -1, (11, )], [np.float32, -1, (16, 7, 5, 9, 11)], 4],
            [[np.float32, 3, (1600, 200)], [np.int32, -1, (200, )], [np.float32, 3, (1600, 200)], 1],
        ]
        for item in shape_format:
            cpu_var, npu_var = create_common_tensor(item[0], -10, 10)
            cpu_index, npu_index = create_common_tensor(item[1], 0, 2)
            cpu_source, npu_source = create_common_tensor(item[2], -10, 10)

            cpu_output = self.cpu_op_exec(cpu_var, cpu_index, cpu_source, item[3])
            npu_output = self.npu_op_exec(npu_var, npu_index, npu_source, item[3])
            self.assertRtolEqual(cpu_output, npu_output)
            cpu_output = self.cpu_op_inter_exec(cpu_var, cpu_index, cpu_source, item[3])
            npu_output = self.npu_op_inter_exec(npu_var, npu_index, npu_source, item[3])
            self.assertRtolEqual(cpu_output, npu_output)

    def test_index_add_int32(self):
        shape_format = [
            [[np.int32, -1, (5, 3)], [np.int32, -1, (3, )], [np.int32, -1, (3, 3)], 0],
            [[np.int32, -1, (6, 4)], [np.int32, -1, (5, )], [np.int32, -1, (5, 4)], 0],
            [[np.int32, -1, (3, 2)], [np.int32, -1, (2, )], [np.int32, -1, (2, 2)], 0],
            [[np.int32, -1, (3, 5)], [np.int32, -1, (2, )], [np.int32, -1, (3, 2)], 1],
            [[np.int32, -1, (4, 6)], [np.int32, -1, (5, )], [np.int32, -1, (4, 5)], 1],
            [[np.int32, -1, (2, 3)], [np.int32, -1, (2, )], [np.int32, -1, (2, 2)], 1],
            [[np.int32, -1, (16, 7, 5, 9, 11)], [np.int32, -1, (11, )], [np.int32, -1, (16, 7, 5, 9, 11)], 4],
            [[np.int32, -1, (1600, 200)], [np.int32, -1, (200, )], [np.int32, -1, (1600, 200)], 1],
        ]
        for item in shape_format:
            cpu_var, npu_var = create_common_tensor(item[0], -10, 10)
            cpu_index, npu_index = create_common_tensor(item[1], 0, 2)
            cpu_source, npu_source = create_common_tensor(item[2], -10, 10)

            cpu_output = self.cpu_op_exec(cpu_var, cpu_index, cpu_source, item[3])
            npu_output = self.npu_op_exec(npu_var, npu_index, npu_source, item[3])
            self.assertRtolEqual(cpu_output, npu_output)
            cpu_output = self.cpu_op_inter_exec(cpu_var, cpu_index, cpu_source, item[3])
            npu_output = self.npu_op_inter_exec(npu_var, npu_index, npu_source, item[3])
            self.assertRtolEqual(cpu_output, npu_output)

    def test_index_add_int8(self):
        shape_format = [
            [[np.int8, -1, (5, 3)], [np.int32, -1, (3, )], [np.int8, -1, (3, 3)], 0],
            [[np.int8, -1, (6, 4)], [np.int32, -1, (5, )], [np.int8, -1, (5, 4)], 0],
            [[np.int8, -1, (3, 2)], [np.int32, -1, (2, )], [np.int8, -1, (2, 2)], 0],
            [[np.int8, -1, (3, 5)], [np.int32, -1, (2, )], [np.int8, -1, (3, 2)], 1],
            [[np.int8, -1, (4, 6)], [np.int32, -1, (5, )], [np.int8, -1, (4, 5)], 1],
            [[np.int8, -1, (2, 3)], [np.int32, -1, (2, )], [np.int8, -1, (2, 2)], 1],
            [[np.int8, -1, (16, 7, 5, 9, 11)], [np.int32, -1, (11, )], [np.int8, -1, (16, 7, 5, 9, 11)], 4],
            [[np.int8, -1, (1600, 200)], [np.int32, -1, (200, )], [np.int8, -1, (1600, 200)], 1],
        ]
        for item in shape_format:
            cpu_var, npu_var = create_common_tensor(item[0], -10, 10)
            cpu_index = torch.arange(0, item[1][2][0])
            npu_index = cpu_index.npu()
            cpu_source, npu_source = create_common_tensor(item[2], -10, 10)

            cpu_output = self.cpu_op_exec(cpu_var, cpu_index, cpu_source, item[3])
            npu_output = self.npu_op_exec(npu_var, npu_index, npu_source, item[3])
            self.assertRtolEqual(cpu_output, npu_output)
            cpu_output = self.cpu_op_inter_exec(cpu_var, cpu_index, cpu_source, item[3])
            npu_output = self.npu_op_inter_exec(npu_var, npu_index, npu_source, item[3])
            self.assertRtolEqual(cpu_output, npu_output)

    def test_index_add_uint8(self):
        shape_format = [
            [[np.uint8, -1, (5, 3)], [np.int32, -1, (3, )], [np.uint8, -1, (3, 3)], 0],
            [[np.uint8, -1, (6, 4)], [np.int32, -1, (5, )], [np.uint8, -1, (5, 4)], 0],
            [[np.uint8, -1, (3, 2)], [np.int32, -1, (2, )], [np.uint8, -1, (2, 2)], 0],
            [[np.uint8, -1, (3, 5)], [np.int32, -1, (2, )], [np.uint8, -1, (3, 2)], 1],
            [[np.uint8, -1, (4, 6)], [np.int32, -1, (5, )], [np.uint8, -1, (4, 5)], 1],
            [[np.uint8, -1, (2, 3)], [np.int32, -1, (2, )], [np.uint8, -1, (2, 2)], 1],
            [[np.uint8, -1, (16, 7, 5, 9, 11)], [np.int32, -1, (11, )], [np.uint8, -1, (16, 7, 5, 9, 11)], 4],
            [[np.uint8, -1, (1600, 200)], [np.int32, -1, (200, )], [np.uint8, -1, (1600, 200)], 1],
        ]
        for item in shape_format:
            cpu_var, npu_var = create_common_tensor(item[0], 0, 10)
            cpu_index = torch.arange(0, item[1][2][0])
            npu_index = cpu_index.npu()
            cpu_source, npu_source = create_common_tensor(item[2], 0, 10)

            cpu_output = self.cpu_op_exec(cpu_var, cpu_index, cpu_source, item[3])
            npu_output = self.npu_op_exec(npu_var, npu_index, npu_source, item[3])
            self.assertRtolEqual(cpu_output, npu_output)
            cpu_output = self.cpu_op_inter_exec(cpu_var, cpu_index, cpu_source, item[3])
            npu_output = self.npu_op_inter_exec(npu_var, npu_index, npu_source, item[3])
            self.assertRtolEqual(cpu_output, npu_output)

    def test_index_add_fp16(self):
        shape_format = [
            [[np.float16, -1, (5, 3)], [np.int32, -1, (3, )], [np.float16, -1, (3, 3)], 0],
            [[np.float16, -1, (3, 2)], [np.int32, -1, (2, )], [np.float16, -1, (2, 2)], 0],
            [[np.float16, -1, (3, 5)], [np.int32, -1, (2, )], [np.float16, -1, (3, 2)], 1],
            [[np.float16, -1, (2, 6)], [np.int32, -1, (4, )], [np.float16, -1, (2, 4)], 1],
            [[np.float16, -1, (16, 7, 5, 9, 11)], [np.int32, -1, (11, )], [np.float16, -1, (16, 7, 5, 9, 11)], 4],
            [[np.float16, -1, (1600, 200)], [np.int32, -1, (200, )], [np.float16, -1, (1600, 200)], 1],
        ]
        for item in shape_format:
            cpu_var, npu_var = create_common_tensor(item[0], 0, 10)
            cpu_index = torch.arange(0, item[1][2][0])
            npu_index = cpu_index.npu()
            cpu_source, npu_source = create_common_tensor(item[2], 0, 10)

            cpu_var = cpu_var.to(torch.float32)
            cpu_source = cpu_source.to(torch.float32)

            cpu_output = self.cpu_op_exec(cpu_var, cpu_index, cpu_source, item[3])
            npu_output = self.npu_op_exec(npu_var, npu_index, npu_source, item[3])
            cpu_output = cpu_output.astype(np.float16)
            self.assertRtolEqual(cpu_output, npu_output)

            cpu_output = self.cpu_op_inter_exec(cpu_var, cpu_index, cpu_source, item[3])
            npu_output = self.npu_op_inter_exec(npu_var, npu_index, npu_source, item[3])
            cpu_output = cpu_output.astype(np.float16)
            self.assertRtolEqual(cpu_output, npu_output)

    def cpu_index_select_backward_0d_exec(self, x, index):
        x.requires_grad = True
        cpu_out = torch.index_select(x, 0, index)
        cpu_out.backward(torch.ones_like(cpu_out))
        cpu_grad = x.grad
        return cpu_out.detach(), cpu_grad

    def npu_index_select_backward_0d_exec(self, x, index):
        x.requires_grad = True
        npu_out = torch.index_select(x, 0, index)
        npu_out.backward(torch.ones_like(npu_out))
        npu_grad = x.grad
        return npu_out.cpu().detach(), npu_grad.cpu()

    def test_index_select_backward_0d(self):
        case = [np.float32, -1, (3, 4)]
        cpu_x, npu_x = create_common_tensor(case, -2, 2)
        cpu_index = torch.tensor([0]).squeeze()
        npu_index = cpu_index.npu()
        cpu_out, cpu_grad = self.cpu_index_select_backward_0d_exec(cpu_x, cpu_index)
        npu_out, npu_grad = self.npu_index_select_backward_0d_exec(npu_x, npu_index)
        self.assertRtolEqual(cpu_out, npu_out)
        self.assertRtolEqual(cpu_grad, npu_grad)


if __name__ == "__main__":
    run_tests()
