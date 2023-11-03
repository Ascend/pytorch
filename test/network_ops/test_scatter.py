import torch
import numpy as np
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor


class TestScatter(TestCase):
    def cpu_op_exec(self, shape, dim, index, src):
        input1 = torch.zeros(shape)
        cpu_output = input1.scatter(dim, index, src)
        return cpu_output.numpy()

    def npu_op_exec(self, shape, dim, index, src, isTensor=True):
        input1 = torch.zeros(shape).npu()
        index = index.npu()
        if (isTensor):
            src = src.npu()
        npu_output = input1.scatter(dim, index, src)
        npu_output = npu_output.cpu()
        return npu_output.numpy()

    def cpu_op_exec_inplace(self, shape, dim, index, src):
        input1 = torch.zeros(shape)
        input1.scatter_(dim, index, src)
        return input1.numpy()

    def npu_op_exec_inplace(self, shape, dim, index, src, isTensor=True):
        input1 = torch.zeros(shape).npu()
        index = index.npu()
        if (isTensor):
            src = src.npu()
        input1.scatter_(dim, index, src)
        input1 = input1.cpu()
        return input1.numpy()

    def test_scatter_shape_format(self):
        shape_format = [
            [0, [3, 5], [np.float32, 0, [2, 5]]],
            [0, [3, 5], [np.float32, 3, [2, 5]]],
            [1, [3, 5], [np.float16, 0, [2, 5]]],
            [-1, [3, 5], [np.float16, 0, [2, 5]]],
        ]
        index = torch.tensor([[0, 1, 2, 0, 0], [2, 0, 0, 1, 2]])
        for item in shape_format:
            cpu_input, npu_input = create_common_tensor(item[2], 1, 100)

            if cpu_input.dtype == torch.float16:
                cpu_input = cpu_input.to(torch.float32)
            if npu_input.dtype == torch.float16:
                npu_input = npu_input.to(torch.float32)
            cpu_output = self.cpu_op_exec(item[1], item[0], index, cpu_input)
            npu_output = self.npu_op_exec(item[1], item[0], index, npu_input)

            if npu_output.dtype == np.float16:
                cpu_output = cpu_output.astype(np.float16)
            self.assertRtolEqual(cpu_output, npu_output)

            cpu_output = self.cpu_op_exec(item[1], item[0], index, 1.23)
            npu_output = self.npu_op_exec(item[1], item[0], index, 1.23, False)
            self.assertRtolEqual(cpu_output, npu_output)

            cpu_output = self.cpu_op_exec_inplace(item[1], item[0], index, cpu_input)
            npu_output = self.npu_op_exec_inplace(item[1], item[0], index, npu_input)

            if npu_output.dtype == np.float16:
                cpu_output = cpu_output.astype(np.float16)
            self.assertRtolEqual(cpu_output, npu_output)

            cpu_output = self.cpu_op_exec_inplace(item[1], item[0], index, 1.23)
            npu_output = self.npu_op_exec_inplace(item[1], item[0], index, 1.23, False)
            self.assertRtolEqual(cpu_output, npu_output)

    def test_scatter_debug(self):
        a = np.random.uniform(-2, 2, (31, 43, 41, 97)).astype(np.float16)
        b = np.random.uniform(0, 30, (31, 43, 41, 97)).astype(np.int32)
        c = np.random.uniform(-2, 2, (31, 43, 41, 97)).astype(np.float16)
        ca = torch.from_numpy(a)
        cb = torch.from_numpy(b).long()
        cc = torch.from_numpy(c)
        na = ca.npu()
        nb = cb.npu()
        nc = cc.npu()
        dim = 0
        cpu_output = torch.scatter(ca, dim, cb, cc)
        npu_output = torch.scatter(na, dim, nb, nc)
        self.assertRtolEqual(cpu_output, npu_output.cpu())

    def test_scatter_value(self):
        a = np.random.uniform(-2, 2, (31, 43, 41, 97)).astype(np.float16)
        b = np.random.uniform(0, 30, (31, 43, 41, 97)).astype(np.int32)
        ca = torch.from_numpy(a)
        cb = torch.from_numpy(b).long()
        na = ca.npu()
        nb = cb.npu()
        dim = 0
        cpu_output = torch.scatter(ca, dim, cb, 10)
        npu_output = torch.scatter(na, dim, nb, 10)
        self.assertRtolEqual(cpu_output, npu_output.cpu())


if __name__ == "__main__":
    run_tests()
