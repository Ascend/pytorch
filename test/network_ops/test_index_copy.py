import torch
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor


class TestIndexCopy(TestCase):
    def op_exec(self, npuflag, input1, dim, indices, updates):
        output = torch.index_copy(input1, dim, indices, updates)
        if npuflag:
            output = output.to("cpu")
        output = output.numpy()
        return output

    def op_inp_exec(self, npuflag, input1, dim, indices, updates):
        input1.index_copy_(dim, indices, updates)
        if npuflag:
            input1 = input1.to("cpu")
        output = input1.numpy()
        return output

    def case_exec(self, input1, dim, indices, updates):
        npu_input = input1.npu()
        npu_indices = indices.npu()
        npu_updates = updates.npu()
        cpu_output = self.op_exec(0, input1, dim, indices, updates)
        npu_output = self.op_exec(1, npu_input, dim, npu_indices, npu_updates)
        self.assertEqual(cpu_output, npu_output)
        cpu_output = self.op_inp_exec(0, input1, dim, indices, updates)
        npu_output = self.op_inp_exec(1, npu_input, dim, npu_indices, npu_updates)
        self.assertEqual(cpu_output, npu_output)

    def test_index_copy_dim0_0(self):
        a = torch.ones(5, dtype=torch.float32)
        indices = torch.LongTensor([3, 2, 1, 0])
        updates = torch.tensor([1, 2, 3, 4], dtype=torch.float32)
        self.case_exec(a, 0, indices, updates)

    def test_index_copy_dim0_1(self):
        a = torch.ones(5, 3, dtype=torch.float32)
        indices = torch.LongTensor([0, 1, 2])
        updates = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=torch.float32)
        self.case_exec(a, 0, indices, updates)

    def test_index_copy_dim0_2(self):
        a = torch.ones(2, 5, 3, dtype=torch.float32)
        indices = torch.LongTensor([0])
        updates = torch.tensor([[[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12], [13, 14, 15]]], dtype=torch.float32)
        self.case_exec(a, 0, indices, updates)

    def test_index_copy_dim1_0(self):
        a = torch.ones(5, 3, dtype=torch.float32)
        indices = torch.LongTensor([0, 1])
        updates = torch.tensor([[1, 2], [5, 6], [8, 9], [3, 4], [0, 1]], dtype=torch.float32)
        self.case_exec(a, 1, indices, updates)

    def test_index_copy_dim1_1(self):
        a = torch.ones(2, 5, 3, dtype=torch.float32)
        indices = torch.LongTensor([0])
        updates = torch.tensor([[[1, 2, 3]], [[4, 5, 6]]], dtype=torch.float32)
        self.case_exec(a, 1, indices, updates)

    def test_index_copy_dim2_0(self):
        a = torch.ones(2, 5, 3, dtype=torch.float32)
        indices = torch.LongTensor([0])
        updates = torch.tensor([[[1], [2], [3], [4], [5]],
                                [[6], [7], [8], [9], [0]]], dtype=torch.float32)
        self.case_exec(a, 2, indices, updates)

    def test_index_copy_dim2_1(self):
        a = torch.ones(2, 5, 3, dtype=torch.float32)
        indices = torch.LongTensor([0, 1])
        updates = torch.tensor([[[3, 2], [1, 2], [1, 3], [1, 4], [1, 5]],
                                [[1, 6], [1, 7], [1, 8], [1, 9], [1, 0]]], dtype=torch.float32)
        self.case_exec(a, 2, indices, updates)


if __name__ == "__main__":
    run_tests()
