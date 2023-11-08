import torch
import torch.nn as nn

import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests


class TestEmbeddingBagBackward(TestCase):
    def cpu_op_exec(self, input1, embedding_matrix, offsets):
        embedding_matrix.requires_grad_()
        m = nn.functional.embedding_bag(input1, embedding_matrix, offsets)
        grads = torch.ones_like(m)
        m.backward(grads)
        output = embedding_matrix.grad.numpy()

        return output

    def npu_op_exec(self, input1, embedding_matrix, offsets):
        embedding_matrix.requires_grad_()
        m = nn.functional.embedding_bag(input1, embedding_matrix, offsets).npu()
        grads = torch.ones_like(m)
        m.backward(grads)
        output = embedding_matrix.grad.cpu().numpy()

        return output

    def test_embedding_bag_backward_shape_format(self):
        test_cases = [
            [torch.tensor([1, 2, 4, 5, 4, 3, 2, 9]).to(torch.int32), torch.tensor([0, 4]).to(torch.int32), 
             torch.randn(10, 3), torch.tensor([1, 2, 4, 5, 4, 3, 2, 9]).to(torch.int32).to("npu"),
             torch.tensor([0, 4]).to(torch.int32).to("npu")],
            [torch.tensor([1, 2, 4, 5, 4, 3, 2, 9, 6]).to(torch.int32), torch.tensor([0, 4]).to(torch.int32),
             torch.randn(10, 3), torch.tensor([1, 2, 4, 5, 4, 3, 2, 9, 6]).to(torch.int32).to("npu"),
             torch.tensor([0, 4]).to(torch.int32).to("npu")],
            [torch.tensor([0, 2, 0, 5]).to(torch.int32), torch.tensor([0, 2]).to(torch.int32),
             torch.randn(10, 3), torch.tensor([0, 2, 0, 5]).to(torch.int32).to("npu"),
             torch.tensor([0, 2]).to(torch.int32).to("npu")],
            [torch.tensor([0, 1, 2, 3]).to(torch.int32), torch.tensor([0, 3]).to(torch.int32),
             torch.randn(4, 3), torch.tensor([0, 1, 2, 3]).to(torch.int32).to("npu"),
             torch.tensor([0, 3]).to(torch.int32).to("npu")],
            [torch.tensor([1, 2, 4, 5, 4, 3, 2, 9]).to(torch.int32), torch.tensor([0, 4]).to(torch.int32),
             torch.randn(10, 4), torch.tensor([1, 2, 4, 5, 4, 3, 2, 9]).to(torch.int32).to("npu"),
             torch.tensor([0, 4]).to(torch.int32).to("npu")],
            [torch.tensor([0, 1, 2, 3]).to(torch.int32), torch.tensor([0, 3]).to(torch.int32),
             torch.randn(10, 4), torch.tensor([0, 1, 2, 3]).to(torch.int32).to("npu"),
             torch.tensor([0, 3]).to(torch.int32).to("npu")]
        ]
        for item in test_cases:
            cpu_input, npu_input = item[0], item[3]
            cpu_offsets, npu_offsets = item[1], item[4]
            cpu_embedding_matrix, npu_embedding_matrix = item[2], item[2].npu()

            cpu_output = self.cpu_op_exec(cpu_input, cpu_embedding_matrix, cpu_offsets)
            npu_output = self.npu_op_exec(npu_input, npu_embedding_matrix, npu_offsets)
            self.assertEqual(cpu_output, npu_output)


if __name__ == "__main__":
    run_tests()
