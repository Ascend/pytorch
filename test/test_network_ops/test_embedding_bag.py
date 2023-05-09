import copy
import torch
import numpy as np
import torch.nn.functional as F

import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests


class TestEmbeddingBag(TestCase):

    def test_embedding_bag_1d(self):
        cpu_weight = torch.rand(10, 3)
        cpu_indices = torch.tensor([1, 2, 4, 5, 4, 3, 2, 9])
        cpu_offsets = torch.tensor([0, 4])
        npu_weight = cpu_weight.npu()
        npu_indices = cpu_indices.npu()
        npu_offsets = cpu_offsets.npu()
        cpu_output = F.embedding_bag(cpu_weight, cpu_indices, cpu_offsets).detach().numpy()
        npu_output = F.embedding_bag(npu_weight, npu_indices, npu_offsets).cpu().detach().numpy()
        self.assertRtolEqual(cpu_output, npu_output)

    def test_embedding_bag_2d(self):
        cpu_weight = torch.rand(10, 3)
        cpu_indices = torch.tensor([[1, 2, 4, 5, 4, 3, 2, 9], [1, 2, 4, 5, 4, 3, 2, 9]])
        npu_weight = cpu_weight.npu()
        npu_indices = cpu_indices.npu()
        cpu_output = F.embedding_bag(cpu_weight, cpu_indices).detach().numpy()
        npu_output = F.embedding_bag(npu_weight, npu_indices).cpu().detach().numpy()
        self.assertRtolEqual(cpu_output, npu_output)

    def test_embedding_bag_0d(self):
        embedding_bag_cpu = torch.nn.EmbeddingBag(5, 2, mode='sum')
        embedding_bag_npu = copy.deepcopy(embedding_bag_cpu).npu()
        cpu_input = torch.tensor([]).int()
        cpu_offset = torch.zeros(5).int()
        cpu_weight = torch.tensor([]).float()
        npu_input = cpu_input.npu()
        npu_offset = cpu_offset.npu()
        npu_weight = cpu_weight.npu()
        cpu_output = embedding_bag_cpu(cpu_input, cpu_offset, cpu_weight).detach().numpy()
        npu_output = embedding_bag_npu(npu_input, npu_offset, npu_weight).cpu().detach().numpy()
        self.assertRtolEqual(cpu_output, npu_output)


if __name__ == "__main__":
    run_tests()
