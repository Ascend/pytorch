import torch
from torch_npu.testing.common_utils import SupportedDevices
from torch_npu.testing.testcase import TestCase, run_tests


class TestSparseCsr(TestCase):
    def _create_sparse_csr_tensor(self):
        crow_indices = torch.tensor([0, 2, 4])
        col_indices = torch.tensor([0, 1, 0, 1])
        values = torch.tensor([1, 2, 3, 4])
        sparse_csr_cpu = torch.sparse_csr_tensor(crow_indices, col_indices, values, dtype=torch.float64)
        sparse_csr_npu = torch.sparse_csr_tensor(crow_indices.npu(), col_indices.npu(), values.npu(),
                                                 dtype=torch.float64)
        return sparse_csr_npu, sparse_csr_cpu

    def test_sparse_csr_indices_and_values(self):
        sparse_csr_npu, sparse_csr_cpu = self._create_sparse_csr_tensor()
        self.assertRtolEqual(sparse_csr_npu.crow_indices(), sparse_csr_cpu.crow_indices())
        self.assertRtolEqual(sparse_csr_npu.col_indices(), sparse_csr_cpu.col_indices())
        self.assertRtolEqual(sparse_csr_npu.values(), sparse_csr_cpu.values())

    def test_sparse_csr_nnz(self):
        sparse_csr_npu, sparse_csr_cpu = self._create_sparse_csr_tensor()
        self.assertRtolEqual(sparse_csr_npu._nnz(), sparse_csr_cpu._nnz())

    def test_sparse_csr_to_dense(self):
        sparse_csr_npu, sparse_csr_cpu = self._create_sparse_csr_tensor()
        self.assertRtolEqual(sparse_csr_npu.to_dense(), sparse_csr_cpu.to_dense())


if __name__ == "__main__":
    run_tests()
