import torch
from torch_npu.testing.testcase import TestCase, run_tests


class TestSparseCoo(TestCase):
    def _create_sparse_coo_tensor(self):
        indices = torch.tensor([[0, 1, 2], [0, 1, 2]])
        values = torch.tensor([1, 2, 4])
        sparse_coo_cpu = torch.sparse_coo_tensor(indices=indices, values=values, size=[4, 4])
        sparse_coo_npu = torch.sparse_coo_tensor(indices=indices.npu(), values=values.npu(), size=[4, 4])
        return sparse_coo_npu, sparse_coo_cpu

    def test_sparse_coo_nnz(self):
        sparse_coo_npu, sparse_coo_cpu = self._create_sparse_coo_tensor()
        self.assertRtolEqual(sparse_coo_npu._nnz(), sparse_coo_cpu._nnz())

    def test_sparse_coo_indices_and_values(self):
        sparse_coo_npu, sparse_coo_cpu = self._create_sparse_coo_tensor()
        self.assertRtolEqual(sparse_coo_npu._indices(), sparse_coo_cpu._indices())
        self.assertRtolEqual(sparse_coo_npu._values(), sparse_coo_cpu._values())

    def test_sparse_coo_dim(self):
        sparse_coo_npu, sparse_coo_cpu = self._create_sparse_coo_tensor()
        self.assertRtolEqual(sparse_coo_npu.sparse_dim(), sparse_coo_cpu.sparse_dim())
        self.assertRtolEqual(sparse_coo_npu.dense_dim(), sparse_coo_cpu.dense_dim())

    def test_sparse_coo_is_coalesced(self):
        sparse_coo_npu, _ = self._create_sparse_coo_tensor()
        self.assertEqual(sparse_coo_npu.is_coalesced(), False)
        sparse_coo_npu._coalesced_(True)
        self.assertEqual(sparse_coo_npu.is_coalesced(), True)

    def test_sparce_coo_neg(self):
        sparse_coo_npu, sparse_coo_cpu = self._create_sparse_coo_tensor()
        sparse_coo_cpu_neg = sparse_coo_cpu.neg()
        sparse_coo_npu_neg = sparse_coo_npu.neg()
        self.assertRtolEqual(sparse_coo_cpu_neg._values(), sparse_coo_npu_neg._values())


if __name__ == "__main__":
    run_tests()
