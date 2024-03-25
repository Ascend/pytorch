import torch
from torch_npu.testing.common_utils import SupportedDevices
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

    def test_sparse_coo_neg(self):
        sparse_coo_npu, sparse_coo_cpu = self._create_sparse_coo_tensor()
        sparse_coo_cpu_neg = sparse_coo_cpu.neg()
        sparse_coo_npu_neg = sparse_coo_npu.neg()
        self.assertRtolEqual(sparse_coo_cpu_neg._values(), sparse_coo_npu_neg._values())

    def test_sparse_coo_copy_(self):
        sparse_coo_npu, _ = self._create_sparse_coo_tensor()
        copy_sparse_tensor = torch.rand(sparse_coo_npu.shape).to_sparse().npu().to(sparse_coo_npu.dtype)
        copy_sparse_tensor.copy_(sparse_coo_npu)
        self.assertRtolEqual(copy_sparse_tensor._values(), sparse_coo_npu._values())
        self.assertRtolEqual(copy_sparse_tensor._indices(), sparse_coo_npu._indices())

    def test_sparse_coo_dense(self):
        sparse_coo_npu, sparse_coo_cpu = self._create_sparse_coo_tensor()
        self.assertRtolEqual(sparse_coo_npu.to_dense(), sparse_coo_cpu.to_dense())

    @SupportedDevices(['Ascend910B'])
    def test_sparse_coalesce(self):
        dtypes = [torch.float, torch.float16, torch.int32]
        for dtype in dtypes:
            i = torch.Tensor([[1, 2, 1, 2], [2, 2, 2, 3]])
            v = torch.Tensor([1, 2, 3, 4]).to(dtype)
            sparse_coo_cpu = torch.sparse_coo_tensor(indices=i, values=v, size=[4, 4]).coalesce()
            sparse_coo_npu = torch.sparse_coo_tensor(indices=i.npu(), values=v.npu(), size=[4, 4]).coalesce()
            self.assertRtolEqual(sparse_coo_npu._values(), sparse_coo_cpu._values())
            self.assertRtolEqual(sparse_coo_npu._indices(), sparse_coo_cpu._indices())

            v2 = torch.Tensor([[1, 2], [2, 3], [3, 3], [4, 9]]).to(dtype)
            sparse_coo_cpu2 = torch.sparse_coo_tensor(indices=i, values=v2, size=[4, 4, 2]).coalesce()
            sparse_coo_npu2 = torch.sparse_coo_tensor(indices=i.npu(), values=v2.npu(), size=[4, 4, 2]).coalesce()
            self.assertRtolEqual(sparse_coo_npu2._values(), sparse_coo_cpu2._values())
            self.assertRtolEqual(sparse_coo_npu2._indices(), sparse_coo_cpu2._indices())

            i2 = torch.Tensor([[1, 2, 1, 2], [2, 2, 2, 3], [2, 2, 2, 3]])
            sparse_coo_cpu3 = torch.sparse_coo_tensor(indices=i2, values=v, size=[4, 4, 4]).coalesce()
            sparse_coo_npu3 = torch.sparse_coo_tensor(indices=i2.npu(), values=v.npu(), size=[4, 4, 4]).coalesce()
            self.assertRtolEqual(sparse_coo_npu3._values(), sparse_coo_cpu3._values())
            self.assertRtolEqual(sparse_coo_npu3._indices(), sparse_coo_cpu3._indices())

    @SupportedDevices(['Ascend910B'])
    def test_sparse_add(self):
        sparse_coo_npu, sparse_coo_cpu = self._create_sparse_coo_tensor()
        input1 = torch.rand(4, 4)

        with self.assertRaises(RuntimeError):
            sparse_coo_npu + input1.npu()

        output_cpu1 = input1 + sparse_coo_cpu
        output_npu1 = input1.npu() + sparse_coo_npu
        self.assertRtolEqual(output_cpu1, output_npu1)

        output_cpu2 = input1.to_sparse() + sparse_coo_cpu
        output_npu2 = input1.to_sparse().npu() + sparse_coo_npu
        self.assertRtolEqual(output_npu2._values(), output_cpu2._values())
        self.assertRtolEqual(output_npu2._indices(), output_cpu2._indices())

    @SupportedDevices(['Ascend910B'])
    def test_sparse_sum(self):
        sparse_coo_npu, sparse_coo_cpu = self._create_sparse_coo_tensor()
        self.assertRtolEqual(sparse_coo_npu.sum(), sparse_coo_cpu.sum())

    def test_sparse_max(self):
        sparse_coo_npu, sparse_coo_cpu = self._create_sparse_coo_tensor()
        self.assertRtolEqual(sparse_coo_npu.max(), sparse_coo_cpu.to_dense().max())


if __name__ == "__main__":
    run_tests()
