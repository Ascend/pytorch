import torch
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests


class TestPtaUnsupportApi(TestCase):
    def test_sparse_coo_tensor(self):
        op_name = "aten::_sparse_coo_tensor_with_dims_and_tensors"
        with self.assertRaisesRegex(RuntimeError, "CAUTION: The operator '{}'".format(op_name)):
            indices = torch.tensor([[0, 1, 2],[1, 2, 0]]).npu()
            value = torch.tensor([3, 4, 5])
            shape = torch.Size([3, 3])
            torch.sparse_coo_tensor(indices, value, shape)

    def test_indices(self):
        op_name = "aten::empty.memory_format"
        with self.assertRaisesRegex(RuntimeError, "CAUTION: The operator '{}'".format(op_name)):
            indices = torch.tensor([[0, 1, 2],[1, 2, 0]])
            value = torch.tensor([3, 4, 5])
            shape = torch.Size([3, 3])
            sparse_tensor = torch.sparse_coo_tensor(indices, value, shape)
            coalesce_tensor = sparse_tensor.coalesce().npu()
            coalesce_tensor.indices()
    
    def test_crow_indices(self):
        op_name = "aten::empty.memory_format"
        with self.assertRaisesRegex(RuntimeError, "CAUTION: The operator '{}'".format(op_name)):
            indices = torch.tensor([[0, 1, 2],[1, 2, 0]])
            value = torch.tensor([3, 4, 5])
            shape = torch.Size([3, 3])
            sparse_tensor = torch.sparse_coo_tensor(indices, value, shape)
            coalesce_tensor = sparse_tensor.coalesce().npu()
            coalesce_tensor.crow_indices()

    def test_col_indices(self):
        op_name = "aten::empty.memory_format"
        with self.assertRaisesRegex(RuntimeError, "CAUTION: The operator '{}'".format(op_name)):
            indices = torch.tensor([[0, 1, 2],[1, 2, 0]])
            value = torch.tensor([3, 4, 5])
            shape = torch.Size([3, 3])
            sparse_tensor = torch.sparse_coo_tensor(indices, value, shape)
            coalesce_tensor = sparse_tensor.coalesce().npu()
            coalesce_tensor.col_indices()

    def test_row_indices(self):
        op_name = "aten::empty.memory_format"
        with self.assertRaisesRegex(RuntimeError, "CAUTION: The operator '{}'".format(op_name)):
            indices = torch.tensor([[0, 1, 2],[1, 2, 0]])
            value = torch.tensor([3, 4, 5])
            shape = torch.Size([3, 3])
            sparse_tensor = torch.sparse_coo_tensor(indices, value, shape)
            coalesce_tensor = sparse_tensor.coalesce().npu()
            coalesce_tensor.row_indices()

    def test_ccol_indices(self):
        op_name = "aten::empty.memory_format"
        with self.assertRaisesRegex(RuntimeError, "CAUTION: The operator '{}'".format(op_name)):
            indices = torch.tensor([[0, 1, 2],[1, 2, 0]])
            value = torch.tensor([3, 4, 5])
            shape = torch.Size([3, 3])
            sparse_tensor = torch.sparse_coo_tensor(indices, value, shape)
            coalesce_tensor = sparse_tensor.coalesce().npu()
            coalesce_tensor.ccol_indices()
        
    def test_to_sparse(self):
        op_name = "aten::empty.memory_format"
        with self.assertRaisesRegex(RuntimeError, "CAUTION: The operator '{}'".format(op_name)):
            dense_tensor = torch.randn(5, 5).npu()
            dense_tensor.to_sparse()

    def test_to_sparse_coo(self):
        op_name = "aten::empty.memory_format"
        with self.assertRaisesRegex(RuntimeError, "CAUTION: The operator '{}'".format(op_name)):
            dense_tensor = torch.randn(5, 5).npu()
            dense_tensor.to_sparse_coo()
if __name__ == "__main__":
    run_tests()
