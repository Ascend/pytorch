import torch
import torch.nn as nn


from torch_npu.testing.testcase import TestCase, run_tests


class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(10, 5)

    def forward(self, x):
        return self.fc(x)


class TestPtaUnsupportApi(TestCase):

    def test_crow_indices(self):
        op_name = "crow_indices"
        with self.assertRaisesRegex(RuntimeError, "{} expected ".format(op_name)):
            indices = torch.tensor([[0, 1, 2], [1, 2, 0]])
            value = torch.tensor([3, 4, 5])
            shape = torch.Size([3, 3])
            sparse_tensor = torch.sparse_coo_tensor(indices, value, shape)
            coalesce_tensor = sparse_tensor.coalesce().npu()
            coalesce_tensor.crow_indices()

    def test_col_indices(self):
        op_name = "col_indices"
        with self.assertRaisesRegex(RuntimeError, "{} expected ".format(op_name)):
            indices = torch.tensor([[0, 1, 2], [1, 2, 0]])
            value = torch.tensor([3, 4, 5])
            shape = torch.Size([3, 3])
            sparse_tensor = torch.sparse_coo_tensor(indices, value, shape)
            coalesce_tensor = sparse_tensor.coalesce().npu()
            coalesce_tensor.col_indices()

    def test_row_indices(self):
        op_name = "row_indices"
        with self.assertRaisesRegex(RuntimeError, "{} expected ".format(op_name)):
            indices = torch.tensor([[0, 1, 2], [1, 2, 0]])
            value = torch.tensor([3, 4, 5])
            shape = torch.Size([3, 3])
            sparse_tensor = torch.sparse_coo_tensor(indices, value, shape)
            coalesce_tensor = sparse_tensor.coalesce().npu()
            coalesce_tensor.row_indices()

    def test_ccol_indices(self):
        op_name = "ccol_indices"
        with self.assertRaisesRegex(RuntimeError, "{} expected ".format(op_name)):
            indices = torch.tensor([[0, 1, 2], [1, 2, 0]])
            value = torch.tensor([3, 4, 5])
            shape = torch.Size([3, 3])
            sparse_tensor = torch.sparse_coo_tensor(indices, value, shape)
            coalesce_tensor = sparse_tensor.coalesce().npu()
            coalesce_tensor.ccol_indices()

    def test_Module_share_memory_npu(self):
        model = SimpleModel().npu()
        model.share_memory()
        self.assertEqual(model.fc.weight.device.type, "npu")

    def test_Tensor_is_shared(self):
        input_tensor = torch.tensor([1, 2, 3])
        input_tensor.is_shared()

    def test_Tensor_share_memory_(self):
        input_tensor = torch.tensor([1, 2, 3])
        input_tensor.share_memory_()

    def test_Module_share_memory(self):
        model = SimpleModel()
        model.share_memory()


if __name__ == "__main__":
    run_tests()
