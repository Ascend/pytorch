import torch
import torch.nn as nn

import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_distributed import skipIfUnsupportMultiNPU


class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(10, 5)

    def forward(self, x):
        return self.fc(x)


class ScriptModel(nn.Module):
    def __init__(self):
        super(ScriptModel, self).__init__()
        self.linear = torch.nn.Linear(4, 4)

    def forward(self, x, h):
        return torch.tanh(self.linear(x) + h)


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

    def test_Tensor_is_shared_runtimeerror(self):
        with self.assertRaisesRegex(RuntimeError, r"(.*) is not supported in npu."):
            input_tensor = torch.tensor([1, 2, 3]).npu()
            input_tensor.is_shared()

    def test_Tensor_share_memory__runtimeerror(self):
        with self.assertRaisesRegex(RuntimeError, r"(.*) is not supported in npu."):
            input_tensor = torch.tensor([1, 2, 3]).npu()
            input_tensor.share_memory_()

    def test_Module_share_memory_runtimeerror(self):
        with self.assertRaisesRegex(RuntimeError, r"(.*) is not supported in npu."):
            model = SimpleModel().npu()
            model.share_memory()

    def test_ScriptModule_register_parameter_runtimeerror(self):
        with self.assertRaisesRegex(RuntimeError, r"(.*) is not supported in npu."):
            model = ScriptModel().npu()
            x, h = torch.rand(3, 4).npu(), torch.rand(3, 4).npu()
            traced_cell = torch.jit.trace(model, (x, h))
            traced_cell.register_parameter("test_parameter", torch.nn.Parameter(torch.ones(1, 1)))

    def test_ScriptModule_add_module_runtimeerror(self):
        with self.assertRaisesRegex(RuntimeError, r"(.*) is not supported in npu."):
            model = ScriptModel().npu()
            x, h = torch.rand(3, 4).npu(), torch.rand(3, 4).npu()
            traced_cell = torch.jit.trace(model, (x, h))
            extra_linear = nn.Linear(5, 2)
            traced_cell.add_module("extra_linear", extra_linear)

    def test_ScriptModule_register_buffer_runtimeerror(self):
        with self.assertRaisesRegex(RuntimeError, r"(.*) is not supported in npu."):
            model = ScriptModel().npu()
            x, h = torch.rand(3, 4).npu(), torch.rand(3, 4).npu()
            traced_cell = torch.jit.trace(model, (x, h))
            traced_cell.register_buffer("test_buff", torch.zeros(3))

    def test_ScriptModule_register_module_runtimeerror(self):
        with self.assertRaisesRegex(RuntimeError, r"(.*) is not supported in npu."):
            model = ScriptModel().npu()
            x, h = torch.rand(3, 4).npu(), torch.rand(3, 4).npu()
            traced_cell = torch.jit.trace(model, (x, h))
            extra_linear = nn.Linear(5, 2)
            traced_cell.register_module("extra_linear", extra_linear)

    def test_ScriptModule_bfloat16_runtimeerror(self):
        with self.assertRaisesRegex(RuntimeError, r"(.*) is not supported in npu."):
            model = ScriptModel().npu()
            x, h = torch.rand(3, 4).npu(), torch.rand(3, 4).npu()
            traced_cell = torch.jit.trace(model, (x, h))
            traced_cell.bfloat16()

    def test_nested_tensor_runtimeerror(self):
        with self.assertRaisesRegex(RuntimeError, r"(.*) is not supported in npu."):
            a = torch.arange(3, dtype=torch.float).npu()
            b = torch.arange(3, dtype=torch.float).npu()
            torch.nested.nested_tensor([a, b])

    def test_as_nested_tensor_runtimeerror(self):
        with self.assertRaisesRegex(RuntimeError, r"(.*) is not supported in npu."):
            a = torch.arange(3, dtype=torch.float).npu()
            b = torch.arange(3, dtype=torch.float).npu()
            torch.nested.as_nested_tensor([a, b])

    def test_Tensor_is_shared(self):
        input_tensor = torch.tensor([1, 2, 3])
        input_tensor.is_shared()

    def test_Tensor_share_memory_(self):
        input_tensor = torch.tensor([1, 2, 3])
        input_tensor.share_memory_()

    def test_Module_share_memory(self):
        model = SimpleModel()
        model.share_memory()

    def test_nested_tensor(self):
        a = torch.arange(3, dtype=torch.float)
        b = torch.arange(3, dtype=torch.float)
        torch.nested.nested_tensor([a, b])

    def test_as_nested_tensor(self):
        a = torch.arange(3, dtype=torch.float)
        b = torch.arange(3, dtype=torch.float)
        torch.nested.as_nested_tensor([a, b])


if __name__ == "__main__":
    run_tests()
