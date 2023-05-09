import os
import torch
import torch.nn as nn

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.hooks import set_dump_path, seed_all, register_acc_cmp_hook
from torch_npu.hooks.tools import compare


class TestModuleOP(nn.Module):
    
    def __init__(self) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(in_features=2, out_features=2)
        self.linear_2 = nn.Linear(in_features=2, out_features=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x1 = self.linear_1(x)
        x2 = self.linear_2(x1)
        r1 = self.relu(x2)
        return r1


class TestAccCmpModuleHook(TestCase):

    def test_module_op(self):
        module = TestModuleOP()
        register_acc_cmp_hook(module)
        seed_all()
        set_dump_path("./cpu_module_tensor_op.pkl")
        x = torch.randn(2, 2)
        out = module(x)
        loss = out.sum()
        loss.backward()
        set_dump_path("./npu_module_tensor_op.pkl")
        module.npu()
        x = x.npu()
        out = module(x)
        loss = out.sum()
        loss.backward()
        assert os.path.exists("./cpu_module_tensor_op.pkl") and os.path.exists("./npu_module_tensor_op.pkl")
        compare("./npu_module_tensor_op.pkl", "./cpu_module_tensor_op.pkl", "./module_tensor_op_result.csv")
        assert os.path.exists("./module_tensor_op_result.csv")

    def tearDown(self) -> None:
        for filename in os.listdir('./'):
            if filename.endswith(".pkl") or filename.endswith(".csv"):
                os.remove("./" + filename)


if __name__ == '__main__':
    run_tests()
