import os

import torch

from torch.testing._internal.common_utils import TestCase, run_tests
import torch_npu


class TestModule(torch.nn.Module):
    def forward(self, a, b, c):
        b = torch.transpose(b, 0, 1)
        add = a + b
        sub = c - a
        mul = add * sub
        return mul + 3


class TestDvmAccuracy(TestCase):
    def test_dvm_accuracy_check(self):
        os.environ["TORCHINDUCTOR_NPU_BACKEND"] = "dvm"
        os.environ["INDUCTOR_ASCEND_CHECK_ACCURACY"] = "1"

        a = torch.normal(0, 0.01, size=(512, 1), dtype=torch.float16).npu()
        b = torch.normal(0, 0.01, size=(512, 4, 256), dtype=torch.float16).npu()
        c = torch.normal(0, 0.01, size=(1, 256), dtype=torch.float16).npu()
        model = TestModule()
        dvm_compiled_model = torch.compile(
            model, backend="inductor", dynamic=False
        )

        with torch.no_grad():
            expect = model(a, b, c)
            result = dvm_compiled_model(a, b, c)
            self.assertEqual(expect, result, atol=1e-3, rtol=1e-3)


if __name__ == "__main__":
    run_tests()
