# Owner(s): ["module: dynamo"]
import sys

import torch
from torch._dynamo.test_case import TestCase

import torch_npu
import torch_npu.dynamo


class TestTorchairLazy(TestCase):
    def test_torchair_lazy(self):
        self.assertTrue('torchair' in sys.modules)
        self.assertTrue('torchair.npu_fx_compiler' not in sys.modules)
        import torchair
        torchair.get_npu_backend()
        self.assertTrue(isinstance(torchair, torch_npu.dynamo._LazyTorchair))
        self.assertTrue('torchair.npu_fx_compiler' in sys.modules)

if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
