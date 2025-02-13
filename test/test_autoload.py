# Owner(s): ["module: unknown"]

import sys
import unittest

import torch
import torch.nn as nn
from torch.testing._internal.common_utils import (TestCase, run_tests)

AUTO_LOAD = hasattr(torch, "_is_device_backend_autoload_enabled") and torch._is_device_backend_autoload_enabled()
RUN_NPU = AUTO_LOAD and torch.npu.is_available()


class AutoloadTest(TestCase):

    # torch_npu should be imported implicitly after running 'import torch'
    @unittest.skipIf(not RUN_NPU, "requires npu")
    def test_autoload(self):
        self.assertTrue("torch_npu" in sys.modules)

    @unittest.skipIf(not RUN_NPU, "requires npu")
    def test_autoload_tensor(self):
        ones_npu = torch.ones(5, 5, device="npu")
        self.assertEqual(ones_npu.device.type, "npu")

    @unittest.skipIf(not RUN_NPU, "requires npu")
    def test_autoload_model(self):
        class Model(nn.Module):
            def __init__(self, input_size, num_classes):
                super(Model, self).__init__()
                self.fc = nn.Linear(input_size, num_classes)

            def forward(self, x):
                out = self.fc(x)
                return out

        model = Model(10, 2)
        model = model.to("npu")

        x = torch.randn(64, 10, device="npu")
        outputs = model(x)
        self.assertEqual(outputs.device.type, "npu")
        self.assertTupleEqual(outputs.shape, (64, 2))


if __name__ == "__main__":
    run_tests()
