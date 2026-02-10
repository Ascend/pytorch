import unittest

import os
import torch
from torch.testing._internal.common_utils import TestCase, run_tests
from torch_npu.npu.utils import get_cann_version


class TestPinMemoryHostRegister(TestCase):

    @unittest.skipUnless(get_cann_version("RUNTIME") >= "8.5.0" and get_cann_version(module="DRIVER") >= "25.5.0", "This feature is not supported in older versions.")
    def test_pin_memory_host_register(self):
        os.environ["PYTORCH_NPU_ALLOC_CONF"] = "pinned_mem_register:True"
        cpu_tensor = torch.ones([2, 3])
        pin_tensor = cpu_tensor.pin_memory()
        self.assertTrue(pin_tensor.is_pinned())

if __name__ == '__main__':
    run_tests()
