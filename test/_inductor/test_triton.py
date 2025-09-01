# Owner(s): ["module: tests"]

import unittest
import torch
from torch.testing._internal.common_utils import run_tests, TestCase, load_tests
from torch.utils._triton import has_triton_package, has_triton, has_triton_tma, has_triton_tma_device
import torch_npu
import torch_npu.testing

# load_tests from torch.testing._internal.common_utils is used to automatically filter tests for
# sharding on sandcastle. This line silences flake warnings
load_tests = load_tests


class TestHasTriton(TestCase):

    def test_has_triton(self):
        if not has_triton_package():
            # no triton library found, skip test_has_triton
            return

        self.assertFalse(has_triton())
        self.assertFalse(has_triton_tma())
        self.assertFalse(has_triton_tma_device())

        from torch_npu.contrib import transfer_to_npu

        self.assertFalse(has_triton())
        self.assertFalse(has_triton_tma())
        self.assertFalse(has_triton_tma_device())




if __name__ == "__main__":
    run_tests()