# Owner(s): ["module: npu"]

# NOTE: this needs to be run in a brand new process

import os
import unittest

import torch
import torch_npu
from torch.testing._internal.common_distributed import MultiProcessTestCase
from torch.testing._internal.common_utils import run_tests

TEST_MULTINPU = torch_npu.npu.device_count() > 1

@unittest.skipIf(not TEST_MULTINPU, "only one NPU detected")
class TestNpuPrimaryCtx(MultiProcessTestCase):
    CTX_ALREADY_CREATED_ERR_MSG = (
        "Tests defined in test_npu_primary_ctx.py must be run in a process "
        "where NPU contexts are never created."
    )

    @property
    def world_size(self):
        return 1

    def setUp(self):
        super().setUp()
        self._spawn_processes()

    def _check_no_ctx(self):
        for device in range(torch_npu.npu.device_count()):
            self.assertFalse(
                torch_npu._C._npu_hasPrimaryContext(device),
                TestNpuPrimaryCtx.CTX_ALREADY_CREATED_ERR_MSG,
            )

    def test_str_repr(self):
        self._check_no_ctx()
        x = torch.randn(1, device="npu:1")

        # We should have only created context on 'npu:1'
        self.assertFalse(torch_npu._C._npu_hasPrimaryContext(0))
        self.assertTrue(torch_npu._C._npu_hasPrimaryContext(1))

        str(x)
        repr(x)

        # We should still have only created context on 'npu:1'
        self.assertFalse(torch_npu._C._npu_hasPrimaryContext(0))
        self.assertTrue(torch_npu._C._npu_hasPrimaryContext(1))

    def test_copy(self):
        self._check_no_ctx()
        x = torch.randn(1, device="npu:1")

        # We should have only created context on 'npu:1'
        self.assertFalse(torch_npu._C._npu_hasPrimaryContext(0))
        self.assertTrue(torch_npu._C._npu_hasPrimaryContext(1))

        y = torch.randn(1, device="cpu")
        y.copy_(x)

        # We should still have only created context on 'npu:1'
        self.assertFalse(torch_npu._C._npu_hasPrimaryContext(0))
        self.assertTrue(torch_npu._C._npu_hasPrimaryContext(1))

    def test_pin_memory(self):
        self._check_no_ctx()
        x = torch.randn(1, device="npu:1")

        # We should have only created context on 'npu:1'
        self.assertFalse(torch_npu._C._npu_hasPrimaryContext(0))
        self.assertTrue(torch_npu._C._npu_hasPrimaryContext(1))

        self.assertFalse(x.is_pinned())

        # We should still have only created context on 'npu:1'
        self.assertFalse(torch_npu._C._npu_hasPrimaryContext(0))
        self.assertTrue(torch_npu._C._npu_hasPrimaryContext(1))

        x = torch.randn(3, device="cpu").pin_memory()

        # We should still have only created context on 'npu:1'
        self.assertFalse(torch_npu._C._npu_hasPrimaryContext(0))
        self.assertTrue(torch_npu._C._npu_hasPrimaryContext(1))

        self.assertTrue(x.is_pinned())

        # We should still have only created context on 'npu:1'
        self.assertFalse(torch_npu._C._npu_hasPrimaryContext(0))
        self.assertTrue(torch_npu._C._npu_hasPrimaryContext(1))

        x = torch.randn(3, device="cpu", pin_memory=True)

        # We should still have only created context on 'npu:1'
        self.assertFalse(torch_npu._C._npu_hasPrimaryContext(0))
        self.assertTrue(torch_npu._C._npu_hasPrimaryContext(1))

        x = torch.zeros(3, device="cpu", pin_memory=True)

        # We should still have only created context on 'npu:1'
        self.assertFalse(torch_npu._C._npu_hasPrimaryContext(0))
        self.assertTrue(torch_npu._C._npu_hasPrimaryContext(1))

        x = torch.empty(3, device="cpu", pin_memory=True)

        # We should still have only created context on 'npu:1'
        self.assertFalse(torch_npu._C._npu_hasPrimaryContext(0))
        self.assertTrue(torch_npu._C._npu_hasPrimaryContext(1))

        x = x.pin_memory()

        # We should still have only created context on 'npu:1'
        self.assertFalse(torch_npu._C._npu_hasPrimaryContext(0))
        self.assertTrue(torch_npu._C._npu_hasPrimaryContext(1))


if __name__ == "__main__":
    os.environ["ACL_OP_INIT_MODE"] = "1"
    run_tests()
