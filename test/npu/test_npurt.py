#!/usr/bin/env python3
import ctypes

import torch
import torch_npu
from torch.testing._internal.common_utils import TestCase, run_tests


ACL_HOST_REG_MAPPED = 0x2
ACL_HOST_REG_PINNED = 0x10000000


class TestNPURT(TestCase):
    def _check_host_register_unregister(self, flag):
        rt = torch_npu.npu.npurt()

        tensor = torch.empty(4096, dtype=torch.uint8).share_memory_()
        ptr = tensor.data_ptr()
        size = tensor.numel() * tensor.element_size()

        ret = rt.npuHostRegister(ptr, size, flag)
        self.assertEqual(int(ret), 0)

        try:
            self.assertGreater(ptr, 0)
            self.assertGreater(size, 0)
            self.assertEqual(ptr % 4096, 0)
            self.assertEqual(size % 4096, 0)
        finally:
            ret = rt.npuHostUnregister(ptr)
            self.assertEqual(int(ret), 0)

    def test_npurt_api_exists(self):
        rt = torch_npu.npu.npurt()
        self.assertIs(rt, torch_npu.npu.npurt())
        self.assertTrue(hasattr(torch_npu._C, "_npurt"))

        for name in (
            "npuHostRegister",
            "npuHostUnregister",
            "npuStreamCreate",
            "npuStreamDestroy",
        ):
            self.assertTrue(hasattr(rt, name))

    def test_npurt_stream_create_destroy(self):
        rt = torch_npu.npu.npurt()

        stream = ctypes.c_void_p()
        stream_p_int = ctypes.addressof(stream)

        ret = rt.npuStreamCreate(stream_p_int)
        self.assertEqual(int(ret), 0)

        try:
            self.assertIsNotNone(stream.value)
            self.assertNotEqual(stream.value, 0)
        finally:
            if stream.value:
                ret = rt.npuStreamDestroy(stream.value)
                self.assertEqual(int(ret), 0)

    def test_npurt_host_register_supported_flags(self):
        # Host memory satisfies the ACL registration requirements (shared memory /
        # page-aligned allocation). In this scenario ACL_HOST_REG_MAPPED and
        # ACL_HOST_REG_PINNED (or their combination) are all expected to succeed.
        for flag in (
            ACL_HOST_REG_PINNED,
            ACL_HOST_REG_MAPPED,
            ACL_HOST_REG_PINNED | ACL_HOST_REG_MAPPED,
        ):
            self._check_host_register_unregister(flag)

    def test_npurt_register_only_pinned(self):
        # Ordinary CPU tensor allocated by PyTorch (non page-aligned / non-shared
        # memory). Only ACL_HOST_REG_PINNED is expected to work, matching the
        # current PyTorch pin_memory registration path.
        ACL_HOST_REG_PINNED = 0x10000000
        t = torch.ones(20)
        npurt = torch_npu.npu.npurt()
        r = npurt.npuHostRegister(t.data_ptr(), t.numel() * t.element_size(), ACL_HOST_REG_PINNED)
        self.assertEqual(r, 0)
        try:
            self.assertGreater(t.data_ptr(), 0)
        finally:
            r = npurt.npuHostUnregister(t.data_ptr())
            self.assertEqual(r, 0)

    def test_npurt_host_register_invalid_size(self):
        rt = torch_npu.npu.npurt()

        tensor = torch.empty(4096, dtype=torch.uint8).share_memory_()
        ret = rt.npuHostRegister(tensor.data_ptr(), 0, ACL_HOST_REG_PINNED)
        self.assertNotEqual(int(ret), 0)

    def test_npurt_null_ptr_returns_error(self):
        rt = torch_npu.npu.npurt()

        self.assertNotEqual(int(rt.npuHostRegister(0, 4096, ACL_HOST_REG_PINNED)), 0)
        self.assertNotEqual(int(rt.npuHostUnregister(0)), 0)
        self.assertNotEqual(int(rt.npuStreamCreate(0)), 0)
        self.assertNotEqual(int(rt.npuStreamDestroy(0)), 0)


if __name__ == "__main__":
    run_tests()
