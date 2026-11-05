import ctypes
import gc
import os
import unittest

import torch
import torch_npu


class TestAllocatorLastError(unittest.TestCase):
    def setUp(self):
        torch_npu.npu.set_device(int(os.environ.get("SET_NPU_DEVICE", "0")))
        self.acl = ctypes.CDLL("libascendcl.so")
        for name in ("aclrtGetLastError", "aclrtPeekAtLastError"):
            getattr(self.acl, name).argtypes = [ctypes.c_int]
            getattr(self.acl, name).restype = ctypes.c_int
        gc.collect()
        torch.npu.synchronize()
        torch.npu.empty_cache()
        self.assertEqual(self.acl.aclrtGetLastError(0), 0)
        self.free, _ = torch.npu.mem_get_info()
        self.before = torch.npu.memory_stats()["num_alloc_retries"]

    def tearDown(self):
        gc.collect()
        torch.npu.synchronize()
        torch.npu.empty_cache()
        torch.npu.memory._set_allocator_settings("expandable_segments:False")

    def _assert_retry_clears_error(self):
        gc.collect()
        torch.npu.synchronize()
        self.assertEqual(self.acl.aclrtPeekAtLastError(0), 0)

        out = torch.empty(int(self.free * 0.60), dtype=torch.uint8, device="npu")
        retries = torch.npu.memory_stats()["num_alloc_retries"] - self.before
        last_error = self.acl.aclrtGetLastError(0)
        del out
        self.assertGreater(retries, 0, "Allocation did not exercise retry")
        self.assertEqual(last_error, 0, f"Successful retry left last_error={last_error}")

    def test_normal_allocator_retry(self):
        torch.npu.memory._set_allocator_settings("expandable_segments:False")
        first = torch.empty(int(self.free * 0.40), dtype=torch.uint8, device="npu")
        second = torch.empty(int(self.free * 0.40), dtype=torch.uint8, device="npu")
        del first, second
        self._assert_retry_clears_error()

    def test_expandable_segment_retry_cross_stream(self):
        torch.npu.memory._set_allocator_settings("expandable_segments:True")
        # The current stream cannot directly reuse the side stream's cache.
        side = torch.npu.Stream()
        with torch.npu.stream(side):
            cached = torch.empty(int(self.free * 0.60), dtype=torch.uint8, device="npu")
        del cached
        self._assert_retry_clears_error()


if __name__ == "__main__":
    unittest.main()
