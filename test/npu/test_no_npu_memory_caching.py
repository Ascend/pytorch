import gc
import os


os.environ["PYTORCH_NO_NPU_MEMORY_CACHING"] = "1"

import torch

import torch_npu
from torch_npu.testing.testcase import run_tests, TestCase


class TestNoNpuMemoryCaching(TestCase):
    def test_aclgraph_capture_reset_releases_reserved(self):
        # With PYTORCH_NO_NPU_MEMORY_CACHING=1, capture-time allocations must still go through
        # caching semantics for address stability, but when the graph is reset/destroyed,
        # the pool should release physical memory promptly.
        torch.npu.set_device(0)
        torch_npu.npu.empty_cache()
        gc.collect()
        torch_npu.npu.synchronize()

        g = torch_npu.npu.NPUGraph()

        x = torch.empty((512, 512), device="npu", dtype=torch.float32)
        y = torch.empty((512, 512), device="npu", dtype=torch.float32)
        torch_npu.npu.synchronize()

        s0 = torch_npu.npu.Stream()
        s0.wait_stream(torch_npu.npu.current_stream())
        with torch_npu.npu.stream(s0):
            g.capture_begin()
            z = x @ y
            g.capture_end()

        capture_alloc = torch_npu.npu.memory_allocated()
        capture_rsv = torch_npu.npu.memory_reserved()
        torch_npu.npu.synchronize()

        g.replay()
        replay_alloc = torch_npu.npu.memory_allocated()
        replay_rsv = torch_npu.npu.memory_reserved()

        del z
        gc.collect()
        torch_npu.npu.synchronize()

        g.reset()
        del g
        gc.collect()
        torch_npu.npu.synchronize()

        self.assertEqual(capture_alloc, replay_alloc)
        self.assertEqual(capture_rsv, replay_rsv)


if __name__ == "__main__":
    run_tests()
