import time
import os
import torch
from torch.testing._internal.common_utils import TestCase, run_tests
os.environ["PYTORCH_NPU_ALLOC_CONF"] = "pinned_use_background_threads:True"

torch.manual_seed(0)

DEVICE = "npu"
ALLOC_SIZE = 1024 * 64
ITERS = 200


class TestPinnedMemoryBackgroundThreads(TestCase):
    @staticmethod
    def copy_tensor(copy_times=20):
        times = []
        dummy = torch.empty(ALLOC_SIZE // 4, device=DEVICE)
        torch.npu.synchronize()
        streams = [torch.npu.Stream() for _ in range(8)]
        for i in range(copy_times):
            t0 = time.perf_counter_ns()
            buf = torch.empty(
                ALLOC_SIZE // 4,
                dtype=torch.float32,
                pin_memory=True
            )

            with torch.npu.stream(streams[i % len(streams)]):
                dummy.copy_(buf, non_blocking=True)
            t1 = time.perf_counter_ns()
            times.append((t1 - t0) / 1e3) # us
        torch.npu.synchronize()
        return times


    def test_pinned_memory_background_threads(self):
        self.copy_tensor(ITERS)

if __name__ == '__main__':
    run_tests() 