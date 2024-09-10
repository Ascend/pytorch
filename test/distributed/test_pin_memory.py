import threading

import torch
from torch.testing._internal.common_utils import TestCase, run_tests

import torch_npu
from torch_npu.testing.common_distributed import skipIfUnsupportMultiNPU


class TestPinMemory(TestCase):

    @skipIfUnsupportMultiNPU(2)
    def test_pin_memory(self):
        torch.npu.set_device(1)

        def worker_function():
            torch.npu.set_device(0)

        t = threading.Thread(target=worker_function)
        t.start()
        t.join()

        device = torch.npu.current_device()
        self.assertEqual(device, 1)
        pinmemory_tensor = torch.empty(32, pin_memory=True)
        device = torch.npu.current_device()
        self.assertEqual(device, 1)


if __name__ == "__main__":
    run_tests()
