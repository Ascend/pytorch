import torch

import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests


class TestNpuStream(TestCase):

    def test_stream_init(self):
        device_number = torch.npu.device_count()
        stream_instance = set()
        for i in range(device_number):
            torch.npu.set_device(i)
            default_stream = torch.npu.default_stream()
            current_stream = torch.npu.current_stream()
            self.assertTrue(default_stream == current_stream)
            stream_instance.add(current_stream)
        self.assertTrue(len(stream_instance) == device_number)

    def test_priority(self):
        s = torch.npu.Stream()
        self.assertTrue((s.stream_id >> 5) == 3)
        s = torch.npu.Stream(priority=0)
        self.assertTrue((s.stream_id >> 5) == 3)
        s = torch.npu.Stream(priority=1)
        self.assertTrue((s.stream_id >> 5) == 3)
        s = torch.npu.Stream(priority=-1)
        self.assertTrue((s.stream_id >> 5) == 4)
        s = torch.npu.Stream(priority=-2)
        self.assertTrue((s.stream_id >> 5) == 4)


if __name__ == "__main__":
    run_tests()
