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

    def test_get_current_stream_interface(self):
        from torch_npu._C import _npu_getCurrentRawStream
        from torch._dynamo.device_interface import get_interface_for_device

        device_number = torch.npu.device_count()
        for i in range(device_number):
            torch.npu.set_device(i)
            stream = torch.npu.Stream()
            with torch.npu.stream(stream):
                current_stream = torch.npu.current_stream()
                current_raw_stream = _npu_getCurrentRawStream(i)
                interface_raw_stream = get_interface_for_device('npu').get_raw_stream(i)
                self.assertTrue(current_stream.npu_stream == current_raw_stream)
                self.assertTrue(current_stream.npu_stream == interface_raw_stream)

if __name__ == "__main__":
    run_tests()
