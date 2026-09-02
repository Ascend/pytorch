import torch
import torch_npu  # noqa: F401
from torch_npu.testing.testcase import TestCase, run_tests


def npu_stream_native_handle(stream: torch.npu.Stream) -> int:
    ts = torch.Stream(stream.stream_id, stream.device_index, stream.device_type)
    return ts.native_handle


class TestStreamNativeHandle(TestCase):
    def test_default_stream_native_handle(self):
        s = torch.npu.current_stream()
        self.assertEqual(npu_stream_native_handle(s), s.npu_stream)

    def test_custom_stream_native_handle(self):
        s = torch.npu.Stream()
        self.assertEqual(npu_stream_native_handle(s), s.npu_stream)

    def test_multiple_streams_unique_and_consistent(self):
        streams = [torch.npu.Stream() for _ in range(3)]
        handles = []
        for s in streams:
            with torch.npu.stream(s):
                nh = npu_stream_native_handle(s)
                self.assertEqual(nh, s.npu_stream)
            handles.append(nh)
        self.assertEqual(len(set(handles)), len(handles))

    def test_current_stream_under_stream_ctx(self):
        s = torch.npu.Stream()
        with torch.npu.stream(s):
            cur = torch.npu.current_stream()
            self.assertEqual(npu_stream_native_handle(cur), cur.npu_stream)
            self.assertEqual(npu_stream_native_handle(cur), s.npu_stream)

    def test_high_priority_stream_native_handle(self):
        s = torch.npu.Stream(priority=-1)
        self.assertEqual(npu_stream_native_handle(s), s.npu_stream)


if __name__ == "__main__":
    run_tests()
