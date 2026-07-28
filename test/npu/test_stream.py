import contextlib
import ctypes

import torch

import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
import torch_npu._inductor  # noqa: F401

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
        from torch_npu._C import _npu_getCurrentRawStream, _npu_getCurrentRawStreamNoWait
        from torch._dynamo.device_interface import get_interface_for_device

        device_number = torch.npu.device_count()
        for i in range(device_number):
            torch.npu.set_device(i)
            stream = torch.npu.Stream()
            with torch.npu.stream(stream):
                current_stream = torch.npu.current_stream()
                current_raw_stream = _npu_getCurrentRawStream(i)
                current_raw_stream_no_wait = _npu_getCurrentRawStreamNoWait(i)
                interface_raw_stream = get_interface_for_device('npu').get_raw_stream(i)
                self.assertTrue(current_stream.npu_stream == current_raw_stream)
                self.assertTrue(current_stream.npu_stream == current_raw_stream_no_wait)
                self.assertTrue(current_stream.npu_stream == interface_raw_stream)

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


class TestExternalStream(TestCase):

    @contextlib.contextmanager
    def _get_external_stream(self, device=None):
        rt = torch_npu.npu.npurt()
        stream = ctypes.c_void_p(0)
        with torch_npu.npu.device(device):
            try:
                ret = rt.npuStreamCreate(ctypes.addressof(stream))
                self.assertEqual(int(ret), 0)
                self.assertNotEqual(stream.value, 0)
                yield stream.value
            finally:
                if stream.value:
                    ret = rt.npuStreamDestroy(stream.value)
                    self.assertEqual(int(ret), 0)

    def test_external_stream_creation(self):
        with self._get_external_stream() as stream_v:
            ext_stream = torch_npu.npu.ExternalStream(stream_v)
            self.assertEqual(stream_v, ext_stream.npu_stream)
            self.assertEqual(ext_stream.device.index, torch_npu.npu.current_device())

    def test_external_stream_as_current(self):
        with self._get_external_stream() as stream_v:
            ext_stream = torch_npu.npu.ExternalStream(stream_v)
            with torch_npu.npu.stream(ext_stream):
                self.assertEqual(
                    torch_npu.npu.current_stream().npu_stream,
                    ext_stream.npu_stream)

    def test_external_stream_op(self):
        with self._get_external_stream() as stream_v:
            ext_stream = torch_npu.npu.ExternalStream(stream_v)
            with torch_npu.npu.stream(ext_stream):
                x = torch.randn(10, device='npu')
                y = x + 1
            torch_npu.npu.synchronize()
            self.assertEqual(y.sum().item(), x.sum().item() + x.numel())

    def test_external_stream_same_ptr(self):
        with self._get_external_stream() as stream_v:
            ext1 = torch_npu.npu.ExternalStream(stream_v)
            ext2 = torch_npu.npu.ExternalStream(stream_v)
            self.assertEqual(ext1.npu_stream, ext2.npu_stream)
            self.assertEqual(ext1, ext2)

    def test_external_stream_isinstance(self):
        with self._get_external_stream() as stream_v:
            ext_stream = torch_npu.npu.ExternalStream(stream_v)
            self.assertIsInstance(ext_stream, torch_npu.npu.Stream)
            self.assertIsInstance(ext_stream, torch_npu.npu.ExternalStream)

    def test_external_stream_synchronize_restriction(self):
        with self._get_external_stream() as stream_v:
            ext_stream = torch_npu.npu.ExternalStream(stream_v)
            with self.assertRaisesRegex(RuntimeError, "NPUStream::synchronize"):
                ext_stream.synchronize()

    def test_external_stream_query_restriction(self):
        with self._get_external_stream() as stream_v:
            ext_stream = torch_npu.npu.ExternalStream(stream_v)
            with self.assertRaisesRegex(RuntimeError, "Cannot query"):
                ext_stream.query()


if __name__ == "__main__":
    run_tests()
