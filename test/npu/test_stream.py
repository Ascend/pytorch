import contextlib
import ctypes
import os
import subprocess
import sys
import tempfile

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
        from torch_npu.utils._dynamo import _dynamo_register_interface_for_device

        # device_interface is an internal Dynamo module. Initialize its NPU
        # registration explicitly instead of relying on import torch_npu.
        _dynamo_register_interface_for_device()

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

    def test_reconstruct_stream_by_unpack3(self):
        """Reconstruct an existing stream via stream_id/device_index/device_type (valid unpack3 path)."""
        s1 = torch.npu.Stream()
        s2 = torch.npu.Stream(
            stream_id=s1.stream_id,
            device_index=s1.device_index,
            device_type=s1.device_type,
        )
        self.assertEqual(s1.stream_id, s2.stream_id)
        self.assertEqual(s1.device_index, s2.device_index)
        self.assertEqual(s1.device_type, s2.device_type)
        # Reconstructed stream should be usable
        s2.synchronize()

    def test_fake_stream_id_fail_fast(self):
        """Fake stream_id should fail at construction (fail-fast)."""
        with self.assertRaises(RuntimeError):
            torch.npu.Stream(stream_id=99999, device_index=0, device_type=20)

    def test_fake_device_index_fail_fast(self):
        """Fake device_index should fail at construction (fail-fast)."""
        with self.assertRaises(RuntimeError):
            torch.npu.Stream(stream_id=0, device_index=999, device_type=20)

    def test_fake_device_type_fail_fast(self):
        """Fake device_type should fail at construction (fail-fast)."""
        with self.assertRaises(RuntimeError):
            torch.npu.Stream(stream_id=0, device_index=0, device_type=999)

    def test_negative_device_index_fail_fast(self):
        """Negative device_index should fail at construction.
        Note: PyArg format "L" (unsigned) intercepts negative values as
        OverflowError before reaching C++ validation; accept both types."""
        with self.assertRaises((RuntimeError, OverflowError)):
            torch.npu.Stream(stream_id=0, device_index=-1, device_type=20)

    def test_negative_stream_id_fail_fast(self):
        """Negative stream_id should fail at construction.
        Note: PyArg format "L" (unsigned) intercepts negative values as
        OverflowError before reaching C++ validation; accept both types."""
        with self.assertRaises((RuntimeError, OverflowError)):
            torch.npu.Stream(stream_id=-1, device_index=0, device_type=20)

    def test_negative_device_type_fail_fast(self):
        """Negative device_type should fail at construction.
        Note: PyArg format "L" (unsigned) intercepts negative values as
        OverflowError before reaching C++ validation; accept both types."""
        with self.assertRaises((RuntimeError, OverflowError)):
            torch.npu.Stream(stream_id=0, device_index=0, device_type=-1)

    def test_sync_launch_stream_index_bounds(self):
        """SYNCLAUNCH stream index out of bounds should fail at construction.

        kSyncLaunchStreamsPerPool = 4, so valid indices are 0-3.
        Adding 4 to a valid SYNCLAUNCH stream_id creates an out-of-bounds index
        (>= 4) while keeping the SYNCLAUNCH type, which should be caught by the
        bounds check in NPUStream_internals.
        """
        s = torch.npu.Stream(is_sync_launch=True)
        fake_stream_id = s.stream_id + 4
        with self.assertRaises(RuntimeError):
            torch.npu.Stream(
                stream_id=fake_stream_id,
                device_index=s.device_index,
                device_type=s.device_type,
            )

    def test_reconstruct_sync_launch_stream(self):
        """Reconstruct a valid SYNCLAUNCH stream via unpack3 (valid path)."""
        s1 = torch.npu.Stream(is_sync_launch=True)
        s2 = torch.npu.Stream(
            stream_id=s1.stream_id,
            device_index=s1.device_index,
            device_type=s1.device_type,
        )
        self.assertEqual(s1.stream_id, s2.stream_id)
        self.assertEqual(s1.device_index, s2.device_index)
        self.assertEqual(s1.device_type, s2.device_type)
        s2.synchronize()

    def test_reconstruct_default_stream(self):
        """Reconstruct the default stream via unpack3 (valid path)."""
        s1 = torch.npu.default_stream()
        s2 = torch.npu.Stream(
            stream_id=s1.stream_id,
            device_index=s1.device_index,
            device_type=s1.device_type,
        )
        self.assertEqual(s1.stream_id, s2.stream_id)
        self.assertEqual(s1.device_index, s2.device_index)
        self.assertEqual(s1.device_type, s2.device_type)
        s2.synchronize()


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

    def test_external_stream_npu_stream_per_stream_queue_restriction(self):
        env = os.environ.copy()
        env["TASK_QUEUE_ENABLE"] = "1"
        env["PER_STREAM_QUEUE"] = "1"
        env.pop("ASCEND_LAUNCH_BLOCKING", None)
        with tempfile.TemporaryDirectory() as tmp:
            script = os.path.join(tmp, "ext_stream_psq.py")
            with open(script, "w") as f:
                f.write(
                    "import ctypes\n"
                    "import torch_npu\n"
                    "rt = torch_npu.npu.npurt()\n"
                    "stream = ctypes.c_void_p(0)\n"
                    "ret = rt.npuStreamCreate(ctypes.addressof(stream))\n"
                    "assert int(ret) == 0 and stream.value != 0\n"
                    "ext = torch_npu.npu.ExternalStream(stream.value)\n"
                    "try:\n"
                    "    _ = ext.npu_stream\n"
                    "    raise SystemExit('expected RuntimeError not raised')\n"
                    "except RuntimeError as e:\n"
                    "    assert 'not supported in NPUStream::stream()' in str(e), str(e)\n"
                    "finally:\n"
                    "    if stream.value:\n"
                    "        rt.npuStreamDestroy(stream.value)\n"
                )
            r = subprocess.run([sys.executable, script], env=env,
                               capture_output=True, text=True, timeout=180)
        self.assertEqual(r.returncode, 0,
                         f"subprocess failed:\nstdout:\n{r.stdout}\nstderr:\n{r.stderr}")

    def test_external_stream_drains_default_repo(self):
        with self._get_external_stream() as stream_v:
            ext_stream = torch_npu.npu.ExternalStream(stream_v)
            x = torch.randn(100, device='npu')
            y = x + 1
            self.assertEqual(ext_stream.npu_stream, stream_v)
            torch_npu.npu.synchronize()
            self.assertEqual(y.sum().item(), x.sum().item() + x.numel())


if __name__ == "__main__":
    run_tests()
