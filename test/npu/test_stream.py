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
                    torch_npu.npu.synchronize()
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

    def test_external_stream_synchronize(self):
        # synchronize() goes through NPUStream::stream(), which drains the host
        # task queue before synchronizing the device stream, so the result is
        # visible without an extra torch_npu.npu.synchronize().
        with self._get_external_stream() as stream_v:
            ext_stream = torch_npu.npu.ExternalStream(stream_v)
            with torch_npu.npu.stream(ext_stream):
                x = torch.ones(1024, device='npu')
                y = x + 1
            ext_stream.synchronize()
            self.assertTrue(ext_stream.query())
            self.assertEqual(y.sum().item(), 2 * x.numel())

    def test_external_stream_query(self):
        with self._get_external_stream() as stream_v:
            ext_stream = torch_npu.npu.ExternalStream(stream_v)
            self.assertIsInstance(ext_stream.query(), bool)
            with torch_npu.npu.stream(ext_stream):
                torch.ones(1024, device='npu').add_(1)
            ext_stream.synchronize()
            self.assertTrue(ext_stream.query())

    def test_external_stream_record_stream(self):
        # A shifted view must protect the whole storage, even after its Python
        # owners are deleted. Check allocator state without a timing-dependent
        # assertion about whether the device has already finished the copy.
        with self._get_external_stream() as stream_v:
            ext = torch_npu.npu.ExternalStream(stream_v)
            base = torch.full((4096,), 7.0, device="npu")
            view = base[1:]
            result = torch.empty_like(view)
            ext.wait_stream(torch_npu.npu.current_stream())
            with torch_npu.npu.stream(ext):
                result.copy_(view)
            view.record_stream(ext)
            view.record_stream(ext)  # Repeated recording must be harmless.
            ext.synchronize()

            before = torch_npu.npu.memory_stats()
            del base, view
            after = torch_npu.npu.memory_stats()
            freed = before["allocated_bytes.all.current"] - after["allocated_bytes.all.current"]
            # free() records an event; malloc() processes it later. The block
            # remains active in between, even if the event has already finished.
            self.assertGreater(freed, 0)
            self.assertEqual(after["active_bytes.all.current"], before["active_bytes.all.current"])
            torch_npu.npu.synchronize()
            self.assertEqual(result.cpu(), torch.full((4095,), 7.0))
            # Retire allocator events while the external stream is still alive.
            torch_npu.npu.empty_cache()

    def test_external_stream_device_synchronize(self):
        # Inspect pinned Host memory directly: copying a device result to CPU
        # here would introduce another synchronization and could hide a missed
        # drain of the shared task queue before device synchronization.
        with self._get_external_stream() as stream_v:
            ext = torch_npu.npu.ExternalStream(stream_v)
            host = torch.zeros(4096, pin_memory=True)
            with torch_npu.npu.stream(ext):
                value = torch.ones(4096, device="npu")
                for _ in range(16):
                    value.add_(1)
                host.copy_(value, non_blocking=True)
            torch_npu.npu.synchronize()
            self.assertEqual(host, torch.full((4096,), 17.0))
            # The pinned allocator records an event when host is freed. The
            # externally owned stream must remain alive until that is submitted.
            del host
            torch_npu.npu.synchronize()

    def test_external_stream_event_record_and_block(self):
        # NPUEvent::record / NPUEvent::block used to reject external streams.
        # The cond child-stream path needs both, and CUDA draws no such
        # distinction, so they now accept any stream.
        with self._get_external_stream() as stream_v:
            ext_stream = torch_npu.npu.ExternalStream(stream_v)
            event = torch_npu.npu.Event()
            with torch_npu.npu.stream(ext_stream):
                x = torch.ones(1024, device='npu')
                y = x + 1
                event.record(ext_stream)
            other = torch_npu.npu.Stream()
            other.wait_event(event)
            with torch_npu.npu.stream(other):
                z = y + 1
            torch_npu.npu.synchronize()
            self.assertEqual(z.sum().item(), 3 * x.numel())

            # The reverse direction: block *on* the external stream. This is the
            # NPUEvent::block / LaunchWaitEventTask side of the same gate; the
            # wait above only exercises record, because it blocks a pool stream.
            back = torch_npu.npu.Event()
            with torch_npu.npu.stream(other):
                w = z + 1
                back.record(other)
            ext_stream.wait_event(back)
            with torch_npu.npu.stream(ext_stream):
                v = w + 1
            ext_stream.synchronize()
            self.assertEqual(v.sum().item(), 5 * x.numel())

    def test_external_stream_capture_allocates_into_graph_pool(self):
        # The consequence of the capture-status gate, not just its return value:
        # the caching allocator asks currentStreamCaptureStatusMayInitCtx() to
        # decide whether an allocation belongs to the graph's private pool. If an
        # external capture stream reports "not capturing", the allocation lands in
        # the default pool, which empty_cache() hands back to the driver while the
        # graph can still be replayed.
        def segment_pool_id_of(tensor):
            storage = tensor.untyped_storage()
            begin = storage.data_ptr()
            end = begin + storage.nbytes()
            for segment in torch_npu.npu.memory_snapshot():
                seg_begin = segment["address"]
                seg_end = seg_begin + segment["total_size"]
                if seg_begin <= begin < seg_end and end <= seg_end:
                    return segment["segment_pool_id"]
            return None

        with self._get_external_stream() as stream_v:
            ext_stream = torch_npu.npu.ExternalStream(stream_v)
            graph = torch_npu.npu.NPUGraph()
            seed = torch.ones(1024, device='npu')
            torch_npu.npu.synchronize()
            with torch_npu.npu.stream(ext_stream):
                graph.capture_begin()
                captured = seed * 2
                graph.capture_end()
            ext_stream.synchronize()
            self.assertEqual(segment_pool_id_of(captured), graph.pool())
            self.assertNotEqual(segment_pool_id_of(seed), graph.pool())
            graph.replay()
            torch_npu.npu.synchronize()
            self.assertEqual(captured.cpu(), torch.full((1024,), 2.0))
            graph.reset()

    def test_external_stream_reports_capture_status(self):
        # currentStreamCaptureStatusMayInitCtx() used to return None for any
        # external current stream. The allocator uses it to decide whether an
        # allocation belongs to a graph pool, so a wrong answer during capture
        # silently routes branch allocations to the default pool.
        with self._get_external_stream() as stream_v:
            ext_stream = torch_npu.npu.ExternalStream(stream_v)
            self.assertFalse(torch_npu.npu.is_current_stream_capturing())
            graph = torch_npu.npu.NPUGraph()
            x = torch.ones(16, device='npu')
            ext_stream.wait_stream(torch_npu.npu.current_stream())
            with torch_npu.npu.stream(ext_stream):
                graph.capture_begin()
                captured = torch_npu.npu.is_current_stream_capturing()
                x.add_(1)
                graph.capture_end()
            ext_stream.synchronize()
            self.assertTrue(captured, "capture status must be visible on an external stream")
            self.assertFalse(torch_npu.npu.is_current_stream_capturing())
            graph.replay()
            torch_npu.npu.synchronize()
            self.assertEqual(x.cpu(), torch.full((16,), 2.0))
            graph.reset()

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
