import unittest
import contextlib
import collections
import multiprocessing
import threading
import sys
from subprocess import check_output

import torch
import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_distributed import skipIfUnsupportMultiNPU


class TorchNPUDeviceTestCase(TestCase):
    def test_npu_current_device(self):
        res = torch_npu.npu.current_device()
        self.assertIsInstance(res, int)

    @skipIfUnsupportMultiNPU(2)
    def test_npu_can_device_access_peer_multi_npu(self):
        res = torch_npu.npu.can_device_access_peer(0, 1)
        self.assertEqual(res, True)

    def test_npu_can_device_access_peer(self):
        res = torch_npu.npu.can_device_access_peer(0, 0)
        self.assertEqual(res, False)
        with self.assertRaisesRegex(AssertionError, "Invalid devide id"):
            torch_npu.npu.can_device_access_peer(-1, 0)
        with self.assertRaisesRegex(AssertionError, "Invalid peer devide id"):
            torch_npu.npu.can_device_access_peer(0, -1)

    def test_npu_device(self):
        res = torch_npu.npu.device("npu:0")
        self.assertIsInstance(res, torch_npu.npu.device)

    def test_npu_device_count(self):
        res = torch_npu.npu.device_count()
        self.assertIsInstance(res, int)

    def test_npu_device_of(self):
        x = torch.Tensor([1, 2, 3]).to("npu")
        res = torch_npu.npu.device_of(x)
        self.assertIsInstance(res, torch_npu.npu.device_of)

    def test_npu_get_device_name(self):
        res = torch_npu.npu.get_device_name(0)
        self.assertIsInstance(res, str)
        res = torch_npu.npu.get_device_name()
        self.assertIsInstance(res, str)
        res = torch_npu.npu.get_device_name("npu:0")
        self.assertIsInstance(res, str)
        device = torch.device("npu:0")
        res = torch_npu.npu.get_device_name(device)
        self.assertIsInstance(res, str)

    def test_npu_get_device_properties(self):
        name = torch_npu.npu.get_device_properties(0).name
        self.assertIsInstance(name, str)
        total_memory = torch_npu.npu.get_device_properties(0).total_memory
        self.assertIsInstance(total_memory, int)

    def test_npu_get_device_capability(self):
        res = torch_npu.npu.get_device_capability()
        self.assertEqual(res, None)
        res = torch_npu.npu.get_device_capability(0)
        self.assertEqual(res, None)
        name = torch_npu.npu.get_device_properties(0).name
        res = torch_npu.npu.get_device_capability(name)
        self.assertEqual(res, None)
        device = torch_npu.npu.device("npu")
        res = torch_npu.npu.get_device_capability(device)
        self.assertEqual(res, None)

    def test_npu_mem_get_info(self):
        before_free_memory, before_total_memory = torch_npu.npu.mem_get_info(0)
        torch.randn(1024 * 1024 * 1024, device='npu:0')
        torch_npu.npu.synchronize()
        after_free_memory, after_total_memory = torch_npu.npu.mem_get_info(0)
        self.assertEqual(before_total_memory, after_total_memory)

class TorchNPUMemoryApiTestCase(TestCase):
    def test_npu_memory_stats(self):
        res = torch_npu.npu.memory_stats()
        self.assertIsInstance(res, collections.OrderedDict)

    def test_npu_memory_summary(self):
        res = torch_npu.npu.memory_summary()
        self.assertIsInstance(res, str)

    def test_npu_memory_snapshot(self):
        res = torch_npu.npu.memory_snapshot()
        self.assertIsInstance(res, list)

    def test_npu_memory_allocated(self):
        res = torch_npu.npu.memory_allocated()
        self.assertIsInstance(res, int)

    def test_npu_max_memory_allocated(self):
        res = torch_npu.npu.max_memory_allocated()
        self.assertIsInstance(res, int)

    def test_npu_reset_max_memory_allocated(self):
        res = torch_npu.npu.reset_max_memory_allocated()
        self.assertIsNone(res)

    def test_npu_memory_reserved(self):
        res = torch_npu.npu.memory_reserved()
        self.assertIsInstance(res, int)

    def test_npu_max_memory_reserved(self):
        res = torch_npu.npu.max_memory_reserved()
        self.assertIsInstance(res, int)

    def test_npu_memory_cached(self):
        res = torch_npu.npu.memory_cached()
        self.assertIsInstance(res, int)

    def test_npu_max_memory_cached(self):
        res = torch_npu.npu.max_memory_cached()
        self.assertIsInstance(res, int)

    def test_npu_reset_max_memory_cached(self):
        res = torch_npu.npu.reset_max_memory_cached()
        self.assertIsNone(res)


class TorchNPUSyncApiTestCase(TestCase):
    def test_set_sync_debug_mode(self):
        with self.assertRaisesRegex(RuntimeError, "invalid value of debug_mode, expected one of 0,1,2"):
            torch.npu.set_sync_debug_mode(-1)
        with self.assertRaisesRegex(RuntimeError, "invalid value of debug_mode, expected one of `default`, `warn`, `error`"):
            torch.npu.set_sync_debug_mode("unexpected")

    def test_get_sync_debug_mode(self):
        res = torch.npu.get_sync_debug_mode()
        self.assertEqual(res, 0)
        torch.npu.set_sync_debug_mode(1)
        res = torch.npu.get_sync_debug_mode()
        self.assertEqual(res, 1)
        torch.npu.set_sync_debug_mode('error')
        res = torch.npu.get_sync_debug_mode()
        self.assertEqual(res, 2)
        with self.assertRaisesRegex(RuntimeError, "invalid value of debug_mode, expected one of 0,1,2"):
            torch.npu.set_sync_debug_mode(3)
        res = torch.npu.get_sync_debug_mode()
        self.assertEqual(res, 2)


class TorchNPUApiTestCase(TestCase):
    def test_npu_current_stream(self):
        res = torch_npu.npu.current_stream()
        self.assertIsInstance(res, torch_npu.npu.streams.Stream)

    def test_npu_default_stream(self):
        res = torch_npu.npu.default_stream()
        self.assertIsInstance(res, torch_npu.npu.streams.Stream)

    def test_npu_current_blas_handle(self):
        res = torch_npu.npu.current_blas_handle()
        self.assertIsNone(res)

    def test_npu_init(self):
        self.assertIsNone(torch_npu.npu.init())

    def test_npu_is_available(self):
        res = torch_npu.npu.is_available()
        self.assertIsInstance(res, bool)

    def test_npu_is_initialized(self):
        res = torch_npu.npu.is_initialized()
        self.assertIsInstance(res, bool)

    def test_npu_stream(self):
        s = torch_npu.npu.current_stream()
        res = torch_npu.npu.stream(s)
        self.assertIsInstance(res, torch_npu.npu.utils.StreamContext)

    def test_npu_streamcontext(self):
        s = torch_npu.npu.current_stream()

        def thread_func():
            res1 = torch_npu.npu.Stream()
            res2 = torch_npu.npu.Stream()
            with torch_npu.npu.stream(res1) as current:
                self.assertEqual(torch_npu.npu.current_stream(), res1)
            with torch_npu.npu.stream(res2) as current:
                self.assertEqual(torch_npu.npu.current_stream(), res2)
            with torch_npu.npu.stream(res1) as current:
                self.assertEqual(torch_npu.npu.current_stream(), res1)
            with torch_npu.npu.stream(res2) as current:
                self.assertEqual(torch_npu.npu.current_stream(), res2)

        thread = threading.Thread(target=thread_func)
        thread.start()
        thread.join()
        self.assertEqual(torch_npu.npu.current_stream(), s)

    def test_npu_synchronize(self):
        res = torch_npu.npu.synchronize()
        self.assertIsNone(res)

    def test_npu_stream_class(self):
        s = torch_npu.npu.Stream()
        self.assertIsInstance(s, torch_npu.npu.Stream)

    def test_npu_stream_record_evnet(self):
        s = torch_npu.npu.current_stream()
        res = s.record_event()
        self.assertIsInstance(res, torch_npu.npu.Event)

    def test_npu_stream_synchronize(self):
        s = torch_npu.npu.current_stream()
        self.assertIsNone(s.synchronize())

    def test_npu_stream_wait_event(self):
        s = torch_npu.npu.current_stream()
        e = torch_npu.npu.Event()
        self.assertIsNone(s.wait_event(e))

    @unittest.skip("skip test_npu_stream_query now")
    def test_npu_stream_query(self):
        t = torch.ones(4096, 4096).npu()
        s = torch_npu.npu.current_stream()
        # .npu() is synchronous interface, all the work has been completed.
        self.assertEqual(s.query(), True)
        t = torch.ones(4096, 4096, device='npu')
        # .ones() op asynchronous execution，work is not completed when Stream.query()
        self.assertEqual(s.query(), False)

    def test_npu_event(self):
        res = torch_npu.npu.Event(enable_timing=True, blocking=True, interprocess=True)
        self.assertIsInstance(res, torch_npu.npu.Event)

    def test_npu_event_elapsed_time(self):
        start_event = torch_npu.npu.Event(enable_timing=True)
        end_event = torch_npu.npu.Event(enable_timing=True)
        start_event.record()
        end_event.record()
        res = start_event.elapsed_time(end_event)
        self.assertIsInstance(res, float)
    
    def test_npu_event_recorded_time(self):
        event_1 = torch_npu.npu.Event(enable_timing=True)
        event_1.record()
        event_2 = torch_npu.npu.Event(enable_timing=True)
        event_2.record()
        try:
            time_stamp_1 = event_1.recorded_time()
        except RuntimeError as e:
            self.assertIn("Failed to find function aclrtEventGetTimestamp", str(e), f"{e}")
        except Exception as e:
            self.fail(f"{e}")
        else:
            self.assertIsInstance(time_stamp_1, int)
            self.assertGreater(time_stamp_1, 0)
            time_stamp_2 = event_2.recorded_time()
            self.assertIsInstance(time_stamp_2, int)
            self.assertGreater(time_stamp_2, 0)
            self.assertGreater(time_stamp_2, time_stamp_1)

    def test_npu_event_query(self):
        event = torch_npu.npu.Event()
        res = event.query()
        self.assertIsInstance(res, bool)

    def test_npu_event_record(self):
        event = torch_npu.npu.Event()
        self.assertIsNone(event.record())

    def test_npu_event_synchronize(self):
        event = torch_npu.npu.Event()
        self.assertIsNone(event.synchronize())

    def test_npu_event_wait(self):
        event = torch_npu.npu.Event()
        self.assertIsNone(event.wait())

    def test_npu_empty_cache(self):
        self.assertIsNone(torch_npu.npu.empty_cache())

    def test_npu_get_aclnn_version(self):
        res = torch_npu.npu.aclnn.version()
        self.assertEqual(res, None)

    def test_lazy_init(self):
        def run(queue):
            try:
                a = torch.tensor([2]).to('npu:0')
            except Exception as e:
                queue.put(e)

        torch_npu.npu.current_device()
        with self.assertRaisesRegex(RuntimeError, "Cannot re-initialize NPU in forked subprocess"):
            result_queue = multiprocessing.Queue()
            p = multiprocessing.Process(target=run, args=(result_queue,))
            p.start()
            p.join(timeout=100)
            if not result_queue.empty():
                raise result_queue.get()

    def test_npu_device_count_without_visible_devices(self):
        test_script = f"import torch; import torch_npu; \
        count1 = torch.npu.device_count(); count2 = torch_npu._C._npu_getDeviceCount(); print(count1 == count2)"
        rc = check_output([sys.executable, '-c', test_script]).decode("ascii").strip()
        self.assertEqual(rc, "True")

    @skipIfUnsupportMultiNPU(2)
    def test_npu_device_count_with_visible_devices(self):
        for var in ['', ',', ' ,', ', ', '0,', ',0', '0, ', '0, 1', '0 ,1', '0,1', '0,32,1', '0,32,0', '0,0', '0,1,1', 'npu0', '1,0']:
            test_script = f"import os; import torch; import torch_npu; os.environ['ASCEND_RT_VISIBLE_DEVICES'] = '{var}'; \
            count1 = torch.npu.device_count(); count2 = torch_npu._C._npu_getDeviceCount(); print(count1 == count2)"
            rc = check_output([sys.executable, '-c', test_script]).decode("ascii").strip()
            self.assertEqual(rc, "True")

    def test_npu_lazy_init(self):
        """ Validate that no NPU calls are made during `import torch_npu` call or `torch.npu.device_count()` call"""
        VISIBLE_DEVICES = "ASCEND_RT_VISIBLE_DEVICES"
        # Check that `rts` was not called during the import
        # By using torch_npu._C._npu_getDeviceCount() because it will not change if `rts` was called
        # torch_npu.npu.device_count() will parses ASCEND_RT_VISIBLE_DEVICES and will change along with it
        test_script = f"import os; import torch; import torch_npu; os.environ['{VISIBLE_DEVICES}']='32';print(torch_npu._C._npu_getDeviceCount())"
        rc = check_output([sys.executable, "-c", test_script]).decode("ascii").strip()
        self.assertEqual(rc, "0")

        test_script = f"import os; import torch; import torch_npu; torch.npu.device_count(); os.environ['{VISIBLE_DEVICES}']='32';print(torch_npu._C._npu_getDeviceCount())"
        rc = check_output([sys.executable, "-c", test_script]).decode("ascii").strip()
        self.assertEqual(rc, "0")

    @skipIfUnsupportMultiNPU(2)
    def test_device_count_not_cached_pre_init(self):
        visible_devices = "ASCEND_RT_VISIBLE_DEVICES"
        test_script = f"""\
import torch
import torch_npu
import os
r1 = torch.npu.device_count()
os.environ['{visible_devices}'] = '0'
r2 = torch.npu.device_count()
torch.empty(10, device='npu')
print(f"{{r1}}, {{r2}}")
"""

        r = (
            check_output([sys.executable, "-c", test_script])
            .decode("ascii")
            .strip()
        )

        x = torch.npu.device_count()
        self.assertEqual(f"{x}, 1", r)


if __name__ == "__main__":
    run_tests()
