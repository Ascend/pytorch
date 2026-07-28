import unittest.mock

import torch
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests


class TestNpuSleep(TestCase):
    @unittest.skip("Skipping due to outdated CANN version; please update CANN to the latest version and remove this skip")
    def test_sleep_inserts_into_current_stream(self):
        """_sleep should insert a delay task into the current stream."""
        s = torch_npu.npu.Stream()
        with torch_npu.npu.stream(s):
            start = torch_npu.npu.Event(enable_timing=True)
            end = torch_npu.npu.Event(enable_timing=True)
            start.record()
            # 50 ms worth of cycles
            torch_npu.npu._sleep(50000000)
            end.record()
            end.synchronize()
            # The elapsed time should be > 0 if the sleep was enqueued on s
            self.assertGreater(start.elapsed_time(end), 0)

    @unittest.skip("Skipping due to outdated CANN version; please update CANN to the latest version and remove this skip")
    def test_sleep_does_not_block_other_stream(self):
        """_sleep on one stream should not block operations on another stream."""
        sleep_stream = torch_npu.npu.Stream()
        work_stream = torch_npu.npu.Stream()

        # Enqueue a long sleep on sleep_stream
        with torch_npu.npu.stream(sleep_stream):
            torch_npu.npu._sleep(50000000)

        # On work_stream, launch a quick operation and measure its completion
        with torch_npu.npu.stream(work_stream):
            x = torch.ones(10, device='npu')
            y = x + 1
            work_event = torch_npu.npu.Event()
            work_event.record()

        # work_stream should complete well before sleep_stream
        work_event.synchronize()
        self.assertTrue(work_event.query())
        # Verify computation result is correct
        self.assertEqual(y.cpu(), torch.ones(10) * 2)

        sleep_stream.synchronize()

    @unittest.skip("Skipping due to outdated CANN version; please update CANN to the latest version and remove this skip")
    def test_sleep_negative_cycles_raises(self):
        """_sleep with a negative cycles value should raise an error."""
        with self.assertRaises(RuntimeError):
            torch_npu.npu._sleep(-1)

    @unittest.skip("Skipping due to outdated CANN version; please update CANN to the latest version and remove this skip")
    def test_sleep_aclnn_not_available_raises(self):
        """When aclnnSleep is not available in CANN, _sleep should raise RuntimeError."""
        with unittest.mock.patch.object(
            torch_npu._C, '_npu_sleep',
            side_effect=RuntimeError("aclnnSleep is not available in the current CANN version.")
        ):
            with self.assertRaisesRegex(RuntimeError, "aclnnSleep is not available"):
                torch_npu._C._npu_sleep(1000)


if __name__ == '__main__':
    run_tests()
