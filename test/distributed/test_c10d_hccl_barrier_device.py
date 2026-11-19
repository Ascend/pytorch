import os
from datetime import timedelta
from unittest.mock import patch

import torch
import torch.distributed as dist
from torch.testing._internal.common_distributed import MultiProcessTestCase
from torch.testing._internal.common_utils import run_tests

import torch_npu  # noqa: F401


class _HcclBarrierTestCase(MultiProcessTestCase):
    @patch.dict(os.environ, {"TORCH_HCCL_ZERO_COPY": "0"})
    def setUp(self):
        super().setUp()
        if not torch.npu.is_available() or torch.npu.device_count() < 2:
            self.skipTest("HCCL barrier selection requires two visible NPUs")
        if not dist.is_backend_available("hccl"):
            self.skipTest("HCCL backend is unavailable")
        try:
            self._spawn_processes()
        finally:
            # Keep the handles even when the base tearDown clears its list.
            self.addCleanup(self._reap_workers, self.processes)

    def _reap_workers(self, processes):
        for process in processes:
            if process.is_alive():
                process.terminate()
        for process in processes:
            process.join(timeout=5)
            if process.is_alive():
                process.kill()
                process.join(timeout=5)
            self.assertFalse(process.is_alive(), "HCCL test worker did not exit")
            process.close()


class HcclBarrierDeviceTest(_HcclBarrierTestCase):
    world_size = 2

    def _test_barrier(self, explicit):
        # No collective history; current differs from the selected device.
        torch.npu.set_device(self.rank)
        selected = torch.device("npu", 1 - self.rank)
        dist.init_process_group(
            "hccl",
            rank=self.rank,
            world_size=self.world_size,
            store=dist.FileStore(self.file_name, self.world_size),
            device_id=None if explicit else selected,
            timeout=timedelta(seconds=60),
        )
        try:
            work = dist.barrier(
                device_ids=[selected.index] if explicit else None, async_op=True
            )
            self.assertEqual([tensor.device for tensor in work.result()], [selected])
            self.assertTrue(work.wait())
            self.assertEqual(torch.npu.current_device(), self.rank)
        finally:
            dist.destroy_process_group()

    def test_barrier_bound_device(self):
        self._test_barrier(explicit=False)

    def test_barrier_device_ids(self):
        self._test_barrier(explicit=True)


class HcclBarrierHistoryTest(_HcclBarrierTestCase):
    world_size = 1

    def test_barrier_with_multiple_used_devices(self):
        torch.npu.set_device(0)
        dist.init_process_group(
            "hccl",
            rank=self.rank,
            world_size=self.world_size,
            store=dist.HashStore(),
            timeout=timedelta(seconds=60),
        )
        try:
            for device_ids, expected in (([0], 0), ([1], 1), (None, 0)):
                work = dist.barrier(device_ids=device_ids, async_op=True)
                self.assertTrue(work.wait())
                self.assertEqual(
                    [t.device for t in work.result()], [torch.device("npu", expected)]
                )
        finally:
            dist.destroy_process_group()


if __name__ == "__main__":
    run_tests()
