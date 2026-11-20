import ctypes
from datetime import timedelta
import os

import torch
import torch.distributed as dist
from torch.testing._internal.common_distributed import MultiProcessTestCase
from torch.testing._internal.common_utils import instantiate_parametrized_tests, parametrize, run_tests

import torch_npu
from torch_npu.testing.common_distributed import skipIfUnsupportMultiNPU


class TestExternalStreamCollectives(MultiProcessTestCase):
    @property
    def world_size(self):
        return 2

    def setUp(self):
        super().setUp()
        self._spawn_processes()

    @skipIfUnsupportMultiNPU(2)
    @parametrize("backend", ["hccl", "lccl"])
    def test_collective_ordering(self, backend):
        if backend == "lccl":
            try:
                ctypes.CDLL("liblcal.so")
            except OSError as error:
                self.skipTest(f"LCCL requires its optional liblcal.so dependency: {error}")
        os.environ["TASK_QUEUE_ENABLE"] = "1"
        os.environ["PER_STREAM_QUEUE"] = "0"
        os.environ.pop("ASCEND_LAUNCH_BLOCKING", None)
        torch_npu.npu.set_device(self.rank)
        dist.init_process_group(
            backend=backend, store=dist.FileStore(self.file_name, self.world_size),
            rank=self.rank, world_size=self.world_size, timeout=timedelta(seconds=60),
        )
        rt = torch_npu.npu.npurt()
        raw = ctypes.c_void_p()
        try:
            self.assertEqual(int(rt.npuStreamCreate(ctypes.addressof(raw))), 0)
            stream = torch_npu.npu.ExternalStream(raw.value, device=self.rank)
            with torch_npu.npu.stream(stream):
                # No device synchronization between producer, collective and
                # consumer: both directions of the stream dependency matter.
                value = torch.full((4096,), float(self.rank + 1), device=f"npu:{self.rank}")
                work = dist.all_reduce(value, async_op=True)
                work.wait()
                reduced = value + 1
                value.fill_(self.rank + 5)
                dist.broadcast(value, src=0)
                broadcasted = value + 1
            stream.synchronize()
            self.assertEqual(reduced.cpu(), torch.full((4096,), 4.0))
            self.assertEqual(broadcasted.cpu(), torch.full((4096,), 6.0))
        finally:
            torch_npu.npu.synchronize()
            dist.destroy_process_group()
            if raw.value:
                self.assertEqual(int(rt.npuStreamDestroy(raw.value)), 0)


instantiate_parametrized_tests(TestExternalStreamCollectives)

if __name__ == "__main__":
    run_tests()
