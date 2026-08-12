import os
import torch
import torch.distributed as dist
from torch.testing._internal.common_utils import run_tests
from torch.testing._internal.distributed._tensor.common_dtensor import DTensorTestBase

import torch_npu  # noqa: F401
from torch_npu.testing.common_distributed import with_comms
from torch.distributed.distributed_c10d import _get_default_group

os.environ["HCCL_NPU_SOCKET_PORT_RANGE"] = "10000,60000"
os.environ["TIMEOUT"] = "1800"
os.environ["TORCH_TEST_TIMEOUT"] = "1800"
class TestHcclGroupStartEnd(DTensorTestBase):

    def _get_hccl_backend(self):
        if dist.get_backend() != "hccl":
            self.skipTest(f"Current backend is {dist.get_backend()}, but this test requires hccl. Skipping.")
        pg = _get_default_group()
        return pg._get_backend(torch.device(self.device_type))

    @with_comms
    def test_group_all_reduce_reliability(self):
        backend = self._get_hccl_backend()
        n = 4 * 1024 * 1024

        x = torch.ones(n, device=self.device_type) * (self.rank + 1)
        y = torch.ones(n, device=self.device_type) * (self.rank + 1)
        w = torch.ones(n, device=self.device_type)

        backend._group_start()
        work_x = torch.distributed.all_reduce(x, async_op=True)
        work_y = torch.distributed.all_reduce(y, async_op=True)
        backend._group_end()

        work_x.wait()
        self.assertTrue(work_x.is_completed())
        self.assertTrue(work_y.is_completed())
        result_x = torch.matmul(x, w)

        work_y.wait()
        result_y = torch.matmul(y, w)

        expected = sum(range(1, self.world_size + 1))

        self.assertEqual(x, torch.full_like(x, expected))
        self.assertEqual(y, torch.full_like(y, expected))
        self.assertEqual(result_x, expected * n)
        self.assertEqual(result_y, expected * n)

    @with_comms
    def test_group_p2p_reliability(self):
        if self.world_size < 2:
            return

        backend = self._get_hccl_backend()
        n = 1024 * 1024

        x = torch.ones(n, device=self.device_type) * (self.rank + 1)
        y = torch.ones(n, device=self.device_type) * (self.rank + 1)

        reqs = []

        backend._group_start()
        if self.rank == 0:
            reqs.append(torch.distributed.isend(x, dst=1))
            reqs.append(torch.distributed.isend(y, dst=1))
        elif self.rank == 1:
            reqs.append(torch.distributed.irecv(x, src=0))
            reqs.append(torch.distributed.irecv(y, src=0))
        backend._group_end()

        for req in reqs:
            req.wait()
            self.assertTrue(req.is_completed())

        if self.rank == 1:
            self.assertEqual(x, torch.full_like(x, 1))
            self.assertEqual(y, torch.full_like(y, 1))

if __name__ == "__main__":
    run_tests()
