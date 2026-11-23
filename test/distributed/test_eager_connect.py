import unittest

import torch
import torch.distributed as dist
from torch.testing._internal.common_distributed import MultiProcessTestCase
from torch.testing._internal.common_utils import run_tests

import torch_npu


@unittest.skipUnless(torch.npu.device_count() >= 2, "requires two NPUs")
class TestEagerConnect(MultiProcessTestCase):
    world_size = 2

    def setUp(self):
        super().setUp()
        self._spawn_processes()

    def test_eager_connect(self):
        device_id = torch.device("npu", self.rank)
        torch.npu.set_device(device_id)
        dist.init_process_group(
            "hccl", store=dist.FileStore(self.file_name, self.world_size),
            rank=self.rank, world_size=self.world_size,
        )
        pg = dist.group.WORLD
        eager_backend = pg._get_backend(device_id)

        # init_comm=False reads the cache without creating a communicator.
        self.assertEqual(eager_backend.get_hccl_comm_name(self.rank, init_comm=False), "")

        eager_backend.eager_connect_single_device(device_id)
        name = eager_backend.get_hccl_comm_name(self.rank, init_comm=False)
        self.assertNotEqual(name, "")

        eager_backend.eager_connect_single_device(device_id)
        self.assertEqual(eager_backend.get_hccl_comm_name(self.rank, init_comm=False), name)
        value = torch.tensor([self.rank + 1.0], device=device_id)
        dist.all_reduce(value)
        self.assertEqual(value.item(), 3.0)


if __name__ == "__main__":
    run_tests()
