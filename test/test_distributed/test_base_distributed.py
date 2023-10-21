import os
import torch
import torch.distributed as dist
import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests

os.environ["MASTER_ADDR"] = '127.0.0.1'
os.environ["MASTER_PORT"] = "29500"


class DistributedApiTestCase(TestCase):

    def setUp(self):
        self.rank = int(os.environ.get("RANK", default="0"))
        self.world_size = int(os.environ.get("RANK", default="1"))
        dist.init_process_group(backend='hccl', world_size=self.world_size, rank=self.rank)

    def tearDown(self):
        dist.destroy_process_group()

    def test_distributed_backend(self):
        res = dist.Backend("hccl")
        self.assertEqual(res, "hccl")

    def test_distributed_get_backend(self):
        backend = dist.get_backend()
        self.assertEqual(backend, "hccl")

    def test_distributed_get_rank(self):
        res = dist.get_rank()
        self.assertEqual(res, self.rank)

    def test_distributed_get_world_size(self):
        res = dist.get_world_size()
        self.assertEqual(res, self.world_size)

    def test_distributed_is_initialized(self):
        initialized = dist.is_initialized()
        self.assertEqual(initialized, True)

    def test_distributed_is_mpi_available(self):
        res = dist.is_available()
        self.assertIsInstance(res, bool)

    def test_distributed_is_mpi_available(self):
        res = dist.is_mpi_available()
        self.assertIsInstance(res, bool)

    def test_distributed_is_nccl_available(self):
        res = dist.is_nccl_available()
        self.assertIsInstance(res, bool)

    def test_distributed_new_group(self):
        new_group = dist.new_group(ranks=list(range(self.world_size)), backend="hccl")
        self.assertIsInstance(new_group, dist.ProcessGroup)
        self.assertIsInstance(new_group._get_backend(torch.device("npu")),
                              torch_npu._C._distributed_c10d.ProcessGroupHCCL)
        dist.destroy_process_group(new_group)


if __name__ == "__main__":
    run_tests()
