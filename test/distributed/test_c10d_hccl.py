# Owner(s): ["module: npu"]
import os

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.distributed_c10d import _world

import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_distributed import skipIfUnsupportMultiNPU
from torch_npu.testing.common_utils import SupportedDevices

# Symmetric kernels are not available for dtypes like int64, so use float32.
NUMEL = 1024 * 1024 * 2


class HcclMemPoolRegistrationTest(TestCase):

    @classmethod
    def _init_dist_hccl(cls, rank, world_size):
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = '29510'
        os.environ['HCCL_WHITELIST_DISABLE'] = '1'
        torch_npu.npu.set_device(rank)
        dist.init_process_group(backend='hccl', world_size=world_size, rank=rank)

    @classmethod
    def _test_mem_pool_register(cls, rank, world_size, symm, c2p, p2c):
        cls._init_dist_hccl(rank, world_size)
        device = torch.device(f"npu:{rank}")
        backend = _world.default_pg._get_backend(torch.device('npu'))

        # register_mem_pool needs the HCCL communicator to already exist; a
        # warm-up collective forces its creation (NCCL does this via device_id).
        dist.all_reduce(torch.ones(8, device=device))
        torch_npu.npu.synchronize()

        # Route allocations through the HCCL allocator (hcclMemAlloc).
        pool = torch_npu.npu.MemPool(backend.mem_allocator)
        with torch_npu.npu.use_mem_pool(pool):
            tensor = torch.arange(NUMEL, device=device, dtype=torch.float32)

        backend.register_mem_pool(pool, symm=symm)
        dist.all_reduce(tensor)

        # Allocations made after registration must be registered too, via the
        # allocator trace hook installed by register_mem_pool.
        with torch_npu.npu.use_mem_pool(pool):
            tensor_post = torch.arange(NUMEL, device=device, dtype=torch.float32)
        dist.all_reduce(tensor_post)
        torch_npu.npu.synchronize()

        backend.deregister_mem_pool(pool)

        result = (tensor.cpu(), tensor_post.cpu())
        del tensor, tensor_post, pool
        c2p.put((rank, result))
        p2c.get()

    def _run_multiprocess(self, symm, world_size):
        ctx = mp.get_context('spawn')
        c2p = ctx.Queue(world_size)
        p2c = ctx.Queue(world_size)

        ps = []
        for rank in range(world_size):
            p = ctx.Process(
                target=HcclMemPoolRegistrationTest._test_mem_pool_register,
                args=(rank, world_size, symm, c2p, p2c))
            p.start()
            ps.append(p)

        expected = torch.arange(NUMEL, dtype=torch.float32) * world_size
        for _ in range(world_size):
            rank, (pre, post) = c2p.get()
            self.assertEqual(pre, expected,
                             f"rank {rank}: allreduce over pre-registration buffer mismatch")
            self.assertEqual(post, expected,
                             f"rank {rank}: allreduce over post-registration buffer mismatch")

        for _ in range(world_size):
            p2c.put(0)
        for p in ps:
            p.join()

    @skipIfUnsupportMultiNPU(8)
    @SupportedDevices(['Ascend910_93', 'Ascend950'])
    def test_hccl_window_registration(self):
        self._run_multiprocess(symm=True, world_size=8)

    @skipIfUnsupportMultiNPU(1)
    @SupportedDevices(['Ascend910_93', 'Ascend950'])
    def test_hccl_window_registration_single_npu(self):
        # Single-rank smoke test via the symmetric-window path.
        self._run_multiprocess(symm=True, world_size=1)


if __name__ == '__main__':
    run_tests()
