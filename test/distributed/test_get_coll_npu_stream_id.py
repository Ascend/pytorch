import os
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.distributed_c10d import _world

import torch_npu
from torch_npu.testing.common_distributed import skipIfUnsupportMultiNPU
from torch_npu.testing.testcase import TestCase, run_tests


class CollNpuStreamIdTest(TestCase):
    @classmethod
    def _init_dist_hccl(cls, rank, world_size):
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = '29502'
        os.environ['HCCL_WHITELIST_DISABLE'] = '1'
        torch_npu.npu.set_device(rank)
        dist.init_process_group(backend='hccl', world_size=world_size, rank=rank)
        return dist

    @classmethod
    def _test_coll_npu_stream_id_after_collective(cls, rank, world_size, init_pg, c2p, p2c):
        dist_group = init_pg(rank, world_size)

        backend = _world.default_pg._get_backend(torch.device('npu'))
        device = torch.device(f'npu:{rank}')

        input1 = torch.tensor([1]).npu()
        dist_group.all_reduce(input1)

        stream_id = backend.get_coll_stream_id(device)

        assert0 = (stream_id != -1)
        if stream_id != -1:
            stream = torch.npu.Stream(stream_id=stream_id, device_type=20, device_index=device.index)
            assert1 = (stream.npu_stream is not None)
        else:
            assert1 = False

        c2p.put(assert0 and assert1)
        p2c.get()

    @classmethod
    def _test_coll_npu_stream_id_idempotent(cls, rank, world_size, init_pg, c2p, p2c):
        dist_group = init_pg(rank, world_size)

        backend = _world.default_pg._get_backend(torch.device('npu'))
        device = torch.device(f'npu:{rank}')

        input1 = torch.tensor([1]).npu()
        dist_group.all_reduce(input1)

        stream_id_first = backend.get_coll_stream_id(device)
        stream_id_second = backend.get_coll_stream_id(device)

        assert0 = (stream_id_first == stream_id_second)

        c2p.put(assert0)
        p2c.get()

    @classmethod
    def _test_coll_npu_stream_id_before_collective(cls, rank, world_size, init_pg, c2p, p2c):
        dist_group = init_pg(rank, world_size)

        backend = _world.default_pg._get_backend(torch.device('npu'))
        device = torch.device(f'npu:{rank}')

        stream_id = backend.get_coll_stream_id(device)

        assert0 = (stream_id != -1)
        if stream_id != -1:
            stream = torch.npu.Stream(stream_id=stream_id, device_type=20, device_index=device.index)
            assert1 = (stream.npu_stream is not None)
        else:
            assert1 = False

        c2p.put(assert0 and assert1)
        p2c.get()

    def _run_multiprocess_test(self, test_func, init_pg, world_size):
        ctx = mp.get_context('spawn')
        c2p = ctx.Queue(world_size)
        p2c = ctx.Queue(world_size)

        ps = []
        for rank in range(world_size):
            p = ctx.Process(target=test_func, args=(rank, world_size, init_pg, c2p, p2c))
            p.start()
            ps.append(p)

        for _ in range(world_size):
            output = c2p.get()
            self.assertEqual(True, output)

        for _ in range(world_size):
            p2c.put(0)

        for p in ps:
            p.join()

    @skipIfUnsupportMultiNPU(2)
    def test_coll_npu_stream_id_after_collective(self):
        with patch.dict(os.environ, {"P2P_HCCL_BUFFSIZE": "0"}):
            self._run_multiprocess_test(
                CollNpuStreamIdTest._test_coll_npu_stream_id_after_collective,
                CollNpuStreamIdTest._init_dist_hccl,
                2
            )

    @skipIfUnsupportMultiNPU(2)
    def test_coll_npu_stream_id_idempotent(self):
        with patch.dict(os.environ, {"P2P_HCCL_BUFFSIZE": "0"}):
            self._run_multiprocess_test(
                CollNpuStreamIdTest._test_coll_npu_stream_id_idempotent,
                CollNpuStreamIdTest._init_dist_hccl,
                2
            )

    @skipIfUnsupportMultiNPU(2)
    def test_coll_npu_stream_id_before_collective(self):
        with patch.dict(os.environ, {"P2P_HCCL_BUFFSIZE": "0"}):
            self._run_multiprocess_test(
                CollNpuStreamIdTest._test_coll_npu_stream_id_before_collective,
                CollNpuStreamIdTest._init_dist_hccl,
                2
            )


if __name__ == '__main__':
    run_tests()
