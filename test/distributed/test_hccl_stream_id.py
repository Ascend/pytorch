import os
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.distributed_c10d import _world
import torch_npu
from torch_npu.testing.common_distributed import skipIfUnsupportMultiNPU
from torch_npu.testing.testcase import TestCase, run_tests


class HcclStreamIdTest(TestCase):
    @classmethod
    def _init_dist_hccl(cls, rank, world_size):
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = '29500'
        os.environ['HCCL_WHITELIST_DISABLE'] = '1'
        torch_npu.npu.set_device(rank)
        dist.init_process_group(backend='hccl', world_size=world_size, rank=rank)
        return dist

    @classmethod
    def _test_hccl_stream_id(cls, rank, world_size, init_pg, c2p, p2c):
        dist_group = init_pg(rank, world_size)
        input1 = torch.tensor([1]).npu()
        dist_group.all_reduce(input1)
        collective_stream_id = _world.default_pg._get_backend(torch.device('npu'))._get_stream_id(False)

        send_tensor = torch.ones(2, 2).to(f"npu:{rank}") * 6
        recv_tensor = torch.ones(2, 2).to(f"npu:{rank}") * -1
        dst = 0
        src = 1
        if src == rank:
            dist_group.send(send_tensor, dst)
            p2p_stream_id = _world.default_pg._get_backend(torch.device('npu'))._get_stream_id(True, dst)
        elif dst == rank:
            dist_group.recv(recv_tensor, src)
            p2p_stream_id = _world.default_pg._get_backend(torch.device('npu'))._get_stream_id(True, src)

        stream_num = os.environ.get("STREAMS_PER_DEVICE", 8)
        try:
            stream_num = int(stream_num)
        except Exception:
            stream_num = 8

        if stream_num != 32:
            stream_num = 8
        assert0 = ((collective_stream_id & stream_num) == stream_num)
        assert1 = (collective_stream_id == p2p_stream_id)
        collective_stream = torch.npu.Stream(stream_id=collective_stream_id, device_type=20)
        p2p_stream = torch.npu.Stream(stream_id=collective_stream_id, device_type=20)
        assert2 = (collective_stream.npu_stream == p2p_stream.npu_stream)

        c2p.put(assert0 and assert1 and assert2)
        p2c.get()

    def _test_multiprocess(self, f, init_pg, world_size):
        ctx = mp.get_context('spawn')
        c2p = ctx.Queue(world_size)
        p2c = ctx.Queue(world_size)

        ps = []
        for rank in range(world_size):
            p = ctx.Process(target=f, args=(rank, world_size, init_pg, c2p, p2c))
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
    def test_dist_get_hccl_stream_id_same(self):
        # CI currently supports only 2 devices
        with patch.dict(os.environ, {"P2P_HCCL_BUFFSIZE": "0"}):
            ranks = [2]
            for world_size in ranks:
                self._test_multiprocess(HcclStreamIdTest._test_hccl_stream_id,
                                        HcclStreamIdTest._init_dist_hccl, world_size)


if __name__ == '__main__':
    run_tests()
