import os

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.distributed_c10d import _get_default_group
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
    def _test_hccl_stream_id(cls, rank, world_size, init_pg, c2p):
        dist_group = init_pg(rank, world_size)
        input1 = torch.tensor([1]).npu()
        dist_group.all_reduce(input1)

        default_pg = _get_default_group()
        collective_stream_id = default_pg._get_stream_id(False)
        p2p_stream_id = default_pg._get_stream_id(True)
        assert0 = ((collective_stream_id & 8) == 8)
        assert1 = (collective_stream_id == p2p_stream_id)
        current_stream = torch.npu.current_stream()
        cdata = current_stream._cdata & 0xffff000000000000
        collective_stream = torch.npu.Stream(_cdata=(collective_stream_id + cdata))
        p2p_stream = torch.npu.Stream(_cdata=(collective_stream_id + cdata))
        assert2 = (collective_stream.npu_stream == p2p_stream.npu_stream)

        c2p.put(assert0 and assert1 and assert2)

    def _test_multiprocess(self, f, init_pg, world_size):
        ctx = mp.get_context('spawn')
        c2p = ctx.Queue(world_size)

        ps = []
        for rank in range(world_size):
            p = ctx.Process(target=f, args=(rank, world_size, init_pg, c2p))
            p.start()
            ps.append(p)

        for _ in range(world_size):
            output = c2p.get()
            self.assertEqual(True, output)

        for p in ps:
            p.join()

    @skipIfUnsupportMultiNPU(2)
    def test_dist_get_hccl_stream_id(self):
        # CI currently supports only 2 devices
        ranks = [2]
        for world_size in ranks:
            self._test_multiprocess(HcclStreamIdTest._test_hccl_stream_id,
                                    HcclStreamIdTest._init_dist_hccl, world_size)


if __name__ == '__main__':
    run_tests()