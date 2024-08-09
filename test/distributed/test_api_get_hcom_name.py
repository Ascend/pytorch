import os

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.distributed_c10d import _world
import torch_npu
from torch_npu.testing.common_distributed import skipIfUnsupportMultiNPU
from torch_npu.testing.testcase import TestCase, run_tests


class GetHcclCommNameTest(TestCase):
    @classmethod
    def _init_dist_hccl(cls, rank, world_size):
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = '29500'
        os.environ['HCCL_WHITELIST_DISABLE'] = '1'
        torch_npu.npu.set_device(rank)
        dist.init_process_group(backend='hccl', world_size=world_size, rank=rank)
        return dist

    @classmethod
    def _test_hccl_name(cls, rank, world_size, init_pg):
        dist_group = init_pg(rank, world_size)
        pg1 = torch.distributed.new_group()
        assert pg1._get_backend(torch.device('npu')).get_hccl_comm_name(rank) != ""
        pg2 = torch.distributed.new_group()
        assert pg2._get_backend(torch.device('npu')).get_hccl_comm_name(rank, init_comm=False) == ""
        assert pg2._get_backend(torch.device('npu')).get_hccl_comm_name(rank, init_comm=True) != ""
        pg3 = torch.distributed.new_group()
        assert pg3._get_backend(torch.device('npu')).get_hccl_comm_name(rank, init_comm=True) != ""

    def _test_multiprocess(self, f, init_pg, world_size):
        ctx = mp.get_context('spawn')
        ps = []
        for rank in range(world_size):
            p = ctx.Process(target=f, args=(rank, world_size, init_pg))
            p.start()
            ps.append(p)

        for p in ps:
            p.join()

    @skipIfUnsupportMultiNPU(2)
    def test_dist_get_hccl_name(self):
        # CI currently supports only 2 devices
        ranks = [2]
        for world_size in ranks:
            self._test_multiprocess(GetHcclCommNameTest._test_hccl_name,
                                    GetHcclCommNameTest._init_dist_hccl, world_size)


if __name__ == '__main__':
    run_tests()
