import os

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.distributed_c10d import _world
import torch_npu
from torch_npu.testing.common_distributed import skipIfUnsupportMultiNPU
from torch_npu.testing.testcase import TestCase, run_tests


class SetHcclCommNameTest(TestCase):
    @classmethod
    def _init_dist_hccl(cls, rank, world_size):
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = '29501'
        os.environ['HCCL_WHITELIST_DISABLE'] = '1'
        torch_npu.npu.set_device(rank)
        dist.init_process_group(backend='hccl', world_size=world_size, rank=rank)
        return dist

    @classmethod
    def _test_set_hccl_name(cls, rank, world_size, init_pg):
        dist_group = init_pg(rank, world_size)

        pg1 = torch.distributed.new_group()
        isSupportHcclName = torch_npu.distributed._is_support_hccl_comm_name()
        assert isSupportHcclName
        pg1._get_backend(torch.device('npu'))._set_hccl_comm_name("test")
        pg1._get_backend(torch.device('npu'))._set_hccl_comm_name("test")
        pg_name = pg1._get_backend(torch.device('npu')).get_hccl_comm_name(rank)
        assert pg_name == "test"
        pg2 = torch.distributed.new_group()
        pg_name = pg2._get_backend(torch.device('npu')).get_hccl_comm_name(rank)
        pg_name_new = pg2._get_backend(torch.device('npu')).get_hccl_comm_name(rank)
        assert pg_name == pg_name_new

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
    def test_dist_set_hccl_name(self):
        # CI currently supports only 2 devices
        ranks = [2]
        for world_size in ranks:
            self._test_multiprocess(SetHcclCommNameTest._test_set_hccl_name,
                                    SetHcclCommNameTest._init_dist_hccl, world_size)

    def test_dist_set_hccl_name_case_failed(self):
        dist_group = SetHcclCommNameTest._init_dist_hccl(0, 1)
        pg1 = torch.distributed.new_group()
        with self.assertRaises(RuntimeError):
            pg1._get_backend(torch.device('npu'))._set_hccl_comm_name("")
        with self.assertRaises(RuntimeError):
            pg1._get_backend(torch.device('npu'))._set_hccl_comm_name(
                "0123456789012345678901234567890123456789012345678901234567890123456789"
                "0123456789012345678901234567890123456789012345678901234567")
        with self.assertRaises(RuntimeError):
            pg1._get_backend(torch.device('npu'))._set_hccl_comm_name("test")
            pg1._get_backend(torch.device('npu'))._set_hccl_comm_name("test2")
        with self.assertRaises(RuntimeError):
            pg2 = torch.distributed.new_group()
            pg2._get_backend(torch.device('npu')).get_hccl_comm_name(0)
            pg2._get_backend(torch.device('npu'))._set_hccl_comm_name("test")


if __name__ == '__main__':
    run_tests()
