import unittest
import os

import tempfile
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch_npu

from torch_npu.utils._path_manager import PathManager
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_distributed import skipIfUnsupportMultiNPU


class HcclNslbTest(TestCase):
    @classmethod
    def _init_dist_hccl(cls, rank, world_size):
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = '29500'
        os.environ['HCCL_WHITELIST_DISABLE'] = '1'
        os.environ['RANK'] = f'{rank}'
        torch_npu.npu.set_device(rank)
        dist.init_process_group(backend='hccl', world_size=world_size, rank=rank)
        return dist

    @classmethod
    def _test_all_reduce(cls, rank, input1, world_size, init_pg):
        dist_group = init_pg(rank, world_size)
        input1 = input1.npu()
        for _ in range(11):
            dist_group.all_reduce(input1)

    def _test_multiprocess(self, f, init_pg, input1, world_size, nslb_dir):
        
        ctx = mp.get_context('spawn')

        ps = []
        for i in range(world_size):
            p = ctx.Process(
                target=f,
                args=(i, input1, world_size, init_pg))
            p.start()
            ps.append(p)

        for p in ps:
            p.join()

        if len(os.listdir(nslb_dir)) != 4:
            raise RuntimeError(f"Test case fail, nslb file name: {os.listdir(nslb_dir)}")

    @skipIfUnsupportMultiNPU(2)
    def test_dist_nslb(self):
        world_size = 2
        input1 = torch.randn(10)

        tmp_dir = tempfile.mkdtemp()
        nslb_dir = os.path.join(tmp_dir, 'nslb')
        os.environ['NSLB_CP'] = nslb_dir
        os.environ['NSLB_MAX_RECORD_NUM'] = '10'
        try:
            os.mkdir(nslb_dir)
            self._test_multiprocess(HcclNslbTest._test_all_reduce,
                                    HcclNslbTest._init_dist_hccl, input1, world_size, nslb_dir)
        finally:
            PathManager.remove_path_safety(tmp_dir)


if __name__ == '__main__':
    run_tests()
