import os
import numpy as np

import torch
import torch.distributed as dist
import torch.distributed.distributed_c10d as c10d
import torch.multiprocessing as mp

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor
from torch_npu.testing.common_distributed import skipIfUnsupportMultiNPU
import torch_npu


class OptionsTest(TestCase):
    @classmethod
    def _init_dist_hccl(cls, rank, options, world_size):
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = '29500'
        os.environ['HCCL_WHITELIST_DISABLE'] = '1'
        torch_npu.npu.set_device(rank)
        dist.init_process_group(backend='hccl', pg_options=options, world_size=world_size, rank=rank)

    @classmethod
    def _test_all_reduce_with_options(cls, rank, ranks, world_size, input1):
        options = torch_npu._C._distributed_c10d.ProcessGroupHCCL.Options()
        hccl_config1 = {"hccl_buffer_size": 300, "group_name": "custom"}
        options.hccl_config = hccl_config1
        OptionsTest._init_dist_hccl(rank, options, world_size)
        input1 = input1.npu()
        dist.all_reduce(input1)
        hccl_config2 = {"hccl_buffer_size": 200}
        options.hccl_config = hccl_config2
        dist.all_reduce(input1)
        default_pg = c10d._get_default_group()._get_backend(torch.device('npu'))
        test_case = TestCase()
        test_case.assertEqual(default_pg.options.hccl_config, hccl_config1,
                              "Once Options are set for a ProcessGroupHCCL, later changes to Options won't affect "
                              "that ProcessGroupHCCL.")
        test_case.assertEqual(default_pg.options.hccl_config.get("group_name", ""), "custom")
        pg = dist.new_group(backend='hccl', ranks=ranks, pg_options=options)
        dist.all_reduce(input1, group=pg)

    def _test_multiprocess(self, f, input1, world_size):
        ctx = mp.get_context('spawn')

        ps = []
        ranks = range(world_size)
        for rank in ranks:
            p = ctx.Process(
                target=f,
                args=(rank, ranks, world_size, input1.cpu()))
            p.start()
            ps.append(p)

        for p in ps:
            p.join()

    @skipIfUnsupportMultiNPU(2)
    def test_all_reduce_with_options(self):
        ranks = [2]
        shape = [np.int32, 0, [2, 3, 16]]
        for world_size in ranks:
            exp_input, input1 = create_common_tensor(shape, -10, 10)
            self._test_multiprocess(OptionsTest._test_all_reduce_with_options,
                                    input1, world_size)


if __name__ == '__main__':
    run_tests()
