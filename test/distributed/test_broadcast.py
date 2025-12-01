import unittest
import os

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor
from torch_npu.testing.common_distributed import skipIfUnsupportMultiNPU


class HcclBroadcastTest(TestCase):

    @classmethod
    def _init_dist_hccl(cls, rank, world_size):
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = '29500'
        os.environ['HCCL_WHITELIST_DISABLE'] = '1'
        torch_npu.npu.set_device(rank)
        dist.init_process_group(backend='hccl', world_size=world_size, rank=rank)
        return dist

    @classmethod
    def _test_broadcast_with_internal_format_and_offset(cls, rank, input1, world_size, init_pg):
        pg = init_pg(rank, world_size)
        first_dim = input1.shape[0]
        other_dims = input1.shape[1:]
        input1 = torch_npu.npu_format_cast(input1.repeat(2, *[1 for i in other_dims]).npu(), 29)[first_dim:]

        test_case = TestCase()
        error_expect = "For a tensor of internal format, it's storage_offset must be 0"
        with test_case.assertRaisesRegex(RuntimeError, error_expect):
            pg.broadcast(input1, 0)

    def _test_multiprocess_with_error(self, fn, init_pg, input1, world_size):
        ctx = mp.get_context('spawn')
        ps = []
        for i in range(world_size):
            p = ctx.Process(
                target=fn,
                args=(i, input1, world_size, init_pg))
            p.start()
            ps.append(p)

        for p in ps:
            p.join()
            self.assertEqual(p.exitcode, 0, "subprocess exit with abnormal code.")

    @skipIfUnsupportMultiNPU(2)
    def test_broadcast_with_internal_format_and_offset(self):
        ranks = [2]
        shape_format = [[np.float32, 2, [32, 32]]]

        for world_size in ranks:
            for shape in shape_format:
                _, npu_input = create_common_tensor(shape, -10, 10)
                self._test_multiprocess_with_error(HcclBroadcastTest._test_broadcast_with_internal_format_and_offset,
                                                   HcclBroadcastTest._init_dist_hccl, npu_input.cpu(), world_size)


if __name__ == '__main__':
    run_tests()
