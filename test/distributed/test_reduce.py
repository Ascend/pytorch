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


class HcclReduceTest(TestCase):

    @classmethod
    def _init_dist_hccl(cls, rank, world_size):
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = '29500'
        os.environ['HCCL_WHITELIST_DISABLE'] = '1'
        torch_npu.npu.set_device(rank)
        dist.init_process_group(backend='hccl', world_size=world_size, rank=rank)
        return dist

    @classmethod
    def _test_reduce(cls, rank, input1, world_size, init_pg, c2p, p2c):
        pg = init_pg(rank, world_size)
        dst = 0
        input1 = input1.npu()
        pg.reduce(input1, dst)
        c2p.put((rank, dst, input1.cpu()))
        p2c.get()

    def _test_multiprocess(self, f, init_pg, expected, input1, world_size):
        ctx = mp.get_context('spawn')
        c2p = ctx.Queue(world_size)
        p2c = ctx.Queue(world_size)
        ps = []

        for i in range(world_size):
            p = ctx.Process(
                target=f,
                args=(i, input1.cpu(), world_size, init_pg, c2p, p2c))
            p.start()
            ps.append(p)

        for _ in range(world_size):
            rank, dst, output = c2p.get()
            if rank == dst:
                self.assertEqual(output, expected,
                                 ("rank {} Expect receive tensor {} but got {}.").format(rank, expected, output))

        for _ in range(world_size):
            p2c.put(0)

        for p in ps:
            p.join()

    @skipIfUnsupportMultiNPU(2)
    def test_reduce_dist(self):
        ranks = [2, 4, 8]
        dtype_list = [np.float32, np.float16, np.int32, np.int8]
        format_list = [0, 2, 3, 29]
        shape_format = [
            [i, j, [12, 56, 256]] for i in dtype_list for j in format_list
        ]
        for world_size in ranks:
            if torch.npu.device_count() < world_size:
                continue
            for shape in shape_format:
                if shape[0] == np.int8:
                    shape[1] = 0
                exp_input, input1 = create_common_tensor(shape, -10, 10)
                expected = 0
                for _ in range(world_size):
                    expected += exp_input
                self._test_multiprocess(HcclReduceTest._test_reduce,
                                        HcclReduceTest._init_dist_hccl, expected, input1, world_size)

    @skipIfUnsupportMultiNPU(2)
    def test_reduce_int64_dist(self):
        ranks = [2, 4, 8]
        dtype_list = [np.int64]
        format_list = [0, 2]
        shape_format = [
            [i, j, [12, 56, 256]] for i in dtype_list for j in format_list
        ]
        for world_size in ranks:
            if torch.npu.device_count() < world_size:
                continue
            for shape in shape_format:
                exp_input, input1 = create_common_tensor(shape, -10, 10)
                expected = 0
                for _ in range(world_size):
                    expected += exp_input
                self._test_multiprocess(HcclReduceTest._test_reduce,
                                        HcclReduceTest._init_dist_hccl, expected, input1, world_size)

    @skipIfUnsupportMultiNPU(2)
    def test_reduce_uint8_dist(self):
        ranks = [2]
        dtype_list = [np.uint8]
        format_list = [0, 2]
        shape_format = [
            [i, j, [12, 56, 256]] for i in dtype_list for j in format_list
        ] + [[i, j, [1]] for i in dtype_list for j in format_list]
        for world_size in ranks:
            for shape in shape_format:
                if len(shape[2]) == 1:
                    continue
                exp_input, input1 = create_common_tensor(shape, -10, 10)
                expected = 0
                for _ in range(world_size):
                    expected += exp_input
                self._test_multiprocess(HcclReduceTest._test_reduce,
                                        HcclReduceTest._init_dist_hccl, expected, input1, world_size)


if __name__ == '__main__':
    run_tests()
