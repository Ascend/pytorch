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


class HcomAllReduceTest(TestCase):
    @classmethod
    def _init_dist_hccl(cls, rank, world_size):
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = '29500'
        os.environ['HCCL_WHITELIST_DISABLE'] = '1'
        torch_npu.npu.set_device(rank)
        dist.init_process_group(backend='hccl', world_size=world_size, rank=rank)
        return dist

    @classmethod
    def _test_all_reduce(cls, rank, input1, world_size, init_pg, c2p):
        dist_group = init_pg(rank, world_size)
        dst = 0
        input1 = input1.npu()
        dist_group.all_reduce(input1)
        c2p.put((rank, dst, input1.cpu()))

    def _test_multiprocess(self, f, init_pg, expected, input1, world_size):
        ctx = mp.get_context('spawn')
        c2p = ctx.Queue(world_size)

        ps = []
        for i in range(world_size):
            p = ctx.Process(
                target=f,
                args=(i, input1.cpu(), world_size, init_pg, c2p))
            p.start()
            ps.append(p)

        for _ in range(world_size):
            rank, dst, output = c2p.get()
            if rank == dst:
                self.assertEqual(output, expected,
                                 ("rank {} Expect receive tensor {} but got {}.").format(rank, expected, output))

        for p in ps:
            p.join()

    @skipIfUnsupportMultiNPU(2)
    def test_dist_all_reduce(self):
        # CI currently supports only 2 devices
        ranks = [2]
        dtype_list = [np.float32, np.int32]
        format_list = [0, 2, 3]
        shape_format = [
            [i, j, [2, 3, 16]] for i in dtype_list for j in format_list
        ]
        for world_size in ranks:
            for shape in shape_format:
                exp_input, input1 = create_common_tensor(shape, -10, 10)
                expected = 0
                for _ in range(world_size):
                    expected += exp_input
                self._test_multiprocess(HcomAllReduceTest._test_all_reduce,
                                        HcomAllReduceTest._init_dist_hccl, expected, input1, world_size)

    @skipIfUnsupportMultiNPU(2)
    def test_dist_all_reduce_int64(self):
        # CI currently supports only 2 devices
        ranks = [2]
        shape_format = [np.int64, 2, [1]]
        for world_size in ranks:
            exp_input, input1 = create_common_tensor(shape_format, -10, 10)
            expected = 0
            for _ in range(world_size):
                expected += exp_input
            self._test_multiprocess(HcomAllReduceTest._test_all_reduce,
                                    HcomAllReduceTest._init_dist_hccl, expected, input1, world_size)

    @skipIfUnsupportMultiNPU(2)
    def test_dist_all_reduce_uint8(self):
        ranks = [2]
        shape_format = [[np.uint8, 2, [2, 3, 16]], [np.uint8, 2, [1]]]
        for world_size in ranks:
            for shape in shape_format:
                if len(shape[2]) == 1:
                    continue
                exp_input, input1 = create_common_tensor(shape, 0, 10)
                expected = 0
                for _ in range(world_size):
                    expected += exp_input
                self._test_multiprocess(HcomAllReduceTest._test_all_reduce,
                                        HcomAllReduceTest._init_dist_hccl, expected, input1, world_size)


if __name__ == '__main__':
    run_tests()
