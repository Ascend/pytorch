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
    # pylint:disable=huawei-too-many-arguments
    def _test_reduce(cls, rank, input1, world_size, init_pg, c2p, p2c, reduce_op=dist.ReduceOp.SUM):
        pg = init_pg(rank, world_size)
        dst = 0
        input1 = input1.npu()
        pg.reduce(input1, dst, reduce_op)
        c2p.put((rank, dst, input1.cpu()))
        p2c.get()

    # pylint:disable=huawei-too-many-arguments
    def _test_multiprocess(self, f, init_pg, expected, input1, world_size, reduce_op=dist.ReduceOp.SUM):
        ctx = mp.get_context('spawn')
        c2p = ctx.Queue(world_size)
        p2c = ctx.Queue(world_size)
        ps = []

        for i in range(world_size):
            p = ctx.Process(
                target=f,
                args=(i, input1.cpu(), world_size, init_pg, c2p, p2c, reduce_op))
            p.start()
            ps.append(p)

        for _ in range(world_size):
            rank, dst, output = c2p.get()
            if rank == dst:
                self.assertEqual(output, expected,
                                 "rank {} world_size {} dtype {} shape {} Expect receive tensor {} but got {}.".format(
                                     rank, world_size, expected.dtype, expected.shape, expected, output))

        for _ in range(world_size):
            p2c.put(0)

        for p in ps:
            p.join()

    def _construct_excepted_result(self, inputs, world_size, dtype=np.float32, reduce_op=dist.ReduceOp.SUM):
        expected = 0
        for _ in range(world_size):
            expected += inputs

        if reduce_op == dist.ReduceOp.AVG:
            if dtype in [np.int32, np.int8, np.int64, np.uint8]:
                expected //= world_size
            else:
                expected /= world_size

        return expected

    @skipIfUnsupportMultiNPU(2)
    def test_reduce_dist(self):
        ranks = [2, 4, 8]
        dtype_list = [np.float32, np.float16]
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
                expected = self._construct_excepted_result(exp_input, world_size)
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
                expected = self._construct_excepted_result(exp_input, world_size)
                self._test_multiprocess(HcclReduceTest._test_reduce,
                                        HcclReduceTest._init_dist_hccl, expected, input1, world_size)

    @skipIfUnsupportMultiNPU(2)
    def test_reduce_dist_avg(self):
        ranks = [2]
        dtype_list = [np.int32, np.int8, np.int64]
        shape_format = [[i, 2, [3, 16]] for i in dtype_list]
        for world_size in ranks:
            if torch.npu.device_count() < world_size:
                continue
            for shape in shape_format:
                if shape[0] == np.int8:
                    shape[1] = 0
                exp_input, input1 = create_common_tensor(shape, -10, 10)
                expected = self._construct_excepted_result(exp_input, world_size, shape[0], dist.ReduceOp.AVG)
                self._test_multiprocess(HcclReduceTest._test_reduce,
                                        HcclReduceTest._init_dist_hccl, expected, input1, world_size, dist.ReduceOp.AVG)


if __name__ == '__main__':
    run_tests()
