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


class HcclScatterTest(TestCase):

    @classmethod
    def _init_dist_hccl(cls, rank, world_size):
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = '29500'
        os.environ['HCCL_WHITELIST_DISABLE'] = '1'
        torch_npu.npu.set_device(rank)
        dist.init_process_group(backend='hccl', world_size=world_size, rank=rank)
        return dist

    @classmethod
    def _test_scatter(cls, rank, input_list, world_size, init_pg, c2p, p2c):
        pg = init_pg(rank, world_size)
        input_list_npu = [input.npu() for input in input_list]
        output = torch.empty_like(input_list_npu[rank])
        if rank != 0:
            input_list_npu = []
        pg.scatter(output, input_list_npu)
        c2p.put((rank, output.cpu()))
        pg.barrier()
        p2c.get()

    def _test_multiprocess(self, fn, init_pg, expected, input1, world_size):
        ctx = mp.get_context('spawn')
        c2p = ctx.Queue(world_size)
        p2c = ctx.Queue(world_size)
        ps = []
        for i in range(world_size):
            p = ctx.Process(
                target=fn,
                args=(i, input1, world_size, init_pg, c2p, p2c))
            p.start()
            ps.append(p)
        for _ in range(world_size):
            rank, output = c2p.get()
            self.assertEqual(output, expected[rank],
                             ("rank {} Expect receive tensor {} but got {}.").format(rank, expected[rank], output))

        for _ in range(world_size):
            p2c.put(0)

        for p in ps:
            p.join()

    def _construct_expected_result(self, inputs, op=dist.scatter):
        if op not in [dist.scatter]:
            raise ValueError("Unsupported op `{}`" % (str(op)))
        return [input.cpu() for input in inputs]

    @skipIfUnsupportMultiNPU(2)
    def test_scatter(self):
        ranks = [2]
        dtype_list = [np.float32, np.float16, bool]
        format_list = [0]
        shape_format = [
            [i, j, [4, 9]] for i in dtype_list for j in format_list] + \
            [[i, j, [8]] for i in dtype_list for j in format_list]

        for world_size in ranks:
            for shape in shape_format:
                input_list = []
                for _ in range(world_size):
                    _, npu_input = create_common_tensor(shape, -10, 10)
                    input_list.append(npu_input.cpu())
                expected = self._construct_expected_result(input_list, dist.scatter)
                self._test_multiprocess(HcclScatterTest._test_scatter,
                                        HcclScatterTest._init_dist_hccl, expected, input_list, world_size)


if __name__ == '__main__':
    run_tests()
