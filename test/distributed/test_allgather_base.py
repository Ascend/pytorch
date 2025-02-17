import unittest
import os
from random import randint

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor
from torch_npu.testing.common_distributed import skipIfUnsupportMultiNPU

from test_allgather import HcclAllGatherTestBase


class HcclAllGatherBaseTest(HcclAllGatherTestBase):

    @classmethod
    def _test_all_gather_base(cls, rank, input1, world_size, init_pg, c2p, p2c):
        pg = init_pg(rank, world_size)
        input1 = input1.npu()
        shape = list(input1.size())
        shape[0] = shape[0] * world_size
        gather_tensor = torch.empty(shape, device=input1.device, dtype=input1.dtype)
        pg._all_gather_base(gather_tensor, input1)
        c2p.put((rank, gather_tensor.cpu()))
        pg.barrier()
        p2c.get()

    @skipIfUnsupportMultiNPU(2)
    def test_all_gather_base_dist(self):
        ranks = [2]
        dtype_list = [np.float32, np.float16, np.int32, np.int8]
        format_list = [0, 2, 3, 29]
        shape_format = [
            [i, j, [4, 9]] for i in dtype_list for j in format_list] + \
            [[i, j, [8]] for i in dtype_list for j in format_list]
        for world_size in ranks:
            for shape in shape_format:
                if shape[0] == np.int8:
                    shape[1] = 0
                _, input1 = create_common_tensor(shape, -10, 10)
                expected = self._construct_excepted_result(input1, world_size, dist._all_gather_base)
                self._test_multiprocess(HcclAllGatherBaseTest._test_all_gather_base,
                                        HcclAllGatherBaseTest._init_dist_hccl, expected, input1, world_size)


if __name__ == '__main__':
    run_tests()
