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

from test_reduce_scatter import HcclReduceScatterTestBase


class HcclReduceScatterTensorTest(HcclReduceScatterTestBase):

    @classmethod
    def _test_reduce_scatter_tensor(cls, rank, input_list, world_size, init_pg, c2p):
        pg = init_pg(rank, world_size)
        input_list_npu = [input.npu() for input in input_list]
        input_tensor = torch.cat(input_list_npu)
        output = torch.empty_like(input_list_npu[rank])
        pg.reduce_scatter_tensor(output, input_tensor)
        c2p.put((rank, output.cpu()))
        pg.barrier()

    @skipIfUnsupportMultiNPU(2)
    def test_reduce_scatter_tensor(self):
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
                input_list = []
                for _ in range(world_size):
                    _, input1 = create_common_tensor(shape, -10, -10)
                    input_list.append(input1.cpu())
                expected = self._construct_excepted_result(input_list, world_size, dist.reduce_scatter_tensor)
                self._test_multiprocess(HcclReduceScatterTensorTest._test_reduce_scatter_tensor,
                                        HcclReduceScatterTensorTest._init_dist_hccl, expected, input_list, world_size)

    @classmethod
    def _test_reduce_scatter_tensor_uneven(cls, rank, input_list, world_size, init_pg, c2p):
        init_pg(rank, world_size)
        input_list_npu = [input.npu() for input in input_list]
        input_tensor = torch.cat(input_list_npu)
        output = torch.empty_like(input_list_npu[rank])
        torch_npu.distributed.reduce_scatter_tensor_uneven(output, input_tensor)
        c2p.put((rank, output.cpu()))
        dist.barrier()

    @skipIfUnsupportMultiNPU(2)
    def test_reduce_scatter_tensor_uneven(self):
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
                input_list = []
                for _ in range(world_size):
                    _, input1 = create_common_tensor(shape, -10, -10)
                    input_list.append(input1.cpu())
                expected = self._construct_excepted_result(input_list, world_size, torch_npu.distributed.reduce_scatter_tensor_uneven)
                self._test_multiprocess(HcclReduceScatterTensorTest._test_reduce_scatter_tensor_uneven,
                                        HcclReduceScatterTensorTest._init_dist_hccl, expected, input_list, world_size)


if __name__ == '__main__':
    run_tests()
