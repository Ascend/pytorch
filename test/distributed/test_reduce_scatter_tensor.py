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
    # pylint:disable=huawei-too-many-arguments
    def _test_reduce_scatter_tensor(cls, rank, input_list, world_size, init_pg, c2p, p2c, reduce_op=dist.ReduceOp.SUM):
        pg = init_pg(rank, world_size)
        input_list_npu = [input.npu() for input in input_list]
        input_tensor = torch.cat(input_list_npu)
        output = torch.empty_like(input_list_npu[rank])
        pg.reduce_scatter_tensor(output, input_tensor, reduce_op)
        c2p.put((rank, output.cpu()))
        pg.barrier()
        p2c.get()

    @classmethod
    # pylint:disable=huawei-too-many-arguments
    def _test_reduce_scatter_tensor_with_input_internal_format_and_offset(cls, rank, input_list, world_size, init_pg):
        torch_npu.npu.config.allow_internal_format = True
        pg = init_pg(rank, world_size)
        input_list_npu = [input.npu() for input in input_list]
        input_tensor = torch.cat(input_list_npu)
        first_dim = input_tensor.shape[0]
        other_dims = input_tensor.shape[1:]
        input_tensor = torch_npu.npu_format_cast(input_tensor.repeat(2, *[1 for i in other_dims]), 29)[first_dim:]
        output = torch.empty_like(input_list_npu[rank])
        test_case = TestCase()
        error_expect = "For a tensor of internal format, it's storage_offset must be 0"
        with test_case.assertRaisesRegex(RuntimeError, error_expect):
            pg.reduce_scatter_tensor(output, input_tensor)

    @classmethod
    # pylint:disable=huawei-too-many-arguments
    def _test_reduce_scatter_tensor_with_output_internal_format_and_offset(cls, rank, input_list, world_size, init_pg):
        torch_npu.npu.config.allow_internal_format = True
        pg = init_pg(rank, world_size)
        input_list_npu = [input.npu() for input in input_list]
        input_tensor = torch.cat(input_list_npu)
        output = torch.empty_like(input_list_npu[rank])
        first_dim = output.shape[0]
        other_dims = output.shape[1:]
        output = torch_npu.npu_format_cast(output.repeat(2, *[1 for i in other_dims]), 29)[first_dim:]
        test_case = TestCase()
        error_expect = "For a tensor of internal format, it's storage_offset must be 0"
        with test_case.assertRaisesRegex(RuntimeError, error_expect):
            pg.reduce_scatter_tensor(output, input_tensor)

    @skipIfUnsupportMultiNPU(2)
    def test_reduce_scatter_tensor(self):
        ranks = [2]
        dtype_list = [np.float32, np.float16]
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
                    _, input1 = create_common_tensor(shape, -10, 10)
                    input_list.append(input1.cpu())
                expected = self._construct_excepted_result(input_list, world_size, dist.reduce_scatter_tensor)
                self._test_multiprocess(HcclReduceScatterTensorTest._test_reduce_scatter_tensor,
                                        HcclReduceScatterTensorTest._init_dist_hccl, expected, input_list, world_size)

    @classmethod
    # pylint:disable=huawei-too-many-arguments
    def _test_reduce_scatter_tensor_uneven(cls, rank, input_list, world_size, init_pg, c2p, p2c, reduce_op=dist.ReduceOp.SUM):
        init_pg(rank, world_size)
        input_list_npu = [input.npu() for input in input_list]
        input_tensor = torch.cat(input_list_npu)
        output = torch.empty_like(input_list_npu[rank])
        torch_npu.distributed.reduce_scatter_tensor_uneven(output, input_tensor, reduce_op)
        c2p.put((rank, output.cpu()))
        dist.barrier()
        p2c.get()

    @skipIfUnsupportMultiNPU(2)
    def test_reduce_scatter_tensor_uneven(self):
        ranks = [2]
        dtype_list = [np.int32, np.int8]
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
                    _, input1 = create_common_tensor(shape, -10, 10)
                    input_list.append(input1.cpu())
                expected = self._construct_excepted_result(input_list, world_size, torch_npu.distributed.reduce_scatter_tensor_uneven)
                self._test_multiprocess(HcclReduceScatterTensorTest._test_reduce_scatter_tensor_uneven,
                                        HcclReduceScatterTensorTest._init_dist_hccl, expected, input_list, world_size)

    @skipIfUnsupportMultiNPU(2)
    def test_reduce_scatter_tensor_avg(self):
        ranks = [2]
        dtype_list = [np.int32, np.int8]
        shape_format = [[i, 2, [4, 9]] for i in dtype_list]
        for world_size in ranks:
            for shape in shape_format:
                if shape[0] == np.int8:
                    shape[1] = 0
                input_list = []
                for _ in range(world_size):
                    _, input1 = create_common_tensor(shape, -10, 10)
                    input_list.append(input1.cpu())
                expected = self._construct_excepted_result(input_list, world_size, dist.reduce_scatter_tensor, dist.ReduceOp.AVG)
                self._test_multiprocess(HcclReduceScatterTensorTest._test_reduce_scatter_tensor,
                                        HcclReduceScatterTensorTest._init_dist_hccl, expected, input_list, world_size, dist.ReduceOp.AVG)

    @skipIfUnsupportMultiNPU(2)
    def test_reduce_scatter_tensor_with_input_internal_format_and_offset(self):
        ranks = [2]
        shape_format = [[np.float32, 2, [31, 31]]]
        for world_size in ranks:
            for shape in shape_format:
                input_list = []
                for _ in range(world_size):
                    _, input1 = create_common_tensor(shape, -10, 10)
                    input_list.append(input1.cpu())
                self._test_multiprocess_with_error(HcclReduceScatterTensorTest._test_reduce_scatter_tensor_with_input_internal_format_and_offset,
                                                   HcclReduceScatterTensorTest._init_dist_hccl, input_list, world_size)

    @skipIfUnsupportMultiNPU(2)
    def test_reduce_scatter_tensor_with_output_internal_format_and_offset(self):
        ranks = [2]
        shape_format = [[np.float32, 2, [31, 31]]]
        for world_size in ranks:
            for shape in shape_format:
                input_list = []
                for _ in range(world_size):
                    _, input1 = create_common_tensor(shape, -10, 10)
                    input_list.append(input1.cpu())
                self._test_multiprocess_with_error(HcclReduceScatterTensorTest._test_reduce_scatter_tensor_with_output_internal_format_and_offset,
                                                   HcclReduceScatterTensorTest._init_dist_hccl, input_list, world_size)

    @skipIfUnsupportMultiNPU(2)
    def test_reduce_scatter_tensor_uneven_avg(self):
        ranks = [2]
        dtype_list = [np.float32, np.float16]
        shape_format = [[i, 2, [4, 9]] for i in dtype_list]
        for world_size in ranks:
            for shape in shape_format:
                if shape[0] == np.int8:
                    shape[1] = 0
                input_list = []
                for _ in range(world_size):
                    _, input1 = create_common_tensor(shape, -10, 10)
                    input_list.append(input1.cpu())
                expected = self._construct_excepted_result(input_list, world_size, dist.reduce_scatter_tensor_uneven, dist.ReduceOp.AVG)
                self._test_multiprocess(HcclReduceScatterTensorTest._test_reduce_scatter_tensor_uneven,
                                        HcclReduceScatterTensorTest._init_dist_hccl, expected, input_list, world_size, dist.ReduceOp.AVG)

    @skipIfUnsupportMultiNPU(2)
    def test_reduce_scatter_tensor_pre_mul(self):
        ranks = [2]
        dtype_list = [np.float32]
        shape_format = [[i, 2, [20, 20]] for i in dtype_list]
        for world_size in ranks:
            for shape in shape_format:
                input_list = []
                for _ in range(world_size):
                    _, input1 = create_common_tensor(shape, -10, 10)
                    input_list.append(input1.cpu())
                expected = self._construct_excepted_result(input_list, world_size, dist.reduce_scatter_tensor, dist.ReduceOp.SUM)
                expected = [i * 2 for i in expected]
                reduce_op = torch.distributed._make_nccl_premul_sum(2.0)
                self._test_multiprocess(HcclReduceScatterTensorTest._test_reduce_scatter_tensor,
                                        HcclReduceScatterTensorTest._init_dist_hccl, expected, input_list, world_size, reduce_op)

if __name__ == '__main__':
    run_tests()
