# Copyright (c) 2022, Huawei Technologies.All rights reserved.
#
# Licensed under the BSD 3-Clause License  (the "License");
# you may not use this file except in compliance with the License.
# You may obtain data copy of the License at
#
# https://opensource.org/licenses/BSD-3-Clause
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import unittest
import os

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor
from torch_npu.testing.common_distributed import skipIfUnsupportMultiNPU
import torch_npu


class HcclReduceScatterTest(TestCase):

    @classmethod
    def _init_dist_hccl(cls, rank, world_size):
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = '29500'
        os.environ['HCCL_WHITELIST_DISABLE'] = '1'
        torch_npu.npu.set_device(rank)
        dist.init_process_group(backend='hccl', world_size=world_size, rank=rank)
        return dist

    @classmethod
    def _test_reduce_scatter(cls, rank, input_list, world_size, init_pg, c2p):
        pg = init_pg(rank, world_size)
        input_list_npu = [input.npu() for input in input_list]
        output = torch.empty_like(input_list_npu[rank])
        pg.reduce_scatter(output, input_list_npu)
        c2p.put((rank, output.cpu()))
        pg.barrier()

    @classmethod
    def _test_reduce_scatter_tensor(cls, rank, input_list, world_size, init_pg, c2p):
        pg = init_pg(rank, world_size)
        input_list_npu = [input.npu() for input in input_list]
        input_tensor = torch.cat(input_list_npu)
        output = torch.empty_like(input_list_npu[rank])
        pg.reduce_scatter_tensor(output, input_tensor)
        c2p.put((rank, output.cpu()))
        pg.barrier()

    @classmethod
    def _test_reduce_scatter_base(cls, rank, input_list, world_size, init_pg, c2p):
        pg = init_pg(rank, world_size)
        input_list_npu = [input.npu() for input in input_list]
        input_tensor = torch.cat(input_list_npu)
        output = torch.empty_like(input_list_npu[rank])
        pg._reduce_scatter_base(output, input_tensor)
        c2p.put((rank, output.cpu()))
        pg.barrier()

    def _test_multiprocess(self, fn, init_pg, expected, input1, world_size):
        ctx = mp.get_context('spawn')
        c2p = ctx.Queue(world_size)
        ps = []
        for i in range(world_size):
            p = ctx.Process(
                target=fn,
                args=(i, input1, world_size, init_pg, c2p))
            p.start()
            ps.append(p)
        for _ in range(world_size):
            rank, output = c2p.get()
            self.assertEqual(output, expected[rank],
                             ("rank {} Expect receive tensor {} but got {}.").format(rank, expected[rank], output))
        for p in ps:
            p.join()

    def _construct_excepted_result(self, inputs, world_size, op=dist.all_gather):
        if op not in [dist.reduce_scatter, dist._reduce_scatter_base, dist.reduce_scatter_tensor]:
            raise ValueError("Unsupported op `{}`" % (str(op)))
        return [input.cpu() * world_size for input in inputs]

    @skipIfUnsupportMultiNPU(2)
    def test_reduce_scatter(self):
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
                expected = self._construct_excepted_result(input_list, world_size, dist.reduce_scatter)
                self._test_multiprocess(HcclReduceScatterTest._test_reduce_scatter,
                                        HcclReduceScatterTest._init_dist_hccl, expected, input_list, world_size)

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
                expected = self._construct_excepted_result(input_list, world_size, dist._reduce_scatter_tensor)
                self._test_multiprocess(HcclReduceScatterTest._test_reduce_scatter_tensor,
                                        HcclReduceScatterTest._init_dist_hccl, expected, input_list, world_size)

    @skipIfUnsupportMultiNPU(2)
    def test_reduce_scatter_base(self):
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
                expected = self._construct_excepted_result(input_list, world_size, dist._reduce_scatter_base)
                self._test_multiprocess(HcclReduceScatterTest._test_reduce_scatter_base,
                                        HcclReduceScatterTest._init_dist_hccl, expected, input_list, world_size)

    @skipIfUnsupportMultiNPU(2)
    def test_reduce_scatter_base_int64(self):
        ranks = [2]
        dtype_list = [np.int64]
        format_list = [0, 2]
        shape_format = [
            [i, j, [4, 9]] for i in dtype_list for j in format_list] + \
            [[i, j, [8]] for i in dtype_list for j in format_list]
        for world_size in ranks:
            for shape in shape_format:
                input_list = []
                for _ in range(world_size):
                    _, input1 = create_common_tensor(shape, -10, -10)
                    input_list.append(input1.cpu())
                expected = self._construct_excepted_result(input_list, world_size, dist._reduce_scatter_base)
                self._test_multiprocess(HcclReduceScatterTest._test_reduce_scatter_base,
                                        HcclReduceScatterTest._init_dist_hccl, expected, input_list, world_size)


if __name__ == '__main__':
    run_tests()
