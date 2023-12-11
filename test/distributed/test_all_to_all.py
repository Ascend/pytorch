# Copyright (c) 2020, Huawei Technologies.All rights reserved.
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

import os

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_distributed import skipIfUnsupportMultiNPU


class HcclAlltoAllTest(TestCase):
    world_size_2p = 2
    world_size_4p = 4
    data = torch.randn(10, 20)

    @classmethod
    def _init_dist_hccl(cls, rank, world_size):
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = '29500'
        os.environ['HCCL_WHITELIST_DISABLE'] = '1'
        os.environ['HCCL_ALGO'] = "level0:fullmesh;level1:fullmesh"
        torch_npu.npu.set_device(rank)
        dist.init_process_group(backend='hccl', world_size=world_size, rank=rank)
        return dist

    @classmethod
    def _test_alltoall_2p(
            cls, rank, data, world_size, init_pg, c2p, p2c):

        pg = init_pg(rank, world_size)
        input1 = torch.arange(2) + rank * 2
        input1 = input1.float().npu()
        input1_list = list(input1.chunk(2))
        output = torch.empty(2).float().npu()
        output_list = list(output.chunk(2))
        cout = 0
        outputdebug = pg.all_to_all(output_list, input1_list)
        c2p.put((rank, [tensor.cpu() for tensor in output_list], cout))

    @classmethod
    def _test_alltoall_2p_size(
            cls, rank, data, world_size, init_pg, c2p, p2c):
        pg = init_pg(rank, world_size)
        input1 = data.float().npu()
        input1 = torch_npu.npu_format_cast(input1, 29)
        output = torch.empty(200, 1).float().npu()
        output = torch_npu.npu_format_cast(output, 29)
        inputsize = [[100, 100], [100, 100]]
        outsize = [[100, 100], [100, 100]]

        input1 = input1.view(-1)
        input1_list = list(input1.split(inputsize[rank]))
        output_list = list(output.split(outsize[rank]))
        cout = 1
        pg.all_to_all(output_list, input1_list)
        if torch_npu.get_npu_format(output.npu()) != 29:
            raise RuntimeError("format error!")
        c2p.put((rank, [tensor.cpu() for tensor in output_list], cout))

    def _test_multiprocess_2p(self, f, init_pg):
        ws = self.world_size_2p
        data = self.data
        # file store will delete the test file on destruction
        ctx = mp.get_context('spawn')
        c2p = ctx.Queue(2)
        p2c = ctx.Queue(2)
        expected = []
        ps = []
        for i in range(ws):
            p = ctx.Process(
                target=f,
                args=(i, data, ws, init_pg, c2p, p2c))
            p.start()
            ps.append(p)

        for _ in range(ws):
            rank, output, cout = c2p.get()
            res = data.cpu().float()

            exp = []
            if rank == 0:
                exp = res[0]
                for i in range(1, 5):
                    exp = torch.cat((exp, res[i]), dim=0)
            else:
                exp = res[5]
                for i in range(6, 10):
                    exp = torch.cat((exp, res[i]), dim=0)

            exp_2p = torch.arange(2) * 2 + rank
            exp_format = torch.cat((exp, exp), dim=0).reshape(200, 1)
            expected = exp_2p if cout == 0 else exp_format
            if cout == 0:
                expected = list(exp_2p.chunk(2))
            else:
                expected = list(expected.split([100, 100]))

            self.assertEqual(
                output,
                expected,
                (
                    "rank {} Expect receive tensor {} but got {}."
                ).format(rank, expected, output)
            )

        for _ in range(ws):
            p2c.put(0)

        for p in ps:
            p.join(2)

    @skipIfUnsupportMultiNPU(2)
    def test_alltoall_2p_dist(self):
        print('devicecount: ', torch.npu.device_count())
        self._test_multiprocess_2p(
            HcclAlltoAllTest._test_alltoall_2p,
            HcclAlltoAllTest._init_dist_hccl)
        print('test_alltoall_2p_dist ')

    @skipIfUnsupportMultiNPU(2)
    def test_alltoall_2p_size_dist(self):
        self._test_multiprocess_2p(
            HcclAlltoAllTest._test_alltoall_2p_size,
            HcclAlltoAllTest._init_dist_hccl)
        print('test_alltoall_2p_size_dist')

    @classmethod
    def _test_alltoall_4p(
            cls, rank, world_size, init_pg, c2p, p2c):
        pg = init_pg(rank, world_size)
        input1 = torch.arange(4) + rank * 4
        input1 = input1.float().npu()
        input1_list = list(input1.chunk(4))
        output = torch.empty(4).float().npu()
        output_list = list(output.chunk(4))
        cout = 0
        pg.all_to_all(output_list, input1_list)
        c2p.put((rank, [tensor.cpu() for tensor in output_list], cout, [1, 1, 1, 1]))

    @classmethod
    def _test_alltoall_4p_size(
            cls, rank, world_size, init_pg, c2p, p2c):
        pg = init_pg(rank, world_size)
        input1 = torch.arange(7) + rank * 4
        input1 = input1.float().npu()
        x = 7
        if rank == 1:
            x = 9
        elif rank == 3:
            x = 5
        output = torch.empty(x).float().npu()
        inputsize = [[1, 2, 2, 2], [1, 3, 2, 1], [2, 3, 1, 1], [3, 1, 2, 1]]
        outsize = [[1, 1, 2, 3], [2, 3, 3, 1], [2, 2, 1, 2], [2, 1, 1, 1]]

        input1_list = list(input1.split(inputsize[rank]))
        output_list = list(output.split(outsize[rank]))
        cout = 1
        pg.all_to_all(output_list, input1_list)
        c2p.put((rank, [tensor.cpu() for tensor in output_list], cout, outsize[rank]))

    def _test_multiprocess_4p(self, f, init_pg):
        ws = self.world_size_4p
        # file store will delete the test file on destruction
        ctx = mp.get_context('spawn')
        c2p = ctx.Queue(4)
        p2c = ctx.Queue(4)
        ps = []
        expected = []
        for i in range(ws):
            p = ctx.Process(
                target=f,
                args=(i, ws, init_pg, c2p, p2c))
            p.start()
            ps.append(p)

        for _ in range(ws):
            rank, output, cout, outsize = c2p.get()
            if cout == 0:
                expected = torch.arange(4) * 4 + rank
            else:
                expected_dict = {
                    0: torch.tensor([0, 4, 8, 9, 12, 13, 14]),
                    1: torch.tensor([1, 2, 5, 6, 7, 10, 11, 12, 15]),
                    2: torch.tensor([3, 4, 8, 9, 13, 16, 17]),
                    3: torch.tensor([5, 6, 10, 14, 18])
                }
                expected = expected_dict.get(rank)

            expected = list(expected.split(outsize))

            self.assertEqual(
                output,
                expected,
                (
                    "rank {} Expect receive tensor {} but got {}."
                ).format(rank, expected, output)
            )

        for _ in range(ws):
            p2c.put(0)

        for p in ps:
            p.join(4)

    @skipIfUnsupportMultiNPU(4)
    def test_alltoall_4p_dist(self):
        self._test_multiprocess_4p(
            HcclAlltoAllTest._test_alltoall_4p,
            HcclAlltoAllTest._init_dist_hccl)
        print('test_alltoall_4p_dist')

    @skipIfUnsupportMultiNPU(4)
    def test_alltoall_4p_size_dist(self):
        self._test_multiprocess_4p(
            HcclAlltoAllTest._test_alltoall_4p_size,
            HcclAlltoAllTest._init_dist_hccl)
        print('test_alltoall_4p_size_dist')


if __name__ == '__main__':
    run_tests()
