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

class HcclAlltoAllSingleTest(TestCase): 
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
    def _test_alltoall_single_2p(
            cls, rank, data, world_size, init_pg, c2p, p2c):
        pg = init_pg(rank, world_size)
        input1 = torch.arange(2) + rank * 2
        input1 = input1.float().npu()
        input1 = input1.npu_format_cast(29)
        output = torch.empty(2).float().npu()
        output = output.npu_format_cast(29)
        cout = 0
        pg.all_to_all_single(output, input1)
        c2p.put((rank, output.cpu(), cout))

    @classmethod 
    def _test_alltoall_single_2p_size(
            cls, rank, data, world_size, init_pg, c2p, p2c):
        pg = init_pg(rank, world_size)
        input1 = data.float().npu()
        input1 = input1.npu_format_cast(29)
        output = torch.empty(200).float().npu()
        output = output.npu_format_cast(29)
        inputsize1 = [100, 100]
        inputsize2 = [100, 100]
        outsize1 = [100, 100]
        outsize2 = [100, 100]
        inputsize = [inputsize1, inputsize2]
        outsize = [outsize1, outsize2]
        cout = 1
        pg.all_to_all_single(output, input1, outsize[rank], inputsize[rank])
        if torch.get_npu_format(output.npu()) != 29:
            raise RuntimeError("format error!")
        c2p.put((rank, output.cpu(), cout))

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
            exp_format = torch.cat((exp, exp), dim=0)
            expected = exp_2p if cout == 0 else exp_format
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

    def test_alltoall_single_2p_dist(self):
        self._test_multiprocess_2p(
            HcclAlltoAllSingleTest._test_alltoall_single_2p,
            HcclAlltoAllSingleTest._init_dist_hccl)

    def test_alltoall_single_2p_size_dist(self):
        self._test_multiprocess_2p(
            HcclAlltoAllSingleTest._test_alltoall_single_2p_size,
            HcclAlltoAllSingleTest._init_dist_hccl)

    @classmethod 
    def _test_alltoall_single_4p(
            cls, rank, world_size, init_pg, c2p, p2c):
        pg = init_pg(rank, world_size)
        input1 = torch.arange(4) + rank * 4
        input1 = input1.float().npu()
        output = torch.empty(4).float().npu()
        cout = 0
        pg.all_to_all_single(output, input1)
        c2p.put((rank, output.cpu(), cout))

    @classmethod 
    def _test_alltoall_single_4p_size(
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
        inputsize1 = [1, 2, 2, 2]
        inputsize2 = [1, 3, 2, 1]
        inputsize3 = [2, 3, 1, 1]
        inputsize4 = [3, 1, 2, 1]
        outsize1 = [1, 1, 2, 3]
        outsize2 = [2, 3, 3, 1]
        outsize3 = [2, 2, 1, 2]
        outsize4 = [2, 1, 1, 1]
        inputsize = [inputsize1, inputsize2, inputsize3, inputsize4]
        outsize = [outsize1, outsize2, outsize3, outsize4]
        cout = 1
        pg.all_to_all_single(output, input1, outsize[rank], inputsize[rank])
        c2p.put((rank, output.cpu(), cout))

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
            rank, output, cout = c2p.get()
            if cout == 0:
                expected = torch.arange(4) * 4 + rank
            else:
                if rank == 0:
                    expected = [0, 4, 8, 9, 12, 13, 14] 
                elif rank == 1:
                    expected = [1, 2, 5, 6, 7, 10, 11, 12, 15]
                elif rank == 2:
                    expected = [3, 4, 8, 9, 13, 16, 17]  
                elif rank == 3:
                    expected = [5, 6, 10, 14, 18]     

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

    def test_alltoall_single_4p_dist(self):
        self._test_multiprocess_4p(
            HcclAlltoAllSingleTest._test_alltoall_single_4p,
            HcclAlltoAllSingleTest._init_dist_hccl)

    def test_alltoall_single_4p_size_dist(self):
        self._test_multiprocess_4p(
            HcclAlltoAllSingleTest._test_alltoall_single_4p_size,
            HcclAlltoAllSingleTest._init_dist_hccl)

if __name__ == '__main__':
    run_tests()
