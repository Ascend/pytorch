# Copyright (c) 2023, Huawei Technologies.All rights reserved.
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
import torch_npu
import torch.distributed as dist
import torch.multiprocessing as mp

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor
from torch_npu.testing.common_distributed import skipIfUnsupportMultiNPU


class HcomAllGatherTest(TestCase):
    @classmethod
    def _init_dist_hccl(cls, rank, world_size):
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = '29500'
        os.environ['HCCL_WHITELIST_DISABLE'] = '1'
        torch_npu.npu.set_device(rank)
        dist.init_process_group(backend='hccl', world_size=world_size, rank=rank)
        return dist

    @classmethod
    def _test_all_gather(cls, rank, input1, world_size, init_pg):
        dist_group = init_pg(rank, world_size)
        input1_npu = input1.npu()

        outputs = []
        for _ in range(world_size):
            outputs.append(
                torch.empty(*input1_npu.shape, dtype=input1_npu.dtype, device=input1_npu.device)
            )

        torch.npu.enable_graph_mode()
        dist_group.all_gather(outputs, input1_npu)
        torch.npu.disable_graph_mode()

        for i in range(world_size):
            max_diff = abs(outputs[i].cpu() - input1).max()
            if max_diff >= 0.001:
                raise ValueError("max_diff >= 0.001")

    def _test_multiprocess(self, f, init_pg, input1, world_size):
        ctx = mp.get_context('spawn')
        ps = []
        for i in range(world_size):
            p = ctx.Process(
                target=f,
                args=(i, input1.cpu(), world_size, init_pg))
            p.start()
            ps.append(p)

        for p in ps:
            p.join()

    @skipIfUnsupportMultiNPU(2)
    def test_dist_all_gather(self):
        ranks = [4, 8]
        dtype_list = [np.float32, np.int32]
        format_list = [0, 2, 3]
        shape_format = [
            [i, j, [2, 3, 16]] for i in dtype_list for j in format_list
        ]
        for world_size in ranks:
            for shape in shape_format:
                _, input1 = create_common_tensor(shape, -10, 10)
                self._test_multiprocess(HcomAllGatherTest._test_all_gather,
                                        HcomAllGatherTest._init_dist_hccl, input1, world_size)


if __name__ == '__main__':
    run_tests()
