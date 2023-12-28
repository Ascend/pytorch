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
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor
from torch_npu.testing.common_distributed import skipIfUnsupportMultiNPU

from test_allgather import HcclAllGatherTestBase


class HcclAllGatherToGatherTest(HcclAllGatherTestBase):

    @classmethod
    def _test_all_gather_togather(cls, rank, input1, world_size, init_pg, c2p):
        pg = init_pg(rank, world_size)

        input1 = input1.npu()
        gather_tensor = torch.empty((world_size, *list(input1.size())), device=input1.device, dtype=input1.dtype)
        pg.all_gather_togather(gather_tensor, input1)
        c2p.put((rank, gather_tensor.cpu()))
        pg.barrier()

    @skipIfUnsupportMultiNPU(2)
    def test_all_gather_togather_dist(self):
        ranks = [2]
        dtype_list = [np.float32, np.float16, np.int32, np.int8, np.bool]
        format_list = [0, 2, 3, 29]
        shape_format = [
            [i, j, [4, 9]] for i in dtype_list for j in format_list] + \
            [[i, j, [8]] for i in dtype_list for j in format_list]
        for world_size in ranks:
            for shape in shape_format:
                if shape[0] == np.int8:
                    shape[1] = 0
                if shape[0] == np.bool:
                    continue
                _, input1 = create_common_tensor(shape, -10, 10)
                expected = self._construct_excepted_result(input1, world_size, dist.all_gather_togather)
                self._test_multiprocess(HcclAllGatherToGatherTest._test_all_gather_togather,
                                        HcclAllGatherToGatherTest._init_dist_hccl, expected, input1, world_size)


if __name__ == '__main__':
    run_tests()
