# Copyright (c) 2022, Huawei Technologies.All rights reserved.
#
# Licensed under the BSD 3-Clause License  (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
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

from torch_npu.testing.testcase import TestCase, run_tests
import torch_npu

os.environ["PYTORCH_NPU_ALLOC_CONF"] = "garbage_collection_threshold:0.1"


class TestGC(TestCase):

    def test_gc(self):

        def create_free_block():
            blocks = []
            for _ in range(100):
                x = torch.randn([10, 800, 1200, 3], device='npu:0')
                blocks.append(x)

        torch.npu.set_per_process_memory_fraction(1.0)

        create_free_block()
        x = torch.randn([50, 800, 1200, 3], device='npu:0')

        size = x.numel() * x.element_size()
        object_mem = 2097152 * int(((size + 2097152 - 1) / 2097152))
        res_mem = torch.npu.memory_reserved()
        self.assertEqual(object_mem, res_mem)


if __name__ == '__main__':
    run_tests()
