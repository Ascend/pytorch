# Copyright (c) 2020, Huawei Technologies.All rights reserved.
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

import torch
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests


class GeneratorTest(TestCase):

    def test_state(self):
        gen = torch_npu._C.Generator()
        gen.set_state(torch.get_rng_state())
        self.assertEqual(gen.get_state(), torch.get_rng_state())

    def test_seed(self):
        gen = torch_npu._C.Generator(torch_npu.npu.native_device + ":" + str(torch_npu.npu.current_device()))
        gen.manual_seed(1234)
        self.assertEqual(gen.initial_seed(), 1234)


if __name__ == '__main__':
    run_tests()