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

import expecttest

import torch
import torch_npu
import torch_npu._C

import numpy as np

import torch_npu.npu.utils as utils

from torch_npu.testing.testcase import TestCase, run_tests

class TestCheckOverFlow(TestCase):

    def test_check_over_flow(self):
        soc_version = utils.get_soc_version()
        a = torch.Tensor([np.inf, np.inf]).npu()
        a = a + a
        if (soc_version < 220):
            rtn = utils.npu_check_over_flow(a)
            self.assertTrue(rtn == utils.get_npu_overflow_flag())
        else:
            rtn = utils.npu_check_over_flow(a)
            self.assertTrue(rtn)
    

if __name__ == "__main__":
    run_tests()