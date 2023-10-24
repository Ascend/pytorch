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

import torch_npu.npu.utils as utils

from torch_npu.testing.testcase import TestCase, run_tests


class TestSocVersion(TestCase):

    def test_get_soc_version(self):
        soc_version = utils.get_soc_version()
        self.assertTrue(soc_version >= -1)


if __name__ == "__main__":
    run_tests()
