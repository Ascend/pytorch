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
from torch.testing._internal.common_utils import TestCase, run_tests

import torch_npu
import torch_npu._C


class TestJitCompile(TestCase):
    def test_jit_compile_false(self):
        torch.npu.set_compile_mode(jit_compile=False)
        self.assertTrue(torch.npu.is_jit_compile_false())

    def test_jit_compile_true(self):
        torch.npu.set_compile_mode(jit_compile=True)
        self.assertFalse(torch.npu.is_jit_compile_false())


if __name__ == "__main__":
    run_tests()
