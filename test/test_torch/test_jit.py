# Copyright (c) 2023, Huawei Technologies.All rights reserved.
# Copyright (c) 2019, Facebook CORPORATION.
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


class TestJit(TestCase):

    def test_calls_in_type_annotations(self):
        with self.assertRaisesRegex(RuntimeError, "Type annotation should not contain calls"):
            def spooky(a):
                # type: print("Hello") -> Tensor
                return a + 2
            torch.jit.annotations.get_signature(spooky, None, 1, True)


if __name__ == "__main__":
    run_tests()
