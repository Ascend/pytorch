# Copyright (c) 2020 Huawei Technologies Co., Ltd
# All rights reserved.
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
from torch_npu.hooks import set_dump_path, seed_all, register_acc_cmp_hook
from torch_npu.hooks.tools import compare


class TestFunctionalOP(torch.nn.Linear):

    def forward(self, x):
        x1 = torch.nn.functional.linear(x, self.weight, self.bias)
        x2 = torch.nn.functional.relu(x1)
        return x2


class TestAccCmpFunctionalHook(TestCase):

    def test_functional_op(self):
        module = TestFunctionalOP(3, 4)
        register_acc_cmp_hook(module)
        seed_all()
        set_dump_path("./cpu_functional_op.pkl")
        x = torch.randn(2, 3, 3)
        x.requires_grad = True
        out = module(x)
        loss = torch.sum(out)
        loss.backward()
        set_dump_path("./npu_functional_op.pkl")
        module.npu()
        x = x.npu()
        out = module(x)
        loss = out.sum()
        loss.backward()
        assert os.path.exists("./cpu_functional_op.pkl") and os.path.exists("./npu_functional_op.pkl")
        compare("./npu_functional_op.pkl", "./cpu_functional_op.pkl", "./functional_op_result.csv")
        assert os.path.exists("./functional_op_result.csv")

    def tearDown(self) -> None:
        for filename in os.listdir('./'):
            if filename.endswith(".pkl"):
                os.remove("./" + filename)


if __name__ == '__main__':
    run_tests()
