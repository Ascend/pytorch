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
import torch.nn as nn

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.hooks import set_dump_path, seed_all, register_acc_cmp_hook
from torch_npu.hooks.tools import compare


class TestTensorBuiltinOP(nn.Module):

    def forward(self, x, y):
        z_add = x + y
        z_sub = x - y
        z = z_add * z_sub
        return z


class TestTensorMethodOP(nn.Module):
    
    def forward(self, x, y):
        z_add = x.add(y)
        z_abs = z_add.abs()
        z = z_abs.softmax(dim=0)
        return z


class TestAccCmpTensorHook(TestCase):

    def test_tensor_bulitin_op(self):
        module = TestTensorBuiltinOP()
        register_acc_cmp_hook(module)
        seed_all()
        set_dump_path("./cpu_tensor_builtin_op.pkl")
        x = torch.randn(2, 3, 4)
        y = torch.randn(2, 3, 4)
        x.requires_grad = True
        y.requires_grad = True
        out = module(x, y)
        loss = out.sum()
        loss.backward()
        set_dump_path("./npu_tensor_builtin_op.pkl")
        module.npu()
        x = x.npu()
        y = y.npu()
        out = module(x, y)
        loss = out.sum()
        loss.backward()
        assert os.path.exists("./cpu_tensor_builtin_op.pkl") and os.path.exists("./npu_tensor_builtin_op.pkl")
        compare("./npu_tensor_builtin_op.pkl", "./cpu_tensor_builtin_op.pkl", "./tensor_builtin_op_result.csv")
        assert os.path.exists("./tensor_builtin_op_result.csv")

    def test_tensor_method_op(self):
        module = TestTensorMethodOP()
        register_acc_cmp_hook(module)
        seed_all()
        set_dump_path("./cpu_tensor_method_op.pkl")
        x = torch.randn(2, 3, 4)
        y = torch.randn(2, 3, 4)
        x.requires_grad = True
        y.requires_grad = True
        out = module(x, y)
        loss = out.sum()
        loss.backward()
        set_dump_path("./npu_tensor_method_op.pkl")
        module.npu()
        x = x.npu()
        y = y.npu()
        out = module(x, y)
        loss = out.sum()
        loss.backward()
        assert os.path.exists("./cpu_tensor_method_op.pkl") and os.path.exists("./npu_tensor_method_op.pkl")
        compare("./npu_tensor_method_op.pkl", "./cpu_tensor_method_op.pkl", "./tensor_method_op_result.csv")
        assert os.path.exists("./tensor_method_op_result.csv")
    
    def tearDown(self) -> None:
        for filename in os.listdir('./'):
            if filename.endswith(".pkl") or filename.endswith(".csv"):
                os.remove("./" + filename)


if __name__ == '__main__':
    run_tests()

