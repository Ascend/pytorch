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


class TestTensorOP(torch.nn.Module):

    def forward(self, x, y):
        z_add = torch.add(x, y)
        z_sub = torch.sub(x, y)
        z_relu = torch.relu(z_add)
        z = torch.add(z_sub, z_relu)
        return z


class TestModuleOP(torch.nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 4, 1)
        self.relu = torch.nn.ReLU()
        self.bn = torch.nn.BatchNorm2d(4)

    def forward(self, x):
        x1 = self.conv(x)
        x2 = self.relu(x1)
        r1 = self.bn(x2)
        return r1


class TestAccCmpHook(TestCase):

    def test_tensor_op(self):
        module = TestTensorOP()
        register_acc_cmp_hook(module)
        seed_all()
        set_dump_path("./cpu_tensor_op.pkl")
        x = torch.randn(2, 3, 4)
        y = torch.randn(2, 3, 4)
        x.requires_grad = True
        y.requires_grad = True
        out = module(x, y)
        loss = torch.sum(out)
        loss.backward()
        set_dump_path("./npu_tensor_op.pkl")
        module.npu()
        x = x.npu()
        y = y.npu()
        out = module(x, y)
        loss = out.sum()
        loss.backward()
        assert os.path.exists("./cpu_tensor_op.pkl") and os.path.exists("./npu_tensor_op.pkl")
    
    def test_module_op(self):
        module = TestModuleOP()
        register_acc_cmp_hook(module)
        seed_all()
        set_dump_path("./cpu_module_op.pkl")
        x = torch.randn(2, 3, 12, 12)
        out = module(x)
        loss = torch.sum(out)
        loss.backward()
        set_dump_path("./npu_module_op.pkl")
        module.npu()
        x = x.npu()
        out = module(x)
        loss = out.sum()
        loss.backward()
        assert os.path.exists("./cpu_module_op.pkl") and os.path.exists("./npu_module_op.pkl")

    def tearDown(self) -> None:
        for filename in os.listdir('./'):
            if filename.endswith(".pkl"):
                os.remove("./" + filename)


if __name__ == '__main__':
    run_tests()
