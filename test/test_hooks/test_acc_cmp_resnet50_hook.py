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
import copy
import torch
import torch.nn as nn
from torchvision import models

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.hooks import set_dump_path, seed_all, register_acc_cmp_hook
from torch_npu.hooks.tools import compare


class TestResNet50AccCmpHook(TestCase):

    def test_resnet50_op(self):
        model_cpu = models.resnet50()
        model_cpu.eval()
        model_npu = copy.deepcopy(model_cpu)
        model_npu.eval()
        register_acc_cmp_hook(model_cpu)
        register_acc_cmp_hook(model_npu)
        seed_all()
        inputs = torch.randn(1, 3, 244, 244)
        labels = torch.randn(1).long()
        criterion = nn.CrossEntropyLoss()
        set_dump_path("./cpu_resnet50_op.pkl")
        output = model_cpu(inputs)
        loss = criterion(output, labels)
        loss.backward()
        set_dump_path("./npu_resnet50_op.pkl")
        model_npu.npu()
        inputs = inputs.npu()
        labels = labels.npu()
        output = model_npu(inputs)
        loss = criterion(output, labels)
        loss.backward()
        assert os.path.exists("./npu_resnet50_op.pkl") and os.path.exists("./cpu_resnet50_op.pkl")
        compare("./npu_resnet50_op.pkl", "./cpu_resnet50_op.pkl", "./resnet50_result.csv")
        assert os.path.exists("./resnet50_result.csv")

    def tearDown(self) -> None:
        for filename in os.listdir('./'):
            if filename.endswith(".pkl") or filename.endswith(".csv"):
                os.remove("./" + filename)

            
if __name__ == '__main__':
    run_tests()

