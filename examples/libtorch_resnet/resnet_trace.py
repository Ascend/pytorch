# Copyright (c) 2023 Huawei Technologies Co., Ltd
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
import torchvision

from torch_npu.testing.testcase import TestCase, run_tests


class TestJitTrace(TestCase):

    def test_jit_trace(self):
        model = torchvision.models.resnet18()
        example_input = torch.rand(1, 3, 244, 244)

        resnet_model = torch.jit.trace(model, example_input)
        torch.jit.save(resnet_model, 'resnet_model.pt')
        assert os.path.isfile('./resnet_model.pt')


if __name__ == '__main__':
    run_tests()
