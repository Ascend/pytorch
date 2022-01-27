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

import os
import shutil
import torch
from torch_npu.testing.common_utils import TestCase, run_tests
import torch_npu

class SmallModel(torch.nn.Module):
    def __init__(self, in_channel, out_channel):
        super(SmallModel, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channel, in_channel, 1)
        self.relu1 = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv2d(in_channel, out_channel, 1)

    def forward(self, input_1):
        input_1 = self.conv1(input_1)
        input_1 = self.relu1(input_1)
        input_1 = self.conv2(input_1)
        return input_1.reshape(input_1.shape[0], -1)

class TestAoe(TestCase):
    results_path = os.path.join(os.path.split(os.path.realpath(__file__))[0], "graphs")

    @classmethod
    def setUpClass(cls):
        if os.path.exists(TestAoe.results_path):
            shutil.rmtree(TestAoe.results_path)
        os.makedirs(TestAoe.results_path)
        TestAoe.enable_aoe()
    
    @classmethod
    def tearDownClass(cls):
        if os.path.exists(TestAoe.results_path):
            shutil.rmtree(TestAoe.results_path)

    @classmethod
    def enable_aoe(cls):
        option = {"autotune": "enable", "autotunegraphdumppath": TestAoe.results_path}
        torch.npu.set_option(option)

    def test_aoe_dumpgraph(self):
        def train():
            for index in range(steps):
                x = torch.rand(input_shape).to(device)
                y = torch.rand(out_shape).reshape(out_shape[0], -1).to(device)
                y_pred = model(x)
                loss = criterion(y_pred, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        input_shape = (4, 3, 24, 24)
        out_shape = (4, 12, 24, 24)
        steps = 5
        device = "npu:0" if torch.npu.is_available() else "cpu"
        model = SmallModel(input_shape[1], out_shape[1]).to(device)
        criterion = torch.nn.MSELoss(reduction='sum')
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)
        train()

        file_list = os.listdir(TestAoe.results_path)
        if torch.npu.is_available():
            self.assertTrue(len(file_list) > 0)
        

if __name__ == '__main__':
    run_tests()
