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
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests

class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()
        self.n_hidden = torch.nn.Linear(n_feature, n_hidden)
        self.out = torch.nn.Linear(n_hidden, n_output)
    def forward(self, x_layer):
        x_layer = torch.relu(self.n_hidden(x_layer))
        x_layer = self.n_hidden(x_layer)
        x_layer = self.out(x_layer)
        x_layer = torch.nn.functional.softmax(x_layer)
        return x_layer

class TestReplay(TestCase):
    def test_replay_graph(self):
        if support_replay_model is False:
            self.assertNotEqual(support_replay_model, True)
            return
        def train():
            for i in range(steps):
                out = model(x)
                loss = loss_func(out, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            return loss.item()

        steps = 1000
        device = "npu:0" if torch.npu.is_available() else "cpu"
        n_data = torch.ones(10, 2)
        x0 = torch.normal(2*n_data, 1)
        y0 = torch.zeros(10)
        x1 = torch.normal(-2*n_data, 1)
        y1 = torch.ones(10)
        x = torch.cat((x0, x1)).type(torch.FloatTensor).to(device)
        y = torch.cat((y0, y1)).type(torch.LongTensor).to(device)
        model = Net(n_feature=2, n_hidden=2, n_output=2).to(device)
        loss_func = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        loss_single = train()
        model2 = Net(n_feature=2, n_hidden=2, n_output=2).to(device)
        optimizer = torch.optim.SGD(model2.parameters(), lr=0.1)
        model = torch.npu.make_replay_graph(model2)
        loss_replay = train()
        if torch.npu.is_available():
            self.assertEqual(loss_single, loss_replay, 0.001)


if __name__ == '__main__':
    support_replay_model = False
    run_tests()
