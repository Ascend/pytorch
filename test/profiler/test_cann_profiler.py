# Copyright (c) 2020 Huawei Technologies Co., Ltd
# Copyright (c) 2019, Facebook CORPORATION.
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
import shutil
from itertools import combinations
import torch
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.utils.path_manager import PathManager


class SmallModel(torch.nn.Module):
    def __init__(self, in_channel=3, out_channel=12):
        super(SmallModel, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channel, in_channel, 3, padding=1)
        self.relu1 = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv2d(in_channel, out_channel, 3, padding=1)

    def forward(self, input_1):
        input_1 = self.conv1(input_1)
        input_1 = self.relu1(input_1)
        input_1 = self.conv2(input_1)
        return input_1.reshape(input_1.shape[0], -1)


def run_ops():
    input_1 = torch.rand(10, 10).npu()
    input_2 = torch.rand(10, 10).npu()
    out = input_1 * input_2


def run_small_model():
    input_shape = (4, 3, 24, 24)
    out_shape = (4, 12, 24, 24)
    device = "npu"
    model = SmallModel(input_shape[1], out_shape[1]).to(device)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0001)
    for i in range(10):
        inputs = torch.rand(input_shape).to(device)
        target = torch.rand(out_shape).reshape(out_shape[0], -1).to(device)
        output = model(inputs)
        loss = criterion(output, target)
        loss.backward()
        optimizer.zero_grad()
        optimizer.step()


def setUp(results_path, use_e2e_profiler=False):
    if not os.path.exists(results_path):
        PathManager.make_dir_safety(results_path)
    if not use_e2e_profiler:
        torch.npu.prof_init(results_path)
    tensor = torch.rand(2, 3).npu()
    result = []

    enevtTypes = [{"ACL_PROF_ACL_API": False}, {"ACL_PROF_TASK_TIME": False},
                  {"ACL_PROF_AICORE_METRICS": False}, {"ACL_PROF_AICPU": False},
                  {"ACL_PROF_L2CACHE": False}, {"ACL_PROF_HCCL_TRACE": False},
                  {"ACL_PROF_TRAINING_TRACE": False}]

    enevtTypeCombinations = list(combinations(enevtTypes, 1)) + list(combinations(enevtTypes, 2)) + \
        list(combinations(enevtTypes, 3)) + list(combinations(enevtTypes, 4)) + \
        list(combinations(enevtTypes, 5)) + list(combinations(enevtTypes, 6))
    for events in enevtTypeCombinations:
        temp_events = {}
        for event in events:
            temp_events.update(event)
        result.append(temp_events)
    return result


class TestCannProfiler(TestCase):
    enevtTypeResults = None
    results_path = "./results"

    @classmethod
    def setUpClass(cls):
        TestCannProfiler.enevtTypeResults = setUp(TestCannProfiler.results_path)

    @classmethod
    def tearDownClass(cls):
        if os.path.exists(TestCannProfiler.results_path):
            PathManager.remove_path_safety(TestCannProfiler.results_path)
        torch.npu.prof_finalize()

    def _test_cann_ops(self, *args, **kwargs):
        config = torch.npu.profileConfig(**kwargs)
        torch.npu.prof_start(config)
        run_ops()
        torch.npu.prof_stop()

    def _test_cann_model(self, *args, **kwargs):
        config = torch.npu.profileConfig(**kwargs)
        torch.npu.prof_start(config)
        run_small_model()
        torch.npu.prof_stop()

    def test_with_ops(self):
        for events in TestCannProfiler.enevtTypeResults:
            for i in range(5):
                self._test_cann_ops(**events, aiCoreMetricsType=i)

    def test_with_small_model(self):
        for events in TestCannProfiler.enevtTypeResults:
            for i in range(5):
                self._test_cann_model(**events, aiCoreMetricsType=i)


class TestE2EProfiler(TestCase):
    enevtTypeResults = None
    results_path = "./results"

    @classmethod
    def setUpClass(cls):
        TestE2EProfiler.enevtTypeResults = setUp(TestE2EProfiler.results_path, True)

    @classmethod
    def tearDownClass(cls):
        if os.path.exists(TestE2EProfiler.results_path):
            PathManager.remove_path_safety(TestE2EProfiler.results_path)

    def _test_e2e_ops(self, *args, **kwargs):
        config = torch.npu.profileConfig(**kwargs)
        with torch.npu.profile(TestE2EProfiler.results_path, True, config):
            run_ops()

    def _test_e2e_model(self, *args, **kwargs):
        config = torch.npu.profileConfig(**kwargs)
        with torch.npu.profile(TestE2EProfiler.results_path, True, config):
            run_small_model()

    def test_with_ops(self):
        for events in TestE2EProfiler.enevtTypeResults:
            for i in range(5):
                self._test_e2e_ops(**events, aiCoreMetricsType=i)
            return

    def test_with_small_model(self):
        for events in TestE2EProfiler.enevtTypeResults:
            for i in range(5):
                self._test_e2e_model(**events, aiCoreMetricsType=i)
            return


if __name__ == "__main__":
    run_tests()
