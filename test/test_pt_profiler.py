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

class TestProfiler(TestCase):

    def mm_op(self, device="cpu"):
        a = torch.rand(5, 5).to(device)
        b = torch.randn(5, 5).to(device)
        c = torch.mm(a, b)

    def test_cpu_op_profiler(self):
        with torch.autograd.profiler.profile(use_npu=False) as prof:
            self.mm_op()
        found_mm = False 

        for e in prof.function_events:
            if "mm" in e.name:
                found_mm = True
        self.assertTrue(found_mm)

    def test_npu_op_profiler(self):
        # test basic function for npu op
        if torch.npu.is_available():
            device = "npu:0"
        else:
            return
        with torch.autograd.profiler.profile(use_npu=True) as prof:
            self.mm_op(device)
        found_mm = False 

        for e in prof.function_events:
            if "mm" in e.name:
                found_mm = True
        self.assertTrue(found_mm)

    def test_memory_profiler(self):
        # test momory usage
        def run_profiler(creat_tensor, metric):
            # collecting allocs / deallocs
            with torch.autograd.profiler.profile(profile_memory=True, 
                                            record_shapes=False, use_npu=True) as prof:
                input_x = None
                with torch.autograd.profiler.record_function("user_allocate"):
                    input_x = creat_tensor()
                with torch.autograd.profiler.record_function("user_deallocate"):
                    del input_x
            return prof.key_averages()

        def check_metrics(stats, metric, allocs=None, deallocs=None):
            stat_metrics = {}
            for stat in stats:
                stat_metrics[stat.key] = getattr(stat, metric)
            if allocs is not None:
                for alloc_fn in allocs:
                    self.assertTrue(alloc_fn in stat_metrics)
                    self.assertTrue(stat_metrics[alloc_fn] > 0)
            if deallocs is not None:
                for dealloc_fn in deallocs:
                    self.assertTrue(dealloc_fn in stat_metrics)
                    self.assertTrue(stat_metrics[dealloc_fn] < 0)

        def create_cpu_tensor():
            return torch.rand(10, 10)

        def create_npu_tensor():
            return torch.rand(20, 30).npu()

        stats = run_profiler(create_cpu_tensor, "cpu_memory_usage")
        check_metrics(
            stats,
            "cpu_memory_usage",
            allocs=[
                "aten::empty",
                "aten::rand",
                "user_allocate",
            ],
            deallocs=[
                "user_deallocate",
            ]
        )

        if torch.npu.is_available():
            create_npu_tensor()
            stats = run_profiler(create_npu_tensor, "npu_memory_usage")
            check_metrics(
                stats,
                "npu_memory_usage",
                allocs=[
                    "user_allocate",
                    "aten::to",
                    "aten::empty_strided",
                ],
                deallocs=[
                    "user_deallocate",
                ]
            )
            check_metrics(
                stats,
                "cpu_memory_usage",
                allocs=[
                    "aten::rand",
                    "aten::empty",
                ]
            )

    def test_model_profiler(self):
        """Checks that model forward and backward.
        """
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

        try:
            train()
        except Exception:
            self.assertTrue(False, "Expected no exception without profiling.")

        def judge(expected_event_count, prof):
            actual_event_count = {}
            for e in prof.function_events:
                if "#" in e.name:
                    key = e.name
                    if key in expected_event_count.keys():
                        actual_event_count[key] = actual_event_count.setdefault(key, 0) + 1
            for key, count in expected_event_count.items():
                self.assertTrue((key in actual_event_count.keys()) and (count == actual_event_count[key]))

        with torch.autograd.profiler.profile(use_npu=True) as prof:
            train()
        expected_event_count = {
            "Optimizer.step#SGD.step": steps,
            "Optimizer.zero_grad#SGD.zero_grad": steps
        }
        judge(expected_event_count, prof)


    def test_npu_simple_profiler(self):
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
        try:
            train()
        except Exception:
            self.assertTrue(False, "Expected no exception without profiling.")
        with torch.autograd.profiler.profile(use_npu=True, use_npu_simple=True) as prof:
            train()
        prof.export_chrome_trace("./test_trace.prof")

if __name__ == '__main__':
    try:
        # to init the device
        torch.rand(2,3).npu()
    except Exception:
        print("there is no npu device")
        exit()
    run_tests()
