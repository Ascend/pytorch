# Copyright (c) 2026 Huawei Technologies Co., Ltd
# All rights reserved.
#
# Licensed under the BSD 3-Clause License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://opensource.org/licenses/BSD-3-Clause
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
# implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Add validation cases for torch.autograd.profiler APIs on NPU:
1. PyTorch community lacks sufficient and direct API validations for some APIs, so this file is added.
2. This file validates torch.autograd.profiler.profile.total_average (extendable).
"""

import torch

from torch_npu.testing.testcase import run_tests, TestCase

device_type = acc.type if (acc := torch.accelerator.current_accelerator()) else "cpu"


class TestProfileTotalAverage(TestCase):
    def test_total_average_returns_aggregate_on_npu(self):
        """total_average() aggregates events produced by NPU work."""
        device = torch.device(device_type)

        with torch.autograd.profiler.profile(use_device=device_type) as profiler:
            tensor = torch.randn(16, 16, device=device)
            result = torch.mm(tensor, tensor)
            result.sum().item()
            torch.npu.synchronize()

        average = profiler.total_average()
        self.assertIsNotNone(average)
        self.assertEqual(average.key, "Total")
        self.assertGreater(average.cpu_time_total, 0)

    def test_total_average_returns_zero_aggregate_for_empty_profile(self):
        """The parameterless API returns a zero aggregate for no events."""
        with torch.autograd.profiler.profile(use_device=device_type) as profiler:
            pass

        average = profiler.total_average()
        self.assertIsNotNone(average)
        self.assertEqual(average.key, "Total")
        self.assertEqual(average.cpu_time_total, 0)


if __name__ == "__main__":
    run_tests()
