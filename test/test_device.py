# Copyright (c) 2021, Huawei Technologies.All rights reserved.
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
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests

class TestDevice(TestCase):
    def device_monitor(func):
        def wrapper(self, *args, **kwargs):
            device_id = 0
            torch.npu.set_device(device_id)
            npu_device = torch.randn(2).npu(device_id).device
            device_types = [
                            "npu",
                            "npu:"+str(device_id),
                            torch.device("npu:"+str(device_id)),
                            torch.device("npu:"+str(device_id)).type,
                            npu_device
                            ]
            for device_type in device_types:
                kwargs["device"] = device_type
                npu_tensor = func(self, *args, **kwargs)
                self.assertEqual(npu_tensor.device.type, "npu")
                self.assertEqual(npu_tensor.device.index, device_id)
        return wrapper


    @device_monitor
    def test_torch_tensor_to_device(self, device=None):
        cpu_tensor = torch.randn(2, 3)
        return cpu_tensor.to(device, torch.int64)


    @device_monitor
    def test_torch_tensor_new_empty_with_device_input(self, device=None):
        npu_tensor = torch.ones(2, 3).to(device)
        return npu_tensor.new_empty((2, 3), dtype=torch.float16, device=device)


    @device_monitor
    def test_torch_func_arange_with_device_input(self, device=None):
        return torch.arange(5, dtype=torch.float32, device=device)


    @device_monitor
    def test_torch_func_zeros_with_device_input(self, device=None):
        return torch.zeros((2, 3), dtype=torch.int8, device=device)


    @device_monitor
    def test_tensor_method_npu_with_device_input(self, device=None):
        if isinstance(device, str):
            device = torch.device(device)
        cpu_input = torch.randn(2,3)
        return cpu_input.npu(device)


if __name__ == '__main__':
    run_tests()