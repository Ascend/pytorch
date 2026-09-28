# Copyright (c) 2026 Huawei Technologies Co., Ltd.
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
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Add validation cases for torch.autograd.forward_ad.enter_dual_level and
torch.autograd.forward_ad.unpack_dual on NPU.

This file validates the forward AD level lifecycle and verifies that unpack_dual
returns the expected primal and tangent values for NPU dual tensors.
"""

import torch
import torch_npu
from torch.autograd import forward_ad
from torch_npu.testing.testcase import TestCase, run_tests


class TestForwardAd(TestCase):
    def test_enter_dual_level(self):
        cpu_primal = torch.tensor([1.0, 2.0], dtype=torch.float32)
        cpu_tangent = torch.tensor([3.0, 4.0], dtype=torch.float32)

        level = forward_ad.enter_dual_level()
        dual = forward_ad.make_dual(
            cpu_primal.npu(),
            cpu_tangent.npu(),
            level=level,
        )
        unpacked = forward_ad.unpack_dual(dual, level=level)

        actual_primal = unpacked.primal.cpu()
        actual_tangent = unpacked.tangent.cpu()
        primal_device = unpacked.primal.device.type
        tangent_device = unpacked.tangent.device.type

        forward_ad.exit_dual_level(level=level)
        tangent_after_exit = forward_ad.unpack_dual(dual).tangent

        self.assertIsInstance(level, int)
        self.assertEqual(primal_device, "npu")
        self.assertEqual(tangent_device, "npu")
        self.assertRtolEqual(cpu_primal, actual_primal)
        self.assertRtolEqual(cpu_tangent, actual_tangent)
        self.assertIsNone(tangent_after_exit)

    def test_unpack_dual(self):
        for dtype in (torch.float16, torch.float32):
            cpu_primal = torch.tensor([1.0, 2.0, -3.0], dtype=dtype)
            cpu_tangent = torch.tensor([0.5, -1.0, 2.0], dtype=dtype)
            expected_primal = cpu_primal * cpu_primal
            expected_tangent = 2 * cpu_primal * cpu_tangent

            primal = cpu_primal.npu()
            tangent = cpu_tangent.npu()

            with forward_ad.dual_level():
                dual = forward_ad.make_dual(primal, tangent)
                unpacked = forward_ad.unpack_dual(dual * dual)
                plain = forward_ad.unpack_dual(primal)

                actual_primal = unpacked.primal.cpu()
                actual_tangent = unpacked.tangent.cpu()
                actual_dtype = unpacked.primal.dtype
                primal_device = unpacked.primal.device.type
                tangent_device = unpacked.tangent.device.type
                plain_primal = plain.primal.cpu()
                plain_tangent = plain.tangent

            self.assertEqual(primal_device, "npu")
            self.assertEqual(tangent_device, "npu")
            self.assertEqual(actual_dtype, dtype)
            self.assertRtolEqual(expected_primal, actual_primal)
            self.assertRtolEqual(expected_tangent, actual_tangent)
            self.assertRtolEqual(cpu_primal, plain_primal)
            self.assertIsNone(plain_tangent)


if __name__ == "__main__":
    print('torch_npu version is ', torch_npu.__version__)
    print('torch version is', torch.__version__)
    current_device = torch.npu.current_device()
    device_name = torch.npu.get_device_name(current_device)
    print(f"Current NPU name   : {device_name}")
    run_tests()
