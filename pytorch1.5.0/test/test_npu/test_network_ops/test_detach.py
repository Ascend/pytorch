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
import torch.nn as nn
import numpy as np
from common_utils import TestCase, run_tests
from common_device_type import dtypes, instantiate_device_type_tests
from util_test import create_common_tensor

class TestDetach(TestCase):
    def test_detach(self, device):
        x = torch.randn(10, 10, requires_grad=True, device="npu")
        y = x + 2
        y = y.detach()
        z = y * 4 + 2
        self.assertFalse(y.requires_grad)
        self.assertFalse(z.requires_grad)

        x = torch.randn(10, 10, requires_grad=True, device="npu")
        y = x * 2
        y = y.detach()
        self.assertFalse(y.requires_grad)
        self.assertIsNone(y.grad_fn)
        z = x + y
        z.sum().backward()
        # This is an incorrect gradient, but we assume that's what the user
        # wanted. detach() is an advanced option.
        self.assertRtolEqual(x.grad.cpu().numpy(), torch.ones(10, 10).numpy())

        # in-place detach
        x = torch.randn(10, 10, requires_grad=True, device="npu")
        y = torch.randn(10, 10, requires_grad=True, device="npu")
        a = x * 2
        (y + a).sum().backward(retain_graph=True)
        a.detach_()
        self.assertFalse(a.requires_grad)
        (y + a).sum().backward()  # this won't backprop to x
        self.assertRtolEqual(x.grad.cpu().numpy(), (torch.ones(10, 10) * 2).numpy())
        self.assertRtolEqual(y.grad.cpu().numpy(), (torch.ones(10, 10) * 2).numpy())

        # in-place deatch on a view raises an exception
        view = x.narrow(0, 1, 4)
        self.assertRaisesRegex(RuntimeError, 'view', lambda: view.detach_())
    
    def test_detach_shape_format(self, device):
        shape_format = [
                [np.float32, 0, (4, 3)],
                [np.float32, 0, (2, 3, 7)]
        ]

        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item, -100, 100)
            cpu_input1.requires_grad = True
            npu_input1.requires_grad = True

            cpu_output1 = cpu_input1.detach()
            npu_output1 = npu_input1.detach().cpu()
            self.assertRtolEqual(cpu_output1, npu_output1)
            self.assertEqual(cpu_output1.requires_grad, npu_output1.requires_grad)

            cpu_output2 = cpu_input1.detach()
            npu_output2 = npu_input1.detach().cpu()
            self.assertRtolEqual(cpu_output2, npu_output2)
            self.assertEqual(cpu_output2.requires_grad, npu_output2.requires_grad)


instantiate_device_type_tests(TestDetach, globals(), except_for="cpu")
if __name__ == "__main__":
    run_tests()