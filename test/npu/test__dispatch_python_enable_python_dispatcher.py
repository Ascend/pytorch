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
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Validate Python dispatcher context behavior with NPU tensors.

The cases cover dispatcher activation, nesting, cleanup after normal and
exceptional exits, argument validation, and a real NPU dispatcher path.
"""

import torch
from torch._dispatch.python import enable_python_dispatcher
from torch.testing._internal.common_utils import TestCase as TorchTestCase
from torch.testing._internal.common_utils import run_tests

from torch_npu.testing.testcase import TestCase


device_type = (
    acc.type if (acc := torch.accelerator.current_accelerator()) else "cpu"
)


def is_python_dispatcher_enabled():
    return torch._C._dispatch_tls_is_dispatch_key_included(
        torch._C.DispatchKey.PythonDispatcher
    )


class TestEnablePythonDispatcher(TorchTestCase, TestCase):
    def test_context_enables_and_restores_dispatcher(self):
        before = is_python_dispatcher_enabled()
        self.assertFalse(before)
        input_tensor = torch.tensor([1.0, 2.0, 3.0]).to(device_type)
        self.assertEqual(input_tensor.device.type, "npu")

        with enable_python_dispatcher():
            self.assertTrue(is_python_dispatcher_enabled())
            output = input_tensor.square() + 1
            self.assertEqual(output.device.type, "npu")
            self.assertEqual(
                output, torch.tensor([2.0, 5.0, 10.0]).to(device_type)
            )

        self.assertEqual(is_python_dispatcher_enabled(), before)

    def test_nested_context_restores_outer_state(self):
        before = is_python_dispatcher_enabled()
        self.assertFalse(before)

        with enable_python_dispatcher():
            self.assertTrue(is_python_dispatcher_enabled())
            with enable_python_dispatcher():
                self.assertTrue(is_python_dispatcher_enabled())
            self.assertTrue(is_python_dispatcher_enabled())

        self.assertEqual(is_python_dispatcher_enabled(), before)

    def test_context_restores_state_after_exception(self):
        before = is_python_dispatcher_enabled()
        self.assertFalse(before)

        with self.assertRaisesRegex(RuntimeError, "sentinel"):
            with enable_python_dispatcher():
                self.assertTrue(is_python_dispatcher_enabled())
                raise RuntimeError("sentinel")

        self.assertEqual(is_python_dispatcher_enabled(), before)

    def test_invalid_argument(self):
        with self.assertRaises(TypeError):
            enable_python_dispatcher("invalid")


if __name__ == "__main__":
    run_tests()
