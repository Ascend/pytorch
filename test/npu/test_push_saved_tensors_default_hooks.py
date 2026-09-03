# Copyright (c) 2026 Huawei Technologies Co., Ltd.
#
# Licensed under the BSD 3-Clause License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.opensource.org/licenses/BSD-3-Clause
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""NPU coverage for torch._C._autograd default saved-tensor hooks."""

import unittest
import torch
from torch_npu.testing.testcase import run_tests, TestCase
from torch.testing._internal.common_utils import TEST_PRIVATEUSE1


@unittest.skipIf(not TEST_PRIVATEUSE1, "test requires NPU")
class TestPushSavedTensorsDefaultHooks(TestCase):

    @staticmethod
    def _run_backward():
        a = torch.randn(5, device="npu", requires_grad=True)
        (a * a).sum().backward()

    def test_push_saved_tensors_default_hooks(self):
        packed = []
        unpacked = []
        hooks_pushed = False

        def pack_hook(tensor):
            packed.append(tensor)
            return {"tensor": tensor}

        def unpack_hook(saved):
            unpacked.append(saved)
            return saved["tensor"]

        a = torch.randn(5, device="npu", requires_grad=True)
        try:
            torch._C._autograd._push_saved_tensors_default_hooks(pack_hook, unpack_hook)
            hooks_pushed = True
            (a * a).sum().backward()
        finally:
            if hooks_pushed:
                torch._C._autograd._pop_saved_tensors_default_hooks()

        self.assertEqual(len(packed), 2)
        self.assertEqual(len(unpacked), 2)
        self.assertTrue(all(tensor.device.type == "npu" for tensor in packed))
        self.assertTrue(all(isinstance(saved, dict) for saved in unpacked))

    def test_invalid_hooks_raise(self):
        # pack_hook raising an exception propagates to the caller.
        def pack_hook(tensor):
            raise RuntimeError("pack hook error")

        def unpack_hook(saved):
            return saved

        try:
            torch._C._autograd._push_saved_tensors_default_hooks(pack_hook, unpack_hook)
            with self.assertRaises(RuntimeError):
                self._run_backward()
        finally:
            torch._C._autograd._pop_saved_tensors_default_hooks()

        # unpack_hook must return a tensor; a non-tensor result raises TypeError.
        def pack_hook(tensor):
            return {"tensor": tensor}

        def unpack_hook(saved):
            return saved  # not a tensor

        try:
            torch._C._autograd._push_saved_tensors_default_hooks(pack_hook, unpack_hook)
            with self.assertRaises(TypeError):
                self._run_backward()
        finally:
            torch._C._autograd._pop_saved_tensors_default_hooks()

    def test_nested_push_pop_stack_order(self):
        outer_packed = []
        outer_unpacked = []
        inner_packed = []
        inner_unpacked = []

        def pack_outer(tensor):
            outer_packed.append(tensor)
            return {"tensor": tensor}

        def unpack_outer(saved):
            outer_unpacked.append(saved)
            return saved["tensor"]

        def pack_inner(tensor):
            inner_packed.append(tensor)
            return {"tensor": tensor}

        def unpack_inner(saved):
            inner_unpacked.append(saved)
            return saved["tensor"]

        # Nested push: only the inner-most pair takes effect.
        torch._C._autograd._push_saved_tensors_default_hooks(pack_outer, unpack_outer)
        torch._C._autograd._push_saved_tensors_default_hooks(pack_inner, unpack_inner)
        self._run_backward()
        self.assertEqual(len(inner_packed), 2)
        self.assertEqual(len(inner_unpacked), 2)
        self.assertEqual(len(outer_packed), 0)
        self.assertEqual(len(outer_unpacked), 0)

        # Pop the inner pair: the outer pair is restored.
        torch._C._autograd._pop_saved_tensors_default_hooks()
        self._run_backward()
        self.assertEqual(len(outer_packed), 2)
        self.assertEqual(len(outer_unpacked), 2)

        # Pop the outer pair: default behavior, hooks no longer apply.
        torch._C._autograd._pop_saved_tensors_default_hooks()
        outer_packed.clear()
        outer_unpacked.clear()
        inner_packed.clear()
        inner_unpacked.clear()
        self._run_backward()
        self.assertEqual(len(outer_packed), 0)
        self.assertEqual(len(outer_unpacked), 0)
        self.assertEqual(len(inner_packed), 0)
        self.assertEqual(len(inner_unpacked), 0)


if __name__ == "__main__":
    run_tests()
