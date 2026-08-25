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

"""Tests torch._C._get_tracing_state inside and outside torch.jit.trace."""

import torch
from torch_npu.testing.testcase import run_tests, TestCase

device_type = acc.type if (acc := torch.accelerator.current_accelerator()) else "cpu"


class TestGetTracingState(TestCase):
    def test_get_tracing_state_outside_trace(self):
        self.assertFalse(bool(torch._C._get_tracing_state()))

    def test_get_tracing_state_during_trace(self):
        seen_states = []

        def fn(x):
            seen_states.append(torch._C._get_tracing_state() is not None)
            return x + x.new_tensor(1)

        x = torch.randn(2, 3, device=device_type, dtype=torch.float32)
        traced = torch.jit.trace(fn, (x,), check_trace=False)

        self.assertEqual(seen_states, [True])
        self.assertFalse(bool(torch._C._get_tracing_state()))

        y = traced(x)
        self.assertTrue(torch.allclose(y, x + 1))
        self.assertIn("aten::add", str(traced.graph))


if __name__ == "__main__":
    run_tests()
