# Copyright (c) 2026 Huawei Technologies Co., Ltd
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

"""
Add validation cases for torch.autograd APIs on NPU:
1. PyTorch community lacks sufficient and direct API validations for some APIs, so this file is added.
2. This file validates torch.autograd.Variable._execution_engine.run_backward (extendable).
"""

import torch

from torch.testing._internal.common_utils import TestCase, run_tests


class TestExecutionEngine(TestCase):

    def test_run_backward_returns_requested_gradients(self):
        accelerator = torch.accelerator.current_accelerator()
        device_type = accelerator.type if accelerator is not None else "cpu"
        x = torch.tensor([2.0], device=device_type, requires_grad=True)
        y = x * x

        grad = torch.autograd.Variable._execution_engine.run_backward(
            (y,),
            (torch.ones_like(y),),
            False,
            False,
            (x,),
            True,
            False,
        )

        self.assertEqual(grad[0].cpu(), torch.tensor([4.0]))


if __name__ == "__main__":
    run_tests()
