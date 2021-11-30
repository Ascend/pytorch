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

class TestNpuBertApplyAdam(TestCase):
    def test_npu_bert_apply_adam(self, device):
        seed = 3
        torch.manual_seed(seed)
        torch.npu.manual_seed(seed)
        torch.npu.manual_seed_all(seed)

        var_in = torch.rand(321538).uniform_(-32., 21.).npu()
        m_in = torch.zeros(321538).npu()
        v_in = torch.zeros(321538).npu()
        grad = torch.rand(321538).uniform_(-0.05, 0.03).npu()

        var_ans = torch.tensor([13.1862, -30.1250, -20.4954])
        m_ans = torch.tensor([0.0014, 0.0018, -0.0021])
        v_ans = torch.tensor([1.8999e-06, 3.2629e-06, 4.4347e-06])

        max_grad_norm = -1.
        beta1 = 0.9
        beta2 = 0.99
        weight_decay = 0.
        lr = 0.
        epsilon = 1e-06
        global_grad_norm = 0.

        var_out, m_out, v_out = torch.npu_bert_apply_adam(
            lr, beta1, beta2, epsilon, grad, max_grad_norm, global_grad_norm, weight_decay, out = (var_in, m_in, v_in))

        self.assertRtolEqual(var_out[:3].cpu(), var_ans)
        self.assertRtolEqual(m_out[:3].cpu(), m_ans)
        self.assertRtolEqual(v_out[:3].cpu(), v_ans)

instantiate_device_type_tests(TestNpuBertApplyAdam, globals(), except_for="cpu")
if __name__ == "__main__":
    run_tests()
