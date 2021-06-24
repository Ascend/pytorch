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
import numpy as np
from common_device_type import dtypes, instantiate_device_type_tests
from util_test import create_common_tensor
from common_utils import TestCase, run_tests


class TestDropOutV2(TestCase):
   def _gen_seeds(self, shape):
       return np.random.uniform(1, 10, size=shape).astype(np.float32)

   def npu_op_exec(self, input1, seed, prob):
        input1.requires_grad = True
        output, mask, seed = torch.npu_dropoutV2(input1, seed, p = prob)
        output.backward(torch.ones_like(output))
        output = output.to("cpu")
        output = output.detach().numpy()
        mask   = mask.to("cpu")
        mask   = mask.numpy()

        output_grad = input1.grad
        output_grad = output_grad.to("cpu")
        output_grad = output_grad.detach().numpy()

        return output_grad, output, mask

   def test_dropoutV2backward(self, device):
        input    = torch.tensor([1.,2.,3.,4.]).npu()
        seed_shape = (int(32 * 1024 * 12),)
        seed = self._gen_seeds(seed_shape)
        seed = torch.from_numpy(seed).to("npu")
        prob     = 0.3
        output_grad, output, mask   = self.npu_op_exec(input, seed, prob) #result is random,only check api can exec success!

instantiate_device_type_tests(TestDropOutV2, globals(), except_for="cpu")
if __name__ == "__main__":
    run_tests()