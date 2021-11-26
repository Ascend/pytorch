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
from common_utils import TestCase, run_tests
from common_device_type import dtypes, instantiate_device_type_tests
from util_test import create_common_tensor
import torch.nn.functional as F
from graph_utils import RunFuncInGraphMode

class TestEmbeddingBackward(TestCase):
    def cpu_op_exec_ext(self, weight, indices):
        weight.requires_grad_(True)
        out = F.embedding(indices, weight, scale_grad_by_freq=True, padding_idx=37)
        out.backward(torch.ones_like(out))
        grad_cpu = weight.grad
        return out.detach().numpy(), grad_cpu.detach().numpy()

    def npu_op_exec_ext(self, weight, indices):
        weight.requires_grad_(True)
        out = F.embedding(indices, weight, scale_grad_by_freq=True, padding_idx=37)
        out.backward(torch.ones_like(out))
        out_npu = out.to("cpu")
        grad_npu = weight.grad
        grad_npu = grad_npu.to("cpu")
        return out_npu.detach().numpy(), grad_npu.detach().numpy()

    @RunFuncInGraphMode
    def test_shape_format_ext(self, device):
        shape_format = [
                        # [[np.float32, 0, [40,32]], [np.int64, 0, [40]]],
                        # [[np.float32, 0, [40,1024]], [np.int64, 0, [40]]],
                        # [[np.float32, 0, [40000,1024]], [np.int64, 0, [3125]]],
                        # [[np.float32, 0, [40000,1024]], [np.int64, 0, [128,8]]],
                        # [[np.float16, 0, [40,32]], [np.int64, 0, [40]]],
                        # [[np.float16, 0, [40,1024]], [np.int64, 0, [128,8]]],
                        # [[np.float16, 0, [33712,1024]], [np.int64, 0, [64,7]]],
                        # [[np.float32, 3, [40000,1024]], [np.int64, 0, [3125]]],
                        # [[np.float32, 2, [40000,1024]], [np.int64, 0, [128,8]]],
                        # [[np.float16, 0, [40,32]], [np.int64, 0, [40]]],
                        # [[np.float16, 29, [40,1024]], [np.int64, 0, [128,8]]]
                        ]
        for item in shape_format:
            weight_cpu, weight_npu = create_common_tensor(item[0], 1, 1)
            indices_cpu, indices_npu = create_common_tensor(item[1], 0, 1)

            if weight_cpu.dtype == torch.float16:
                weight_cpu = weight_cpu.to(torch.float32)


            cpu_out, cpu_grad = self.cpu_op_exec_ext(weight_cpu, indices_cpu)
            npu_out, npu_grad = self.npu_op_exec_ext(weight_npu, indices_npu)

            cpu_out = cpu_out.astype(npu_out.dtype)
            cpu_grad = cpu_grad.astype(npu_grad.dtype)

            self.assertRtolEqual(cpu_out, npu_out)
            self.assertRtolEqual(cpu_grad, npu_grad)


instantiate_device_type_tests(TestEmbeddingBackward, globals(), except_for="cpu")
if __name__ == "__main__":
    run_tests()