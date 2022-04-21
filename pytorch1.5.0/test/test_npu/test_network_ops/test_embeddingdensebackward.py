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


class TestEmbeddingDenseBackward(TestCase):
    def cpu_op_exec(self, weight, indices):
        weight.requires_grad_(True)
        out = F.embedding(indices, weight, scale_grad_by_freq=False, padding_idx=-1)
        out.backward(torch.ones_like(out))
        grad_cpu = weight.grad
        return out.detach().numpy(), grad_cpu.detach().numpy()

    def npu_op_exec(self, weight, indices):
        weight.requires_grad_(True)
        out = F.embedding(indices, weight, scale_grad_by_freq=False, padding_idx=-1)
        out.backward(torch.ones_like(out))
        out_npu = out.to("cpu")
        grad_npu = weight.grad
        grad_npu = grad_npu.to("cpu")
        return out_npu.detach().numpy(), grad_npu.detach().numpy()

    def test_embedding_dense_backward_shape_format_fp32(self, device):
        format_list = [0]
        shape_list1 = [[40, 32], [40, 1024], [40000, 1024], [33712, 1024]]
        shape_list2 = [[40], [40], [40000], [33712]]
        shape_format1 = [
            [np.float32, i, j] for i in format_list for j in shape_list1
        ]
        shape_format2 = [
            [np.int64, i, j] for i in format_list for j in shape_list2
        ]
        shape_format = [
            [shape_format1[i], shape_format2[i]] for i in range(len(shape_list1))
        ]
        for item in shape_format:
            weight_cpu, weight_npu = create_common_tensor(item[0], 1, 1)
            indices_cpu, indices_npu = create_common_tensor(item[1], 0, min(item[0][2][0:-1]))

            cpu_out, cpu_grad = self.cpu_op_exec(weight_cpu, indices_cpu)
            npu_out, npu_grad = self.npu_op_exec(weight_npu, indices_npu)

            self.assertRtolEqual(cpu_out, npu_out)
            self.assertRtolEqual(cpu_grad, npu_grad)


instantiate_device_type_tests(TestEmbeddingDenseBackward, globals(), except_for="cpu")
if __name__ == "__main__":
    run_tests()

