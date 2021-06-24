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
import random


class TestCosineEmbeddingLoss(TestCase):
    def generate_target(self, shape, dtype):
        target = np.random.randint(2, size=shape, dtype=dtype)
        target = target*2-1
        target = torch.from_numpy(target)
        return target

    def cpu_op_exec(self, input1, input2, target, margin, reduction):
        output = torch.nn.functional.cosine_embedding_loss(
            input1, input2, target, margin=margin, reduction=reduction)
        output = output.numpy()
        return output

    def npu_op_exec(self, input1, input2, target, margin, reduction):
        input1 = input1.to("npu")
        input2 = input2.to("npu")
        target = target.to("npu")
        output = torch.nn.functional.cosine_embedding_loss(
            input1, input2, target, margin=margin, reduction=reduction)
        output = output.to("cpu")
        output = output.numpy()
        return output

    def test_cosine_embedding_loss_common_shape_format(self, device):
        shape_format = [
            [[np.float32, -1, (5, 3)], [np.float32, -1, (5, 3)],
             [np.int32, (5, )], 'sum'],
            [[np.float32, 0, (16, 4, 3)], [np.float32, 0,
                                            (16, 4, 3)], [np.int32, (16, 3)], 'mean'],
            [[np.float32, 3, (64, 10, 10)], [np.float32, 3,
                                              (64, 10, 10)], [np.int32, (64, 10)], 'none'],
        ]
        for item1, item2, target, reduction in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item1, 1, 100)
            cpu_input2, npu_input2 = create_common_tensor(item2, 1, 100)
            target = self.generate_target(target[1], target[0])
            margin = np.random.uniform(0, 1)
            cpu_output = self.cpu_op_exec(
                cpu_input1, cpu_input2, target, margin, reduction)
            npu_output = self.npu_op_exec(
                npu_input1, npu_input2, target, margin, reduction)
            self.assertRtolEqual(cpu_output, npu_output)

    def test_cosine_embedding_loss_float16_shape_format(self, device):
        def cpu_op_exec_fp16(input1, input2, target, margin, reduction):
            input1 = input1.to(torch.float32)
            input2 = input2.to(torch.float32)
            output = torch.nn.functional.cosine_embedding_loss(
                input1, input2, target, margin=margin, reduction=reduction)
            output = output.numpy().astype(np.float16)
            return output

        shape_format = [
            [[np.float16, 3, (4, 1, 3)], [np.float16, 3,
                                           (4, 1, 3)], [np.int32, (4, 3)], 'sum'],
            [[np.float16, -1, (16, 8)], [np.float16, -1, (16, 8)],
             [np.int32, (16, )], 'mean'],
            [[np.float16, 4, (64, 10, 10)], [np.float16, 3,
                                              (64, 10, 10)], [np.int32, (64, 10)], 'none']
        ]

        for item1, item2, target, reduction in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item1, 1, 100)
            cpu_input2, npu_input2 = create_common_tensor(item2, 1, 100)
            target = self.generate_target(target[1], target[0])
            margin = np.random.uniform(0, 1)
            cpu_output = cpu_op_exec_fp16(
                cpu_input1, cpu_input2, target, margin, reduction)
            npu_output = self.npu_op_exec(
                npu_input1, npu_input2, target, margin, reduction)
            self.assertRtolEqual(cpu_output, npu_output)

instantiate_device_type_tests(
    TestCosineEmbeddingLoss, globals(), except_for='cpu')
if __name__ == "__main__":
    torch.npu.set_device("npu:5")
    run_tests()
