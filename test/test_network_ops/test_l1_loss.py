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
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests


class TestL1Loss(TestCase):
    def generate_data(self, lo, hi, shape, dtype):
        predict = np.random.uniform(lo, hi, shape).astype(dtype)
        target = np.random.uniform(lo, hi, shape).astype(dtype)
        # modify from numpy.ndarray to torch.tensor
        npu_predict = torch.from_numpy(predict)
        npu_target = torch.from_numpy(target)
        return npu_predict, npu_target

    def cpu_op_exec(self, predict, target, reduction):
        loss = torch.nn.L1Loss(reduction=reduction)
        output = loss(predict, target)
        output = output.numpy()
        return output

    def npu_op_exec(self, predict, target, reduction):
        predict = predict.to("npu")
        target = target.to("npu")
        loss = torch.nn.L1Loss(reduction=reduction)
        output = loss(predict, target)
        output = output.to("cpu")
        output = output.numpy()
        return output

    def test_l1_loss_float32_mean(self):
        predict, target = self.generate_data(-2, 2, (4, 3), np.float32)
        cpu_output = self.cpu_op_exec(predict, target, "mean")
        npu_output = self.npu_op_exec(predict, target, "mean")
        self.assertRtolEqual(cpu_output, npu_output)

    def test_l1_loss_float32_none(self):
        predict, target = self.generate_data(-2, 2, (4, 3), np.float32)
        cpu_output = self.cpu_op_exec(predict, target, "none")
        npu_output = self.npu_op_exec(predict, target, "none")
        self.assertEqual(cpu_output, npu_output)

    def test_l1_loss_float32_sum(self):
        predict, target = self.generate_data(-2, 2, (4, 3), np.float32)
        cpu_output = self.cpu_op_exec(predict, target, "sum")
        npu_output = self.npu_op_exec(predict, target, "sum")
        self.assertRtolEqual(cpu_output, npu_output, 1e-5)

    def test_l1_loss_float32_mean_large(self):
        predict, target = self.generate_data(-2, 2, (11, 1023, 1025), np.float32)
        cpu_output = self.cpu_op_exec(predict, target, "mean")
        npu_output = self.npu_op_exec(predict, target, "mean")
        self.assertRtolEqual(cpu_output, npu_output)


if __name__ == '__main__':
    run_tests()
