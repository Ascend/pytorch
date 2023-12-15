# Copyright (c) 2020 Huawei Technologies Co., Ltd
# Copyright (c) 2019, Facebook CORPORATION.
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

import math
import torch
import numpy as np
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests


class TestNpuGiouBackward(TestCase):
    def generate_giou_data(self, n, m, dtype):
        data_bboxes1 = np.array([]).astype(dtype)
        for i in range(4):
            data_bboxes_array1 = i // 2 + math.pow(-1, i // 2) * 0.5 * np.random.rand(1, n).astype(dtype)
            data_bboxes1 = np.append(data_bboxes1, data_bboxes_array1)
        data_bboxes1 = data_bboxes1.reshape([4, n])
        data_gtboxes1 = np.array([]).astype(dtype)
        for i in range(4):
            data_gtboxes_array1 = i // 2 + math.pow(-1, i // 2) * 0.5 * np.random.rand(1, m).astype(dtype)
            data_gtboxes1 = np.append(data_gtboxes1, data_gtboxes_array1)
        data_gtboxes1 = data_gtboxes1.reshape([4, m])
        cpu_input1 = torch.from_numpy(data_bboxes1)
        cpu_input2 = torch.from_numpy(data_gtboxes1)
        npu_input1 = cpu_input1.npu()
        npu_input2 = cpu_input2.npu()
        list1 = [cpu_input1, cpu_input2, npu_input1, npu_input2]
        return list1

    def npu_op_exec(self, box1, box2, trans=False, is_cross=False, mode=0):
        box1.requires_grad = True
        box2.requires_grad = True
        output = torch_npu.npu_giou(box1, box2, trans, is_cross, mode)
        output.backward(torch.ones_like(output))
        box1_grad = box1.grad
        box2_grad = box2.grad
        box1_grad = box1_grad.detach().cpu().numpy()
        box2_grad = box2_grad.detach().cpu().numpy()
        output = output.detach().cpu().numpy()
        return output, box1_grad, box2_grad

    def test_npu_giou_backward_shape_format(self):
        np.random.seed(1234)
        shape_list1 = [
            [1, 1]
        ]
        is_trans_list1 = [True]
        mode_list1 = ["iou"]
        shape_format1 = [[j, k, m]
                         for j in shape_list1
                         for k in is_trans_list1
                         for m in mode_list1]

        for item in shape_format1:
            mode_digit = 0 if item[-1] == "iou" else 1
            is_cross = False if item[0][0] == item[0][1] else True
            expected_cpu_grad1 = np.array([[1.0218241],
                                           [-1.4181931],
                                           [0.18631615],
                                           [0.1747725]], dtype=np.float32)
            expected_cpu_grad2 = np.array([[-1.0218241],
                                           [1.4181931],
                                           [0.17999186],
                                           [0.23653218]], dtype=np.float32)
            list1 = self.generate_giou_data(*item[0], np.float32)
            _, npu_grad1, npu_grad2 = self.npu_op_exec(list1[2], list1[3], item[1], is_cross, mode_digit)
            self.assertRtolEqual(expected_cpu_grad1, npu_grad1)
            self.assertRtolEqual(expected_cpu_grad2, npu_grad2)


if __name__ == "__main__":
    run_tests()
