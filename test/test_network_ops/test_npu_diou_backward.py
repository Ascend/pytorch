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

import itertools
import math
import torch
import numpy as np
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests


class TestNpuDiouBackward(TestCase):
    def generate_diou_data(self, n, m, dtype):
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

    def cpu_op_exec(self, box1, box2, trans=False, is_cross=False, mode="iou", eps=1e-9):
        if box1.dtype == torch.half:
            box1 = box1.astype(torch.float32)
            box2 = box2.astype(torch.float32)
        _, n = box1.shape
        _, m = box2.shape
        if trans:
            b1_x1, b1_x2 = box1[0] - box1[2] / 2, box1[0] + box1[2] / 2
            b1_y1, b1_y2 = box1[1] - box1[3] / 2, box1[1] + box1[3] / 2
            b2_x1, b2_x2 = box2[0] - box2[2] / 2, box2[0] + box2[2] / 2
            b2_y1, b2_y2 = box2[1] - box2[3] / 2, box2[1] + box2[3] / 2
        else:
            b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
            b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]

        iter_list = itertools.product(list(range(n)), list(range(m)))
        for i, j in iter_list:
            cw = torch.max(b1_x2[i], b2_x2[j]) - torch.min(b1_x1[i], b2_x1[j])
            ch = torch.max(b1_y2[i], b2_y2[j]) - torch.min(b1_y1[i], b2_y1[j])
            c2 = cw ** 2 + ch ** 2 + eps
            rho2 = ((b2_x1[i] + b2_x2[j] - b1_x1[i] - b1_x2[j]) ** 2 +
                    (b2_y1[i] + b2_y2[j] - b1_y1[i] - b1_y2[j]) ** 2) / 4
            inter_area = (torch.min(b1_x2[i], b2_x2[j]) - torch.max(b1_x1[i], b2_x1[j])).clamp(0) * \
                         (torch.min(b1_y2[i], b2_y2[j]) - torch.max(b1_y1[i], b2_y1[j])).clamp(0)
            w1, h1 = b1_x2[i] - b1_x1[i], b1_y2[i] - b1_y1[i] + eps
            w2, h2 = b2_x2[j] - b2_x1[j], b2_y2[j] - b2_y1[j] + eps
            union_area = w1 * h1 + w2 * h2 - inter_area + eps
            diou_ij = inter_area / union_area - (rho2 / c2)
            return diou_ij

    def cpu_in(self, box1, box2, trans, is_cross, mode):
        box1.requires_grad = True
        box2.requires_grad = True
        diou_ij = self.cpu_op_exec(box1, box2, trans, is_cross, mode)
        diou_ij.backward(torch.ones_like(diou_ij))
        box1_grad = box1.grad
        box2_grad = box2.grad
        box1_grad = box1_grad.numpy()
        box2_grad = box2_grad.numpy()
        return diou_ij, box1_grad, box2_grad

    def npu_op_exec(self, box1, box2, trans=False, is_cross=False, mode=0):
        box1.requires_grad = True
        box2.requires_grad = True
        output = torch_npu.npu_diou(box1, box2, trans, is_cross, mode)
        output.backward(torch.ones_like(output))
        box1_grad = box1.grad
        box2_grad = box2.grad
        box1_grad = box1_grad.detach().cpu().numpy()
        box2_grad = box2_grad.detach().cpu().numpy()
        output = output.detach().cpu().numpy()
        return output, box1_grad, box2_grad

    def test_npu_diou_backward_shape_format(self):
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
            list1 = self.generate_diou_data(*item[0], np.float32)
            _, cpu_grad1, cpu_grad2 = self.cpu_in(list1[0], list1[1], item[1], is_cross, item[-1])
            _, npu_grad1, npu_grad2 = self.npu_op_exec(list1[2], list1[3], item[1], is_cross, mode_digit)
            self.assertRtolEqual(cpu_grad1, npu_grad1)
            self.assertRtolEqual(cpu_grad2, npu_grad2)


if __name__ == "__main__":
    run_tests()
