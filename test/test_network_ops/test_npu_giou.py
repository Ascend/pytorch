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


class TestNpuGiou(TestCase):
    def generate_giou_data(self, n, m, dtype):
        data_bboxes = np.array([]).astype(dtype)
        for i in range(4):
            data_bboxes_array = i // 2 + math.pow(-1, i // 2) * 0.5 * np.random.rand(1, n).astype(dtype)
            data_bboxes = np.append(data_bboxes, data_bboxes_array)
        data_bboxes = data_bboxes.reshape([4, n])
        data_gtboxes = np.array([]).astype(dtype)
        for i in range(4):
            data_gtboxes_array = i // 2 + math.pow(-1, i // 2) * 0.5 * np.random.rand(1, m).astype(dtype)
            data_gtboxes = np.append(data_gtboxes, data_gtboxes_array)
        data_gtboxes = data_gtboxes.reshape([4, m])
        cpu_input1 = torch.from_numpy(data_bboxes)
        cpu_input2 = torch.from_numpy(data_gtboxes)
        npu_input1 = cpu_input1.npu()
        npu_input2 = cpu_input2.npu()
        list1 = [cpu_input1, cpu_input2, npu_input1, npu_input2]
        return list1

    def cpu_op_exec(self, box1, box2, trans=False, is_cross=False, mode="iou"):
        box1 = box1.numpy()
        box2 = box2.numpy()
        dtype = box1.dtype
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
        w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1
        w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1
        area1 = w1 * h1
        area2 = w2 * h2
        giou_res = np.array([], dtype=dtype)

        iter_list = itertools.product(list(range(n)), list(range(m)))
        for i, j in iter_list:
            inter_x1 = max(b1_x1[i], b2_x1[j])
            inter_x2 = min(b1_x2[i], b2_x2[j])
            inter_y1 = max(b1_y1[i], b2_y1[j])
            inter_y2 = min(b1_y2[i], b2_y2[j])
            outer_x1 = min(b1_x1[i], b2_x1[j])
            outer_x2 = max(b1_x2[i], b2_x2[j])
            outer_y1 = min(b1_y1[i], b2_y1[j])
            outer_y2 = max(b1_y2[i], b2_y2[j])
            inter_area = max(0, (inter_x2 - inter_x1)) * max(0, (inter_y2 - inter_y1))
            outer_area = abs(outer_x2 - outer_x1) * abs(outer_y2 - outer_y1)
            union_area = area1[i] + area2[j] - inter_area + 1e-16
            other_area = outer_area - union_area
            giou_ij = inter_area / union_area - other_area / outer_area
            if not is_cross:
                if i == j:
                    giou_res = np.append(giou_res, giou_ij)
            else:
                giou_res = np.append(giou_res, giou_ij)

        if not is_cross:
            res = giou_res.reshape(1, n)
        else:
            res = giou_res.reshape(n, m)
            res = np.transpose(res)
        res = np.transpose(res)
        return res

    def npu_op_exec(self, box1, box2, trans=False, is_cross=False, mode=0):
        output = torch_npu.npu_giou(box1, box2, trans, is_cross, mode)
        output = output.detach().cpu().numpy()
        return output

    def test_npu_giou_shape_format_fp32(self):
        self._test_npu_giou_shape_format(np.float32)

    def test_npu_giou_shape_format_fp16(self):
        self._test_npu_giou_shape_format(np.float16)

    def _test_npu_giou_shape_format(self, dtype):
        shape_list = [
            [10, 10],
            [12, 12],
            [100, 100]
        ]
        is_trans_list = [True]
        mode_list = ["iou"]
        shape_format = [[j, k, m]
                        for j in shape_list
                        for k in is_trans_list
                        for m in mode_list]

        for item in shape_format:
            mode_digit = 0 if item[-1] == "iou" else 1
            is_cross = False if item[0][0] == item[0][1] else True
            list1 = self.generate_giou_data(*item[0], dtype)
            cpu_output = self.cpu_op_exec(list1[0], list1[1], item[1], is_cross, item[-1])
            npu_output = self.npu_op_exec(list1[2], list1[3], item[1], is_cross, mode_digit)
            cpu_output = cpu_output.astype(npu_output.dtype)
            if dtype == np.float16:
                self.assertRtolEqual(cpu_output, npu_output, prec16=1e-2)
            else:
                self.assertRtolEqual(cpu_output, npu_output)


if __name__ == "__main__":
    run_tests()
