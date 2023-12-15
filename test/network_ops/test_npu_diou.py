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


class TestNpudiou(TestCase):
    def generate_diou_data(self, n, m, dtype):
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

    def cpu_op_exec(self, box1, box2, trans=False, is_cross=False, mode="iou", eps=1e-9):
        box3 = box1.numpy()
        dtype = box3.dtype
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

        diou_res = np.array([], dtype=dtype)
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
            if not is_cross:
                if i == j:
                    diou_res = np.append(diou_res, diou_ij)
            else:
                diou_res = np.append(diou_res, diou_ij)

        if not is_cross:
            res = diou_res.reshape(1, n)
        else:
            res = diou_res.reshape(n, m)
        return res

    def npu_op_exec(self, box1, box2, trans=False, is_cross=False, mode=0):
        output = torch_npu.npu_diou(box1, box2, trans, is_cross, mode)
        output = output.detach().cpu().numpy()
        return output

    def test_npu_diou_shape_format_fp32(self):
        self._test_npu_diou_shape_format(np.float32)

    def test_npu_diou_shape_format_fp16(self):
        self._test_npu_diou_shape_format(np.float16)

    def _test_npu_diou_shape_format(self, dtype):
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
            if dtype == np.float16:
                dtype = np.float32
            list1 = self.generate_diou_data(*item[0], dtype)
            cpu_output = self.cpu_op_exec(list1[0], list1[1], item[1], is_cross, item[-1])
            npu_output = self.npu_op_exec(list1[2], list1[3], item[1], is_cross, mode_digit)
            cpu_output = cpu_output.astype(npu_output.dtype)

            if dtype == np.float16:
                self.assertRtolEqual(cpu_output, npu_output, prec16=1e-2)
            else:
                self.assertRtolEqual(cpu_output, npu_output)


if __name__ == "__main__":
    run_tests()
