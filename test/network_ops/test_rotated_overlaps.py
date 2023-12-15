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

import torch
import numpy as np
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests


class TestRotatedOverlaps(TestCase):
    def generate_rto_data(self, item):
        np.random.seed(1234)
        min_value, max_value = 30, 60
        scope = 20
        dtype = item[0][0]
        shape_one = item[0][-1]
        shape_two = item[1][-1]

        boxes_center = np.random.uniform(min_value, max_value, shape_one[:2] + [2]).astype(dtype)
        boxes_wh = np.random.randint(1, scope, size=shape_one[:2] + [2])
        boxes_angle = np.random.randint(-180, 180, size=shape_one[:2] + [1])
        boxes = np.concatenate([boxes_center, boxes_wh, boxes_angle], axis=-1, dtype=dtype)
        # query_boxes
        query_boxes_center = np.random.uniform(min_value, max_value, shape_two[:2] + [2]).astype(dtype)
        query_boxes_wh = np.random.randint(1, scope, size=shape_two[:2] + [2])
        query_boxes_angle = np.random.randint(-180, 180, size=shape_two[:2] + [1])
        query_boxes = np.concatenate([query_boxes_center, query_boxes_wh, query_boxes_angle], axis=-1, dtype=dtype)

        cpu_input1 = torch.from_numpy(boxes)
        cpu_input2 = torch.from_numpy(query_boxes)
        npu_input1 = cpu_input1.npu()
        npu_input2 = cpu_input2.npu()
        return npu_input1, npu_input2

    def cpu_expect_result(self, dtype):
        if dtype == np.float16:
            output = np.array([[[0., 13.27, 1.022, 0.],
                                [0., 0., 54.12, 0.],
                                [0., 0., 0., 19.17]]], dtype=np.float16)
        else:
            output = np.array([[[0., 10.289731],
                                [0., 0.],
                                [0., 0.]]], dtype=np.float32)
        return output

    def npu_op_exec(self, box1, box2, trans=False):
        output = torch_npu.npu_rotated_overlaps(box1, box2, trans)
        output = output.detach().cpu().numpy()
        return output

    def test_rotated_overlaps_shape_format_fp32(self, device="npu"):
        dtype = np.float32
        shape_list = [
            [[1, 3, 5], [1, 2, 5]],
        ]
        is_trans_list = [False]
        shape_format = [[[dtype, -1, m[0]], [dtype, -1, m[1]], k]
                        for m in shape_list
                        for k in is_trans_list]

        for item in shape_format:
            npu_input1, npu_input2 = self.generate_rto_data(item[:-1])
            cpu_output = self.cpu_expect_result(dtype)
            npu_output = self.npu_op_exec(npu_input1, npu_input2, item[-1])
            # fp32 has't enough precission, but match model need currently.
            self.assertRtolEqual(cpu_output, npu_output, prec=0.00005)

    def test_rotated_overlaps_shape_format_fp16(self, device="npu"):
        dtype = np.float16
        shape_list = [
            [[1, 3, 5], [1, 4, 5]],
        ]
        # true is xyxyt, false is xywh format
        is_trans_list = [False]
        shape_format = [[[dtype, -1, m[0]], [dtype, -1, m[1]], k]
                        for m in shape_list
                        for k in is_trans_list]
        for item in shape_format:
            npu_input1, npu_input2 = self.generate_rto_data(item)
            cpu_output = self.cpu_expect_result(dtype)
            npu_output = self.npu_op_exec(npu_input1, npu_input2, item[-1])
            self.assertRtolEqual(cpu_output, npu_output)


if __name__ == "__main__":
    run_tests()
