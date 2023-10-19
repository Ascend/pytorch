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
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests


class TestRotatedBox(TestCase):
    def npu_op_encode_exec(self, anchor_boxes, gt_bboxes, weight):
        out = torch_npu.npu_rotated_box_encode(anchor_boxes, gt_bboxes, weight)
        out = out.to("cpu")
        return out.detach().numpy()

    def npu_op_decode_exec(self, anchor_boxes, deltas, weight):
        out = torch_npu.npu_rotated_box_decode(anchor_boxes, deltas, weight)
        out = out.to("cpu")
        return out.detach().numpy()

    def test_rotated_boxes_encode_fp32(self, device="npu"):
        anchor_boxes = torch.tensor([[[44.2877], [9.1412], [88.7575], [25.8879], [64.8047]]]).to("npu")
        gt_bboxes = torch.tensor([[[39.1763], [0.9838], [78.1028], [29.5997], [51.5907]]]).to("npu")
        weight = torch.tensor([1., 1., 1., 1., 1.]).npu()
        expect_cpu = torch.tensor([[[-0.1773], [-0.1327], [-0.1331], [0.5358], [-0.8643]]])
        npu_output = self.npu_op_encode_exec(anchor_boxes, gt_bboxes, weight)
        self.assertRtolEqual(expect_cpu.numpy(), npu_output)

    def test_rotated_boxes_decode_fp32(self, device="npu"):
        anchor_boxes = torch.tensor([[[32.1855], [41.9922], [64.1435], [62.5325], [34.607]]]).to("npu")
        deltas = torch.tensor([[[1.8725], [-1.8915], [0.2395], [-0.4622], [-34.6539]]]).to("npu")
        weight = torch.tensor([1., 1., 1., 1., 1.]).npu()
        expect_cpu = torch.tensor([[[87.70366], [6.9412346], [128.31055], [19.879467], [-88.313515]]])
        npu_output = self.npu_op_decode_exec(anchor_boxes, deltas, weight)
        self.assertRtolEqual(expect_cpu.numpy(), npu_output)

    def test_rotated_boxes_encode_fp16(self, device="npu"):
        anchor_boxes = torch.tensor([[[30.69], [32.6], [45.94], [59.88], [-44.53]]], dtype=torch.float16).to("npu")
        gt_bboxes = torch.tensor([[[30.44], [18.72], [33.22], [45.56], [8.5]]], dtype=torch.float16).to("npu")
        weight = torch.tensor([1., 1., 1., 1., 1.], dtype=torch.float16).npu()
        expect_cpu = torch.tensor([[[-0.4253], [-0.5166], [-1.702], [-0.0162], [1.133]]], dtype=torch.float16)
        npu_output = self.npu_op_encode_exec(anchor_boxes, gt_bboxes, weight)
        self.assertRtolEqual(expect_cpu.numpy(), npu_output)

    def test_rotated_boxes_decode_fp16(self, device="npu"):
        anchor_boxes = torch.tensor([[[4.137], [33.72], [29.4], [54.06], [41.28]]], dtype=torch.float16).to("npu")
        deltas = torch.tensor([[[0.0244], [-1.992], [0.2109], [0.315], [-37.25]]], dtype=torch.float16).to("npu")
        weight = torch.tensor([1., 1., 1., 1., 1.], dtype=torch.float16).npu()
        expect_cpu = torch.tensor([[[1.786], [-10.58], [33.], [17.3], [-88.44]]], dtype=torch.float16)
        npu_output = self.npu_op_decode_exec(anchor_boxes, deltas, weight)
        self.assertRtolEqual(expect_cpu.numpy(), npu_output)


if __name__ == "__main__":
    run_tests()
