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


class TestYoloBoxesEncode(TestCase):
    def npu_op_exec(self, anchor_boxes, gt_bboxes, stride, impl_mode=False):
        out = torch_npu.npu_yolo_boxes_encode(anchor_boxes, gt_bboxes, stride, impl_mode)
        out = out.to("cpu")
        return out.detach().numpy()

    def test_yolo_boxes_encode(self, device="npu"):
        torch.manual_seed(1234)
        anchor_boxes_list = [(2, 4)]
        gt_bboxes_list = [(2, 4)]
        stride_list = [[2, 2]]
        expect_cpu_list = [[[0.7921727, 0.5314963, -0.74224466, -13.815511],
                            [0.7360072, 0.58343244, 4.3334002, -0.51378196]]]

        shape_format = [[i, j, k, h] for i in anchor_boxes_list
                        for j in gt_bboxes_list for k in stride_list for h in expect_cpu_list]

        for item in shape_format:
            anchor_boxes_tensor = torch.rand(item[0], dtype=torch.float32).to("npu")
            gt_bboxes_tensor = torch.rand(item[1], dtype=torch.float32).to("npu")
            stride_tensor = torch.tensor(item[2], dtype=torch.int32).to("npu")
            expect_cpu_tensor = torch.tensor(item[3], dtype=torch.float32)
            npu_output = self.npu_op_exec(anchor_boxes_tensor, gt_bboxes_tensor, stride_tensor, False)

            self.assertRtolEqual(expect_cpu_tensor.numpy(), npu_output)


if __name__ == "__main__":
    run_tests()
