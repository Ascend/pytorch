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
from torch.nn import functional as F
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor


class TestAnchorResponseFlags(TestCase):
    def cpu_op_exec(self, gt_bboxes, featmap_size, strides, num_base_anchors):
        feat_h, feat_w = featmap_size
        gt_bboxes_cx = ((gt_bboxes[:, 0] + gt_bboxes[:, 2]) * 0.5)
        gt_bboxes_cy = ((gt_bboxes[:, 1] + gt_bboxes[:, 3]) * 0.5)
        gt_bboxes_grid_x = torch.floor(gt_bboxes_cx / strides[0]).int()
        gt_bboxes_grid_y = torch.floor(gt_bboxes_cy / strides[1]).int()
        gt_bboxes_grid_idx = gt_bboxes_grid_y * feat_w + gt_bboxes_grid_x
        responsible_grid = torch.zeros(feat_h * feat_w, dtype=torch.uint8)
        gt_bboxes_grid_idx = gt_bboxes_grid_idx.long()
        responsible_grid[gt_bboxes_grid_idx] = 1
        responsible_grid = responsible_grid[:, None].expand(
            responsible_grid.size(0), num_base_anchors).contiguous().view(-1)
        return responsible_grid.numpy()

    def npu_op_exec(self, input_npu, featmap_size, strides, num_base_anchors):
        out = torch_npu.npu_anchor_response_flags(input_npu, featmap_size, strides, num_base_anchors)
        out = out.to("cpu")
        return out.detach().numpy()
        
    def test_anchor_response_flags(self, device="npu"):
        shape_format = [
            [[np.float32, -1, [100, 4]], [60, 60], [2, 2], 9],
            [[np.float16, -1, [200, 4]], [10, 10], [32, 32], 3],
            [[np.float16, -1, [500, 4]], [32, 32], [16, 16], 5]
        ]
        for item in shape_format:
            cpu_input, npu_input = create_common_tensor(item[0], 0, 100)
            if cpu_input.dtype == torch.float16:
                cpu_input = cpu_input.to(torch.float32)
            cpu_output = self.cpu_op_exec(cpu_input, *item[1:])
            npu_output = self.npu_op_exec(npu_input, *item[1:])
            self.assertRtolEqual(cpu_output, npu_output)


if __name__ == "__main__":
    run_tests()
