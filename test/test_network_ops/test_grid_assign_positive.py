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


class TestGridAssignPositive(TestCase):
    def npu_op_exec(self, *args):
        out = torch_npu.npu_grid_assign_positive(*args)
        out = out.to("cpu")
        return out.detach().numpy()

    def test_grid_assign_positive(self, device="npu"):
        torch.manual_seed(1234)
        assigned_gt_inds = torch.rand((4,), dtype=torch.float32).to("npu")
        overlaps = torch.rand((2, 4), dtype=torch.float32).to("npu")
        box_responsible_flags = torch.tensor([1, 1, 1, 0], dtype=torch.uint8).to("npu")
        max_overlap = torch.rand((4,), dtype=torch.float32).to("npu")
        argmax_overlap = torch.tensor([1, 0, 1, 0], dtype=torch.int32).to("npu")
        gt_max_overlaps = torch.rand((2,), dtype=torch.float32).to("npu")
        gt_argmax_overlaps = torch.tensor([1, 0], dtype=torch.int32).to("npu")
        inputs = [assigned_gt_inds, overlaps, box_responsible_flags, max_overlap,
                  argmax_overlap, gt_max_overlaps, gt_argmax_overlaps]
        num_gts = 128
        pos_iou_thr = .5
        min_pos_iou = .0
        gt_max_assign_all = True
        attrs = [num_gts, pos_iou_thr, min_pos_iou, gt_max_assign_all]

        params = inputs + attrs
        expect_cpu = torch.tensor([2., 1., 0.25984418, 0.36664134], dtype=torch.float32)
        npu_output = self.npu_op_exec(*params)
        self.assertRtolEqual(expect_cpu.numpy(), npu_output)


if __name__ == "__main__":
    run_tests()
