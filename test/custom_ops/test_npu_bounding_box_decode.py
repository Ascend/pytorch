import unittest
import torch

import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests


class TestBoundingBoxDecode(TestCase):
    def npu_bounding_box_decode(self, rois, deltas, means0, means1, means2, means3,
                                stds0, stds1, stds2, stds3, max_shape, wh_ratio_clip):
        means = [means0, means1, means2, means3]
        stds = [stds0, stds1, stds2, stds3]
        means = deltas.new_tensor(means).repeat(1, deltas.size(1) // 4)
        stds = deltas.new_tensor(stds).repeat(1, deltas.size(1) // 4)
        denorm_deltas = deltas * stds + means

        dx = denorm_deltas[:, 0::4]
        dy = denorm_deltas[:, 1::4]
        dw = denorm_deltas[:, 2::4]
        dh = denorm_deltas[:, 3::4]
        max_ratio = torch.abs(torch.log(torch.tensor(wh_ratio_clip)))

        dw = torch.clamp(dw, min=-max_ratio, max=max_ratio)
        dh = torch.clamp(dh, min=-max_ratio, max=max_ratio)

        ax = ((rois[:, 0] + rois[:, 2]) * 0.5).unsqueeze(1).expand_as(dx)
        ay = ((rois[:, 1] + rois[:, 3]) * 0.5).unsqueeze(1).expand_as(dy)
        aw = (rois[:, 2] - rois[:, 0] * 0.5).unsqueeze(1).expand_as(dw)
        ah = (rois[:, 3] - rois[:, 1] * 0.5).unsqueeze(1).expand_as(dh)

        pw = aw * dw.exp()
        ph = ah * dh.exp()
        px = torch.addcmul(ax, 1, aw, dx)
        py = torch.addcmul(ay, 1, ah, dy)

        x1 = px - pw * 0.5 + 0.5
        y1 = py - ph * 0.5 + 0.5
        x2 = px + pw * 0.5 - 0.5
        y2 = py + ph * 0.5 - 0.5

        if max_shape is not None:
            x1 = torch.clamp(x1, min=0, max=(max_shape[1] - 1))
            y1 = torch.clamp(y1, min=0, max=(max_shape[0] - 1))
            x2 = torch.clamp(x2, min=0, max=(max_shape[1] - 1))
            y2 = torch.clamp(y2, min=0, max=(max_shape[0] - 1))
        boxes = torch.stack([x1, y1, x2, y2], dim=-1).view_as(deltas)
        return boxes

    def custom_op_exec(self, rois, deltas, means0, means1, means2, means3,
                       stds0, stds1, stds2, stds3, max_shape, wh_ratio_clip):
        output = self.npu_bounding_box_decode(rois, deltas, means0, means1,
                                              means2, means3, stds0, stds1,
                                              stds2, stds3, max_shape, wh_ratio_clip)
        output = output.to("cpu")
        output = output.numpy()
        return output

    def npu_op_exec(self, rois, deltas, means0, means1, means2, means3,
                    stds0, stds1, stds2, stds3, max_shape, wh_ratio_clip):
        output = torch_npu.npu_bounding_box_decode(rois, deltas, means0, means1,
                                                   means2, means3, stds0, stds1,
                                                   stds2, stds3, max_shape, wh_ratio_clip)
        output = output.to("cpu")
        output = output.numpy()
        return output

    @unittest.skip("skip test_decode_shape_format_fp32 now")
    def test_decode_shape_format_fp32(self):
        input1 = torch.tensor([[1., 2., 3., 4.], [3., 4., 5., 6.]],
                              dtype=torch.float32).to("npu")
        input2 = torch.tensor([[5., 6., 7., 8.], [7., 8., 9., 6.]],
                              dtype=torch.float32).to("npu")

        npu_output = self.npu_op_exec(input1, input2, 0, 0, 0, 0,
                                      1, 1, 1, 1, (10, 10), 0.1)
        custom_output = self.custom_op_exec(input1, input2, 0, 0, 0, 0,
                                            1, 1, 1, 1, (10, 10), 0.1)
        self.assertRtolEqual(npu_output, custom_output)

    @unittest.skip("skip test_decode_shape_format_fp16 now")
    def test_decode_shape_format_fp16(self):
        input1_fp16 = torch.tensor([[1., 2., 3., 4.], [3., 4., 5., 6.]],
                                   dtype=torch.float16).to("npu")
        input2_fp16 = torch.tensor([[5., 6., 7., 8.], [7., 8., 9., 6.]],
                                   dtype=torch.float16).to("npu")

        npu_output = self.npu_op_exec(input1_fp16, input2_fp16, 0, 0, 0, 0,
                                      1, 1, 1, 1, (10, 10), 0.1)
        custom_output = self.custom_op_exec(input1_fp16, input2_fp16, 0, 0, 0, 0,
                                            1, 1, 1, 1, (10, 10), 0.1)
        self.assertRtolEqual(npu_output, custom_output)


if __name__ == "__main__":
    run_tests()
