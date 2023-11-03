import numpy as np
import torch

import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor


class TestPSROIPooling(TestCase):

    def cal_sum(self, hstart, hend, wstart, wend, image):
        out_sum = 0.0
        for row in range(hstart, hend):
            for col in range(wstart, wend):
                out_sum += image[row][col]
        return out_sum

    def supported_op_exec(self, input1, rois, spatial_scale, group_size, output_dim):
        dst_type = input1.dtype
        if dst_type != torch.float32:
            input1 = input1.to(torch.float32)
            rois = rois.to(torch.float32)

        n, channels, height, width = input1.shape
        tensor_height = torch.tensor([height]).npu()
        tensor_width = torch.tensor([width]).npu()
        tensor_zero = torch.tensor([0]).npu()
        tensor_one_tenth = torch.tensor([0.1]).npu()

        output_size = [rois.size(0) * rois.size(2), output_dim, group_size, group_size]
        rois = rois.transpose(2, 1)
        rois = torch.reshape(rois, (rois.shape[0] * rois.shape[1], rois.shape[2]))
        output = torch.zeros(output_size).npu()
        for out_n in range(output_size[0]):
            for ctop in range(output_size[1]):
                for ph in range(output_size[2]):
                    for pw in range(output_size[3]):
                        roi_batch_ind = rois[out_n][0].to(int)
                        roi_start_w = torch.round(rois[out_n][1]) * spatial_scale
                        roi_start_h = torch.round(rois[out_n][2]) * spatial_scale
                        roi_end_w = torch.round(rois[out_n][3] + 1.0) * spatial_scale
                        roi_end_h = torch.round(rois[out_n][4] + 1.0) * spatial_scale
                        # Force too small ROIs to be 1x1
                        roi_width = torch.max(roi_end_w - roi_start_w, tensor_one_tenth)
                        roi_height = torch.max(roi_end_h - roi_start_h, tensor_one_tenth)
                        # Compute w and h at bottom
                        bin_size_h = roi_height / group_size
                        bin_size_w = roi_width / group_size

                        # Add roi offsets and clip to input boundaries
                        hstart = (ph * bin_size_h + roi_start_h).to(int)
                        wstart = torch.floor(pw * bin_size_w + roi_start_w).to(int)
                        hend = torch.ceil((ph + 1) * bin_size_h + roi_start_h).to(int)
                        wend = torch.ceil((pw + 1) * bin_size_w + roi_start_w).to(int)
                        hstart = torch.min(torch.max(hstart, tensor_zero), tensor_height).to(int)
                        hend = torch.min(torch.max(hend, tensor_zero), tensor_height).to(int)
                        wstart = torch.min(torch.max(wstart, tensor_zero), tensor_width).to(int)
                        wend = torch.min(torch.max(wend, tensor_zero), tensor_width).to(int)
                        is_empty = ((hend <= hstart) or (wend <= wstart))
                        c = (ctop * group_size + ph) * group_size + pw
                        out_sum = self.cal_sum(hstart,
                                               hend,
                                               wstart,
                                               wend,
                                               image=input1[roi_batch_ind][c])
                        bin_area = (hend - hstart) * (wend - wstart)
                        if is_empty:
                            output[out_n][ctop][ph][pw] = 0.0
                        else:
                            output[out_n][ctop][ph][pw] = out_sum / bin_area

        if dst_type != torch.float32:
            output = output.to(dst_type)
        return output.cpu().detach()

    def custom_op_exec(self, input1, rois, spatial_scale, group_size, output_dim):
        output = torch_npu.npu_ps_roi_pooling(input1, rois, spatial_scale, group_size, output_dim)
        return output.cpu().detach()

    def test_npu_ps_roi_pooling(self, device="npu"):
        item = [np.float32, 0, (2, 961, 127, 127)]
        _, npu_input = create_common_tensor(item, 0.1, 1)
        rois = torch.tensor([[[0], [3], [2], [9], [9]], [[1], [1], [4], [9], [9]]], dtype=torch.float32).npu()
        spatial_scale = 0.25
        group_size = 31
        output_dim = 1

        supported_output = self.supported_op_exec(npu_input, rois, spatial_scale, group_size, output_dim)
        custom_output = self.custom_op_exec(npu_input, rois, spatial_scale, group_size, output_dim)
        self.assertRtolEqual(supported_output, custom_output)


if __name__ == "__main__":
    run_tests()
