import torch

import torch_npu
from torch_npu.testing.common_utils import SupportedDevices
from torch_npu.testing.testcase import TestCase, run_tests


class TestBoundingBoxEncode(TestCase):
    def npu_bounding_box_encode(self, anchor_box, ground_truth_box, means0, means1,
                                means2, means3, stds0, stds1, stds2, stds3):
        means = [means0, means1, means2, means3]
        stds = [stds0, stds1, stds2, stds3]
        px = (anchor_box[..., 0] + anchor_box[..., 2]) * 0.5
        py = (anchor_box[..., 1] + anchor_box[..., 3]) * 0.5
        pw = anchor_box[..., 2] - anchor_box[..., 0] + 1.0
        ph = anchor_box[..., 3] - anchor_box[..., 1] + 1.0

        gx = (ground_truth_box[..., 0] + ground_truth_box[..., 2]) * 0.5
        gy = (ground_truth_box[..., 1] + ground_truth_box[..., 3]) * 0.5
        gw = ground_truth_box[..., 2] - ground_truth_box[..., 0] + 1.0
        gh = ground_truth_box[..., 3] - ground_truth_box[..., 1] + 1.0

        eps = 1e-7
        dx = (gx - px) / (pw + eps)
        dy = (gy - py) / (ph + eps)
        dw = torch.log(torch.abs(gw) / torch.abs(pw + eps))
        dh = torch.log(torch.abs(gh) / torch.abs(ph + eps))
        deltas = torch.stack([dx, dy, dw, dh], dim=-1)

        means = deltas.new_tensor(means).unsqueeze(0)
        stds = deltas.new_tensor(stds).unsqueeze(0)
        deltas = deltas.sub_(means) .div_(stds)

        return deltas

    def custom_op_exec(self, anchor_box, ground_truth_box, means0, means1,
                       means2, means3, stds0, stds1, stds2, stds3):
        output = self.npu_bounding_box_encode(anchor_box, ground_truth_box, means0, means1,
                                              means2, means3, stds0, stds1, stds2, stds3)
        output = output.to("cpu")
        output = output.numpy()
        return output

    def npu_op_exec(self, anchor_box, ground_truth_box, means0, means1,
                    means2, means3, stds0, stds1, stds2, stds3):
        output = torch_npu.npu_bounding_box_encode(anchor_box, ground_truth_box, means0, means1,
                                                   means2, means3, stds0, stds1, stds2, stds3)
        output = output.to("cpu")
        output = output.numpy()
        return output

    @SupportedDevices(['Ascend910A', 'Ascend910P'])
    def test_encode_shape_format_fp32(self):
        input1 = torch.tensor([[1., 2., 3., 4.], [3., 4., 5., 6.]],
                              dtype=torch.float32).to("npu")
        input2 = torch.tensor([[5., 6., 7., 8.], [7., 8., 9., 6.]],
                              dtype=torch.float32).to("npu")

        npu_output = self.npu_op_exec(input1, input2, 0, 0, 0, 0,
                                      0.1, 0.1, 0.2, 0.2)
        custom_output = self.custom_op_exec(input1, input2, 0, 0, 0, 0,
                                            0.1, 0.1, 0.2, 0.2)
        self.assertRtolEqual(npu_output, custom_output, 1e-3)

    @SupportedDevices(['Ascend910A', 'Ascend910P'])
    def test_encode_shape_format_fp16(self):
        input1_fp16 = torch.tensor([[1., 2., 3., 4.], [3., 4., 5., 6.]],
                                   dtype=torch.float16).to("npu")
        input2_fp16 = torch.tensor([[5., 6., 7., 8.], [7., 8., 9., 6.]],
                                   dtype=torch.float16).to("npu")

        npu_output = self.npu_op_exec(input1_fp16, input2_fp16, 0, 0, 0, 0,
                                      0.1, 0.1, 0.2, 0.2)
        custom_output = self.custom_op_exec(input1_fp16, input2_fp16, 0, 0, 0, 0,
                                            0.1, 0.1, 0.2, 0.2)
        self.assertRtolEqual(npu_output, custom_output, 1e-3)


if __name__ == "__main__":
    run_tests()
