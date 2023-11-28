import unittest
import torch
import numpy as np
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor


class TestDeformableConv2dBackward(TestCase):
    def create_single_npu_tensor(self, item, minvalue, maxvalue):
        dtype = item[0]
        format1 = item[1]
        shape = item[2]
        input1 = np.random.uniform(minvalue, maxvalue, shape).astype(dtype)
        npu_input = torch.from_numpy(input1).to("npu")
        if format1 != -1:
            npu_input = torch_npu.npu_format_cast(npu_input, format1)
        return npu_input

    @unittest.skip("skip test_deformable_conv2d_backward_fp32 now")
    def test_deformable_conv2d_backward_fp32(self):
        np.random.seed(1234)
        input1 = self.create_single_npu_tensor([np.float32, 0, (16, 32, 32, 32)], 0, 10)
        weight = self.create_single_npu_tensor([np.float32, 0, (32, 32, 5, 5)], 0, 10)
        offset = self.create_single_npu_tensor([np.float32, 0, (16, 75, 32, 32)], 0, 10)
        input1.requires_grad = True
        weight.requires_grad = True
        offset.requires_grad = True

        npu_output, offset_out = torch_npu.npu_deformable_conv2d(
            input1, weight, offset, None, kernel_size=[5, 5], stride=[1, 1, 1, 1], padding=[2, 2, 2, 2])
        npu_output.backward(torch.ones_like(npu_output))
        npu_output1 = npu_output.cpu().detach()

        output = npu_output1.select(1, 2).select(1, 2).select(1, 2)
        expect_output = torch.tensor([65504., 65504., 65504., 65504., 65504., 65504., 65504., 65504., 65504.,
                                      65504., 65504., 65504., 65504., 65504., 65504., 65504.])
        self.assertRtolEqual(expect_output, output)

        offset_out_select = offset_out.select(1, 2).select(1, 2).select(1, 2)
        expect_offset_out = torch.tensor([13.6374, 6.0295, 79.6111, 33.5996, 53.6248, 62.2289, 14.3497, 47.6463,
                                          52.3312, 34.1246, 8.6705, 3.3515, 9.9513, 15.3604, 38.9772, 57.1306])
        self.assertRtolEqual(expect_offset_out, offset_out_select.cpu().detach())

        input_grad = input1.grad.select(1, 2).select(1, 2).select(1, 3)
        expect_input_grad = torch.tensor([1018.5208, 1080.2323, 2533.2463, 1305.0685, 3977.8293, 2363.5681,
                                          1414.5939, 2116.5427, 1401.0662, 2064.0400, 1945.2327, 2338.5208,
                                          300.2462, 2646.7798, 1899.1229, 2165.7280])
        self.assertRtolEqual(expect_input_grad, input_grad.cpu())

        offest_grad = offset.grad.select(1, 2).select(1, 2).select(1, 3)
        expect_offest_grad = torch.tensor([-4708.0259, -139.2554, -2387.8149, 31017.8438, 19861.9528,
                                           -1209.2686, -24085.7285, -3950.3850, -31044.7070, 4571.3936,
                                           582.9868, -5514.0459, 78401.6562, -1778.3700, -14311.4365,
                                           -2065.9717])
        self.assertRtolEqual(expect_offest_grad, offest_grad.cpu())

        weight_grad = weight.grad.select(1, 2).select(1, 2).select(1, 3)
        expect_weight_grad = torch.tensor([279501.8438, 279501.8438, 279501.8438, 279501.8438, 279501.8438,
                                           279501.8438, 279501.8438, 279501.8438, 279501.8750, 279501.8750,
                                           279501.8750, 279501.8750, 279501.8750, 279501.8750, 279501.8750,
                                           279501.8750, 279501.8438, 279501.8438, 279501.8438, 279501.8438,
                                           279501.8125, 279501.8125, 279501.8125, 279501.8125, 279501.8750,
                                           279501.8750, 279501.8750, 279501.8750, 279501.8750, 279501.8750,
                                           279501.8750, 279501.8750])
        self.assertRtolEqual(expect_weight_grad, weight_grad.cpu())

    @unittest.skip("skip test_deformable_conv2d_backward_fp16 now")
    def test_deformable_conv2d_backward_fp16(self):
        np.random.seed(1234)
        input_fp16 = self.create_single_npu_tensor([np.float16, 0, (16, 32, 32, 32)], 0, 10)
        weight = self.create_single_npu_tensor([np.float16, 0, (32, 32, 5, 5)], 0, 10)
        offset = self.create_single_npu_tensor([np.float16, 0, (16, 75, 32, 32)], 0, 10)
        input_fp16.requires_grad = True
        weight.requires_grad = True
        offset.requires_grad = True

        npu_output, offset_out = torch_npu.npu_deformable_conv2d(
            input_fp16, weight, offset, None, kernel_size=[5, 5], stride=[1, 1, 1, 1], padding=[2, 2, 2, 2])
        npu_output.backward(torch.ones_like(npu_output))
        npu_output1 = npu_output.cpu().detach()

        output = npu_output1.select(1, 2).select(1, 2).select(1, 2)
        expect_output = torch.tensor([65504., 65504., 65504., 65504., 65504., 65504., 65504., 65504., 65504.,
                                      65504., 65504., 65504., 65504., 65504., 65504., 65504.], dtype=torch.float16)
        self.assertRtolEqual(expect_output, output)

        offset_out_select = offset_out.select(1, 2).select(1, 2).select(1, 2)
        expect_offset_out = torch.tensor([13.6562, 6.0352, 79.4375, 33.5938, 53.6875, 62.2188, 14.3438, 47.6562,
                                          52.3750, 34.0938, 8.6797, 3.3516, 9.9531, 15.3750, 39.0625, 57.1875],
                                         dtype=torch.float16)
        self.assertRtolEqual(expect_offset_out, offset_out_select.cpu().detach())

        input_grad = input_fp16.grad.select(1, 2).select(1, 2).select(1, 3)
        expect_input_grad = torch.tensor([1019.0000, 1082.0000, 2534.0000, 1304.0000, 3978.0000, 2364.0000,
                                          1413.0000, 2114.0000, 1402.0000, 2064.0000, 1944.0000, 2340.0000,
                                          299.7500, 2648.0000, 1895.0000, 2164.0000], dtype=torch.float16)
        self.assertRtolEqual(expect_input_grad, input_grad.cpu())

        offest_grad = offset.grad.select(1, 2).select(1, 2).select(1, 2)
        expect_offest_grad = torch.tensor([141.2500, 34784.0000, -12384.0000, -1885.0000, -5440.0000,
                                           4416.0000, 13920.0000, -19952.0000, 4160.0000, 24848.0000,
                                           -1464.0000, -21088.0000, -1060.0000, -22544.0000, 9152.0000,
                                           -4312.0000], dtype=torch.float16)
        self.assertRtolEqual(expect_offest_grad, offest_grad.cpu())

        weight_grad = weight.grad.select(1, 2).select(1, 2).select(1, 3)
        expect_weight_grad = torch.tensor([65504., 65504., 65504., 65504., 65504., 65504., 65504., 65504., 65504.,
                                           65504., 65504., 65504., 65504., 65504., 65504., 65504., 65504., 65504.,
                                           65504., 65504., 65504., 65504., 65504., 65504., 65504., 65504., 65504.,
                                           65504., 65504., 65504., 65504., 65504.], dtype=torch.float16)
        self.assertRtolEqual(expect_weight_grad, weight_grad.cpu())


if __name__ == "__main__":
    run_tests()
