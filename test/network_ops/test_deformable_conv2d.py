import unittest
import torch
import numpy as np
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor


class TestDeformableConv2d(TestCase):
    def create_single_npu_tensor(self, item, minvalue, maxvalue):
        dtype = item[0]
        format1 = item[1]
        shape = item[2]
        input1 = np.random.uniform(minvalue, maxvalue, shape).astype(dtype)
        npu_input = torch.from_numpy(input1).to("npu")
        if format1 != -1:
            npu_input = torch_npu.npu_format_cast(npu_input, format1)
        return npu_input

    @unittest.skip("skip test_deformable_conv2d_fp32 now")
    def test_deformable_conv2d_fp32(self):
        np.random.seed(1234)
        input1 = self.create_single_npu_tensor([np.float32, 0, (16, 32, 32, 32)], 0, 10)
        weight = self.create_single_npu_tensor([np.float32, 0, (32, 32, 5, 5)], 0, 10)
        offset = self.create_single_npu_tensor([np.float32, 0, (16, 75, 32, 32)], 0, 10)

        npu_output, offset_out = torch_npu.npu_deformable_conv2d(
            input1, weight, offset, None, kernel_size=[5, 5], stride=[1, 1, 1, 1], padding=[2, 2, 2, 2])
        npu_output = npu_output.cpu().detach()

        output = npu_output.select(1, 2).select(1, 2).select(1, 2)
        expect_output = torch.tensor([65504., 65504., 65504., 65504., 65504., 65504., 65504., 65504., 65504.,
                                      65504., 65504., 65504., 65504., 65504., 65504., 65504.])
        self.assertRtolEqual(expect_output, output)

        offset_out_select = offset_out.select(1, 2).select(1, 2).select(1, 2)
        expect_offset_out = torch.tensor([13.6374, 6.0295, 79.6111, 33.5996, 53.6248, 62.2289, 14.3497, 47.6463,
                                          52.3312, 34.1246, 8.6705, 3.3515, 9.9513, 15.3604, 38.9772, 57.1306])
        self.assertRtolEqual(expect_offset_out, offset_out_select.cpu().detach())

    @unittest.skip("skip test_deformable_conv2d_fp16 now")
    def test_deformable_conv2d_fp16(self):
        np.random.seed(1234)
        input_fp16 = self.create_single_npu_tensor([np.float16, 0, (16, 32, 32, 32)], 0, 10)
        weight = self.create_single_npu_tensor([np.float16, 0, (32, 32, 5, 5)], 0, 10)
        offset = self.create_single_npu_tensor([np.float16, 0, (16, 75, 32, 32)], 0, 10)

        npu_output, offset_out = torch_npu.npu_deformable_conv2d(
            input_fp16, weight, offset, None, kernel_size=[5, 5], stride=[1, 1, 1, 1], padding=[2, 2, 2, 2])
        npu_output = npu_output.cpu().detach()

        output = npu_output.select(1, 2).select(1, 2).select(1, 2)
        expect_output = torch.tensor([65504., 65504., 65504., 65504., 65504., 65504., 65504., 65504., 65504.,
                                      65504., 65504., 65504., 65504., 65504., 65504., 65504.], dtype=torch.float16)
        self.assertRtolEqual(expect_output, output)

        offset_out_select = offset_out.select(1, 2).select(1, 2).select(1, 2)
        expect_offset_out = torch.tensor([13.6562, 6.0352, 79.4375, 33.5938, 53.6875, 62.2188, 14.3438, 47.6562,
                                          52.3750, 34.0938, 8.6797, 3.3516, 9.9531, 15.3750, 39.0625, 57.1875],
                                         dtype=torch.float16)
        self.assertRtolEqual(expect_offset_out, offset_out_select.cpu().detach())


if __name__ == "__main__":
    run_tests()
