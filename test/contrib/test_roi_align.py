import unittest
import torch
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.contrib.module import ROIAlign


class TestRoiAlign(TestCase):

    def npu_roi_align(self, input1, roi, output_size, spatial_scale, sampling_ratio, aligned):

        input1.requires_grad_(True)
        roi.requires_grad_(True)
        model = ROIAlign(output_size, spatial_scale, sampling_ratio, aligned=aligned).npu()
        output = model(input1, roi)
        output.sum().backward()
        return output.detach().cpu(), input1.grad.cpu()

    def generate_input(self):

        input1 = torch.FloatTensor([[[[1, 2, 3, 4, 5, 6],
                                      [7, 8, 9, 10, 11, 12],
                                      [13, 14, 15, 16, 17, 18],
                                      [19, 20, 21, 22, 23, 24],
                                      [25, 26, 27, 28, 29, 30],
                                      [31, 32, 33, 34, 35, 36]]]]).npu()
        return input1

    def test_npu_roi_align_1(self):

        input1 = self.generate_input()
        roi = torch.tensor([[0, -2.0, -2.0, 22.0, 22.0]]).npu()
        output_size = (3, 3)
        spatial_scale = 0.25
        sampling_ratio = 2
        aligned = False

        npu_output, npu_inputgrad = self.npu_roi_align(input1, roi, output_size, spatial_scale, sampling_ratio, aligned)
        expedt_cpu_output = torch.tensor([[[[4.5000, 6.5000, 8.5000],
                                            [16.5000, 18.5000, 20.5000],
                                            [28.5000, 30.5000, 32.5000]]]],
                                         dtype=torch.float32)
        expedt_cpu_inputgrad = torch.tensor([[[[0.2500, 0.2500, 0.2500, 0.2500, 0.2500, 0.2500],
                                               [0.2500, 0.2500, 0.2500, 0.2500, 0.2500, 0.2500],
                                               [0.2500, 0.2500, 0.2500, 0.2500, 0.2500, 0.2500],
                                               [0.2500, 0.2500, 0.2500, 0.2500, 0.2500, 0.2500],
                                               [0.2500, 0.2500, 0.2500, 0.2500, 0.2500, 0.2500],
                                               [0.2500, 0.2500, 0.2500, 0.2500, 0.2500, 0.2500]]]],
                                            dtype=torch.float32)

        self.assertRtolEqual(expedt_cpu_output, npu_output)
        self.assertRtolEqual(expedt_cpu_inputgrad, npu_inputgrad)

    def test_npu_roi_align_2(self):
        input1 = self.generate_input()
        roi = torch.tensor([[0, -2.0, -2.0, 22.0, 22.0]]).npu()
        output_size = (3, 3)
        spatial_scale = 0.25
        sampling_ratio = 2
        aligned = True

        npu_output, npu_inputgrad = self.npu_roi_align(input1, roi, output_size, spatial_scale, sampling_ratio, aligned)

        expedt_cpu_output = torch.tensor([[[[2.7500, 4.5000, 6.5000],
                                            [13.2500, 15.0000, 17.0000],
                                            [25.2500, 27.0000, 29.0000]]]], dtype=torch.float32)
        expedt_cpu_inputgrad = torch.tensor([[[[0.5625, 0.3750, 0.3750, 0.3750, 0.3750, 0.1875],
                                            [0.3750, 0.2500, 0.2500, 0.2500, 0.2500, 0.1250],
                                            [0.3750, 0.2500, 0.2500, 0.2500, 0.2500, 0.1250],
                                            [0.3750, 0.2500, 0.2500, 0.2500, 0.2500, 0.1250],
                                            [0.3750, 0.2500, 0.2500, 0.2500, 0.2500, 0.1250],
                                            [0.1875, 0.1250, 0.1250, 0.1250, 0.1250, 0.0625]]]],
                                            dtype=torch.float32)

        self.assertRtolEqual(expedt_cpu_output, npu_output)
        self.assertRtolEqual(expedt_cpu_inputgrad, npu_inputgrad)


if __name__ == "__main__":
    run_tests()
