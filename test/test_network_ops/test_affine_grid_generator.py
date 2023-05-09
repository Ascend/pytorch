import torch
import numpy as np
from torch.nn import functional as F
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests


class TestAffineGridGenerator(TestCase):
    def cpu_op_exec(self, theta, size, align_corners):
        output = F.affine_grid(theta, torch.Size(size), align_corners)
        output = output.numpy()
        return output

    def npu_op_exec(self, theta, size, align_corners):
        theta = theta.npu()
        output = torch.affine_grid_generator(theta, size, align_corners)
        output = output.cpu().numpy()
        return output

    def test_affine_grid_generator_2D(self, device="npu"):
        theta_list = [[1, 0, 0],
                      [0, 1, 0],
                     ]
        size = (1, 3, 10, 10)
        align_corners_list = [True, False]
        dtype_list = [torch.float32, torch.float16]
        shape_format = [
            [theta_list, size, i, j] for i in align_corners_list for j in dtype_list
        ]
        for item in shape_format:
            theta = torch.tensor([item[0]], dtype = item[3])
            cpu_input = theta
            npu_input = theta           
            if cpu_input.dtype == torch.float16:
                cpu_input = cpu_input.to(torch.float32)            
            cpu_output = self.cpu_op_exec(cpu_input, item[1], item[2])
            npu_output = self.npu_op_exec(npu_input, item[1], item[2])
            cpu_output = cpu_output.astype(npu_output.dtype)
            self.assertRtolEqual(cpu_output, npu_output, 0.001)

    def test_affine_grid_generator_3D(self, device="npu"):
        theta_list = [[1, 0, 0, 0],
                      [0, 1, 0, 0],
                      [0, 0, 1, 0],
                      ]
        size = (1, 3, 10, 10, 10)
        align_corners_list = [True, False]
        dtype_list = [torch.float16, torch.float32]
        shape_format = [
            [theta_list, size, i, j] for i in align_corners_list for j in dtype_list
        ]
        for item in shape_format:
            theta = torch.tensor([item[0]], dtype = item[3])
            cpu_input = theta
            npu_input = theta           
            if cpu_input.dtype == torch.float16:
                cpu_input = cpu_input.to(torch.float32)            
            cpu_output = self.cpu_op_exec(cpu_input, item[1], item[2])
            npu_output = self.npu_op_exec(npu_input, item[1], item[2])
            cpu_output = cpu_output.astype(npu_output.dtype)
            self.assertRtolEqual(cpu_output, npu_output, 0.001)


if __name__ == "__main__":
    run_tests()
