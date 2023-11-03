import torch
import numpy as np
from torch import linalg as LA

import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor


class TestLinalgNorm(TestCase):

    def cpu_dtype_out_exec(self, data, ord_):
        cpu_output = LA.vector_norm(data, ord_)
        return cpu_output.numpy()

    def npu_dtype_out_exec(self, data, ord_):
        npu_output = LA.vector_norm(data, ord_)
        return npu_output.cpu().numpy()

    def test_linalg_vector_norm(self):
        # linalg.vector_norm only support float and complex
        dtype_list = [np.float32, np.float16]
        ords = [torch.inf, -torch.inf, 0, 2, 3]
        for item in dtype_list:
            for ord_ in ords:
                cpu_input_1, npu_input_1 = create_common_tensor([item, 0, (2, 3, 4)], -100, 100)
                cpu_input_2, npu_input_2 = create_common_tensor([item, 0, (2, 3, 4)], -100, 100)
                cpu_output_1 = self.cpu_dtype_out_exec(cpu_input_1, ord_)
                npu_output_1 = self.npu_dtype_out_exec(npu_input_1, ord_)
                self.assertRtolEqual(cpu_output_1, npu_output_1)
                cpu_output_2 = self.cpu_dtype_out_exec(torch.stack([cpu_input_1, cpu_input_2]), ord_)
                npu_output_2 = self.npu_dtype_out_exec(torch.stack([npu_input_1, npu_input_2]), ord_)
                self.assertRtolEqual(cpu_output_2, npu_output_2)

    def test_linalg_norm_fp32(self):
        ords = [torch.inf, -torch.inf, -1, 1, -2, 2, "fro", "nuc"]
        keepdim_list = [True, False]
        for ord_ in ords:
            for keepdim in keepdim_list:
                cpu_input_1, npu_input_1 = create_common_tensor([np.float32, 0, (6, 7)], -100, 100)
                cpu_output_1 = LA.norm(cpu_input_1, ord=ord_, keepdim=keepdim)
                npu_output_1 = LA.norm(npu_input_1, ord=ord_, keepdim=keepdim)
                self.assertRtolEqual(cpu_output_1, npu_output_1)

    def test_linalg_norm_fp16(self):
        ords = [torch.inf, -torch.inf, -1, 1, "fro"]
        keepdim_list = [True, False]
        for ord_ in ords:
            for keepdim in keepdim_list:
                cpu_input_1, npu_input_1 = create_common_tensor([np.float16, 0, (6, 7)], -100, 100)
                cpu_output_1 = LA.norm(cpu_input_1, ord=ord_, keepdim=keepdim)
                npu_output_1 = LA.norm(npu_input_1, ord=ord_, keepdim=keepdim)
                self.assertRtolEqual(cpu_output_1, npu_output_1)

    def test_linalg_matrix_norm_fp32(self):
        ords = [torch.inf, -torch.inf, -1, 1, -2, 2, "fro", "nuc"]
        keepdim_list = [True, False]
        dim_list = [[0, 1], [0, 2], [1, 2]]

        cpu_input_1, npu_input_1 = create_common_tensor([np.float32, 0, (6, 7, 8)], -100, 100)
        cpu_output_1 = LA.norm(cpu_input_1)
        npu_output_1 = LA.norm(npu_input_1)
        self.assertRtolEqual(cpu_output_1, npu_output_1)

        for ord_ in ords:
            for dim in dim_list:
                for keepdim in keepdim_list:
                    cpu_input_1, npu_input_1 = create_common_tensor([np.float32, 0, (6, 7, 8)], -100, 100)
                    cpu_output_1 = LA.matrix_norm(cpu_input_1, dim=dim, ord=ord_, keepdim=keepdim)
                    npu_output_1 = LA.matrix_norm(npu_input_1, dim=dim, ord=ord_, keepdim=keepdim)
                    self.assertRtolEqual(cpu_output_1, npu_output_1)

    def test_linalg_matrix_norm_fp16(self):
        ords = [torch.inf, -torch.inf, -1, 1, "fro"]
        keepdim_list = [True, False]
        for ord_ in ords:
            cpu_input_1, npu_input_1 = create_common_tensor([np.float16, 0, (6, 7)], -100, 100)
            cpu_output_1 = LA.matrix_norm(cpu_input_1, ord=ord_)
            npu_output_1 = LA.matrix_norm(npu_input_1, ord=ord_)
            self.assertRtolEqual(cpu_output_1, npu_output_1)


if __name__ == "__main__":
    run_tests()
