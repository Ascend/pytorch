import torch
import numpy as np

import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor


class TestLinalgVectorNorm(TestCase):

    def cpu_dtype_out_exec(self, data, ord_):
        cpu_output = torch._C._linalg.linalg_vector_norm(data, ord_)
        return cpu_output.numpy()
    
    def npu_dtype_out_exec(self, data, ord_):
        npu_output = torch.linalg.vector_norm(data, ord_)
        return npu_output.cpu().numpy()

    def test_linalg_vector_norm(self):
        # linalg.vector_norm only support float and complex
        dtype_list = [np.float32, np.float16]
        ords = [torch.inf, 2, 3]
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


if __name__ == "__main__":
    run_tests()
