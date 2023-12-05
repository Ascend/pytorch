import itertools
import torch
import numpy as np

import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor


class TestIm2colBackward(TestCase):

    def test_im2col_backward(self):
        dtype_list = [np.float16, np.float32]
        shape_list = [[1, 144, 256], [144, 256]]
        for dtype, shape in itertools.product(dtype_list, shape_list):
            fold_cpu = torch.nn.Fold(output_size=(18, 18), kernel_size=(3, 3))
            fold_npu = fold_cpu.npu()
            cpu_input, npu_input = create_common_tensor((dtype, 0, shape), -100, 100)
            cpu_output = fold_cpu(cpu_input)
            npu_output = fold_npu(npu_input)
            self.assertRtolEqual(cpu_output, npu_output.cpu())


if __name__ == '__main__':
    run_tests()
