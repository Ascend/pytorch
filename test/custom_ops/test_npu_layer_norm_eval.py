import numpy as np
import torch

import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor


class TestLayernormeval(TestCase):

    def supported_op_exec(self, input1, normalized_shape, weight, bias, eps):
        result, _, _ = torch.native_layer_norm(input1, normalized_shape, weight, bias, eps)
        return result

    def custom_op_exec(self, input1, normalized_shape, weight, bias, eps):
        return torch_npu.npu_layer_norm_eval(input1, normalized_shape, weight, bias, eps)

    def test_npu_layer_norm_eval(self, device="npu"):
        input1 = torch.rand((6, 4), dtype=torch.float32).npu()
        normalized_shape = input1.size()[1:]
        weight = torch.Tensor(*normalized_shape).npu()
        bias = torch.Tensor(*normalized_shape).npu()

        supported_result = self.supported_op_exec(input1, normalized_shape, weight, bias, 1e-5)
        custom_result = self.custom_op_exec(input1, normalized_shape, weight, bias, 1e-5)
        self.assertRtolEqual(supported_result, custom_result)


if __name__ == "__main__":
    run_tests()
