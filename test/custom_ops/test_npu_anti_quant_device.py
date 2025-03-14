import unittest
import numpy as np
import torch

import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor, SupportedDevices
from torch_npu.testing.common_distributed import skipIfUnsupportMultiNPU


class TestAntiQuant(TestCase):
    def npu_op_exec(self, input_x, scale, offset, dst_dtype, src_dtype):
        output = torch_npu.npu_anti_quant(input_x, scale, offset=offset, dst_dtype=dst_dtype, src_dtype=src_dtype)
        return output

    @skipIfUnsupportMultiNPU(2)
    @SupportedDevices(['Ascend910B'])
    def test_anti_quant_device_check(self, device="npu"):
        shape_format = [
            [[np.int8, -1, [10, 100]], [np.float32, -1, [100]], [np.float32, -1, [100]], torch.float16, None],
        ]
        
        for item in shape_format:
            _, npu_input_x = create_common_tensor(item[0], -127, 127)
            _, npu_scale = create_common_tensor(item[1], -100, 100)
            _, npu_offset = (None, None) if item[2] is None else create_common_tensor(item[2], -100, 100)

            msg = "Expected all tensors to be on the same device, but found at least two devices, npu:"
            with self.assertRaisesRegex(RuntimeError, msg):
                self.npu_op_exec(npu_input_x, npu_scale, npu_offset.to("npu:1"), *item[3:])


if __name__ == "__main__":
    run_tests()
