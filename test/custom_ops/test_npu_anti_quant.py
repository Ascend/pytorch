import unittest
import numpy as np
import torch

import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor
DEVICE_NAME = torch_npu.npu.get_device_name(0)[:10]


class TestAntiQuant(TestCase):

    def custom_op_exec(self, input_x, scale, offset, dst_dtype, src_dtype):
        scale = torch.broadcast_to(scale, input_x.shape)
        if offset is None:
            offset = torch.zeros_like(scale)
        
        x = input_x.to(torch.float32)
        
        offset_temp = x + offset
        output = offset_temp * scale
        
        output = output.to(dst_dtype)
        return output.cpu().detach()

    def npu_op_exec(self, input_x, scale, offset, dst_dtype, src_dtype):
        output = torch_npu.npu_anti_quant(input_x, scale, offset=offset, dst_dtype=dst_dtype, src_dtype=src_dtype)
        return output.cpu().detach()

    @unittest.skipIf(DEVICE_NAME != 'Ascend910B',
        "OP `AscendAntiQuant` is only supported on 910B, skip this ut for this device type!")
    def test_anti_quant(self, device="npu"):
        shape_format = [
            [[np.int8, -1, [10, 100]], [np.float32, -1, [100]], [np.float32, -1, [100]], torch.float16, None],
            [[np.int8, -1, [10, 100]], [np.float32, -1, [100]], [np.float32, -1, [100]], torch.bfloat16, None],
            [[np.int8, -1, [10, 100]], [np.float32, -1, [100]], [np.float32, -1, [100]], torch.float16, torch.int8],
            [[np.int8, -1, [10, 100]], [np.float32, -1, [100]], [np.float32, -1, [100]], torch.bfloat16, torch.int8],
        ]
        
        for item in shape_format:
            cpu_input_x, npu_input_x = create_common_tensor(item[0], -127, 127)
            cpu_scale, npu_scale = create_common_tensor(item[1], -100, 100)
            cpu_offset, npu_offset = (None, None) if item[2] is None else create_common_tensor(item[2], -100, 100)
            
            npu_output = self.npu_op_exec(npu_input_x, npu_scale, npu_offset, *item[3:])
            custom_output = self.custom_op_exec(cpu_input_x, cpu_scale, cpu_offset, *item[3:])

            if item[3] == torch.bfloat16:
                npu_output = npu_output.to(torch.float32)
                custom_output = custom_output.to(torch.float32)
            self.assertRtolEqual(npu_output, custom_output)

if __name__ == "__main__":
    run_tests()
