import unittest
import numpy as np
from ml_dtypes import int4
import torch

import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor, SupportedDevices
from torch_npu.testing.common_distributed import skipIfUnsupportMultiNPU


def unpack_int4(s32arr):
    dst_shape = s32arr.numpy().shape
    if len(dst_shape) == 0:
        dst_shape = (8, )
    else:
        dst_shape = (*(dst_shape[:-1]), dst_shape[-1] * 8)

    sa1 = s32arr.numpy().astype(np.int32)
    sa2 = sa1.tobytes()
    sa3 = np.frombuffer(sa2, dtype=np.uint8)
    shift = np.array([0, 4], dtype=np.uint8)
    sa4 = np.bitwise_and(sa3.reshape([-1, 1]) >> shift, 0b00001111).astype(int4).astype(np.int8).reshape(dst_shape)
    return torch.from_numpy(sa4)


class TestAntiQuant(TestCase):

    def custom_op_exec(self, input_x, scale, offset, dst_dtype, src_dtype):
        if input_x.dtype == torch.int32:
            input_x = unpack_int4(input_x)
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

    @SupportedDevices(['Ascend910B'])
    def test_anti_quant(self, device="npu"):
        shape_format = [
            [[np.int8, -1, [10, 100]], [np.float32, -1, [100]], [np.float32, -1, [100]], torch.float16, None],
            [[np.int8, -1, [10, 100]], [np.float32, -1, [100]], [np.float32, -1, [100]], torch.bfloat16, None],
            [[np.int8, -1, [10, 100]], [np.float32, -1, [100]], [np.float32, -1, [100]], torch.float16, torch.int8],
            [[np.int8, -1, [10, 100]], [np.float32, -1, [100]], [np.float32, -1, [100]], torch.bfloat16, torch.int8],
            [[np.int32, -1, [10, 25]], [np.float32, -1, [200]], [np.float32, -1, [200]], torch.float16, None],
            [[np.int32, -1, [10, 25]], [np.float32, -1, [200]], [np.float32, -1, [200]], torch.bfloat16, None],
            [[np.int32, -1, [10, 25]], [np.float32, -1, [200]], [np.float32, -1, [200]], torch.float16, None],
            [[np.int32, -1, [10, 25]], [np.float32, -1, [200]], [np.float32, -1, [200]], torch.bfloat16, None],
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
