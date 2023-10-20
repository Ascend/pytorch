import torch
import numpy as np
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor


class TestLess(TestCase):
    def cpu_op_exec(self, input1, size):
        sign_data = np.sign(input1)
        sign_data = sign_data + 1
        bool_data = np.bool_(sign_data)
        pack_bit = np.packbits(bool_data, bitorder="little")
        return pack_bit.reshape(size, pack_bit.shape[0] // size)

    def npu_op_exec(self, input1, size):
        output = torch_npu.npu_sign_bits_pack(input1, size)
        output = output.to("cpu")
        output = output.numpy()
        return output

    def test_sign_bits_pack(self):
        shape_format = [
            [[np.float16, (17,)], 1],
            [[np.float32, (8,)], 1],
            [[np.float32, (32,)], 1],
            [[np.float32, (33,)], 1],
            [[np.float32, (16,)], 2]
        ]

        for item in shape_format:
            input1 = np.random.uniform(-10, 10, item[0][1]).astype(item[0][0])
            npu_input1 = torch.from_numpy(input1).npu()
            cpu_output = self.cpu_op_exec(input1, item[1])
            npu_output = self.npu_op_exec(npu_input1, item[1])

            self.assertRtolEqual(cpu_output, npu_output)


if __name__ == "__main__":
    run_tests()
