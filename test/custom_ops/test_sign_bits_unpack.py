import torch
import numpy as np

import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor


class TestSignBitsUnpack(TestCase):

    def custom_sign_unpack(self, input_data, size, dtype):
        bits = 8
        mask = 2**torch.arange(bits).to(input_data.device, input_data.dtype)
        unpack_data = input_data.unsqueeze(-1).bitwise_and(mask).ne(0).byte().reshape(-1).to(dtype)
        unpack_data = (unpack_data - 0.5) * 2.0
        return unpack_data.reshape(size, unpack_data.shape[0] // size)

    def custom_op_exec(self, input_data, dtype, size):
        output = self.custom_sign_unpack(input_data, size, dtype)
        return output.cpu().numpy()

    def npu_op_exec(self, npu_input, dtype, size):
        nup_out = torch_npu.npu_sign_bits_unpack(npu_input, size, dtype)
        return nup_out.cpu().numpy()

    def test_sign_bits_unpack(self):
        shape = np.random.uniform(1, 10**5, 1)
        shape = shape // (10 ** int(np.random.uniform(0, int(np.log10(shape) + 1), 1)))
        shape = max(int(shape), 1)
        size = int(np.random.uniform(1, 100))
        shape = shape * size

        shape_format = [np.uint8, 2, [shape]]
        cpu_input, npu_input = create_common_tensor(shape_format, 0, 255)
        dtypes = [torch.float16, torch.float32]
        for dtype in dtypes:
            cpu_output = self.custom_op_exec(npu_input, dtype, size)
            npu_output = self.npu_op_exec(npu_input, dtype, size)
            self.assertRtolEqual(cpu_output, npu_output)


if __name__ == "__main__":
    run_tests()
