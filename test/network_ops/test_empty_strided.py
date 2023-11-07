import torch
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor


class TestEmptyStrided(TestCase):
    def cpu_op_exec(self, size, strides, dtype):
        output = torch.empty_strided(size, strides, dtype=dtype, device="cpu").fill_(1)
        output = output.contiguous().numpy()
        return output

    def npu_op_exec(self, size, strides, dtype):
        output = torch.empty_strided(size, strides, dtype=dtype, device="npu").fill_(1)
        output = output.contiguous().to("cpu")
        output = output.numpy()
        return output

    def test_empty_strided(self):
        shape_stride = [
            [(2, 1, 5, 4), (2, 4, 3, 7), torch.float32],
            [(2, 1, 5, 4), (2, 4, 3, 7), torch.int32],
            [(2, 1, 5, 4), (8, 4, 3, 7), torch.float32],
            [(2, 1, 5, 4), (8, 4, 3, 7), torch.int32],
        ]
        for item in shape_stride:
            size, strides, dtype = item[0], item[1], item[2]
            cpu_output = self.cpu_op_exec(size, strides, dtype)
            npu_output = self.npu_op_exec(size, strides, dtype)
            self.assertRtolEqual(cpu_output, npu_output)


if __name__ == "__main__":
    run_tests()
