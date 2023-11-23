import unittest
import torch
import numpy as np
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor


class TestNorm(TestCase):
    def norm_output_size(self, data, dimVal, keepdimVal):
        output_size = list(data.size())
        for i in dimVal:
            if i < 0:
                i = i + data.dim()
            if i < data.dim() and keepdimVal:
                output_size[i] = 1
            if i < data.dim() and not keepdimVal:
                output_size.pop(i)
        return output_size

    def cpu_dtype_out_exec(self, data, pVal, dimVal, keepdimVal, dtypeVal):
        output_size = self.norm_output_size(data, dimVal, keepdimVal)
        cpu_output = torch.randn(output_size)
        torch.norm(data, p=pVal, dim=dimVal, keepdim=keepdimVal, out=cpu_output, dtype=dtypeVal)
        return cpu_output.numpy()

    def npu_dtype_out_exec(self, data, pVal, dimVal, keepdimVal, dtypeVal):
        output_size = self.norm_output_size(data, dimVal, keepdimVal)
        npu_output = torch.randn(output_size).npu()
        torch.norm(data, p=pVal, dim=dimVal, keepdim=keepdimVal, out=npu_output, dtype=dtypeVal)
        return npu_output.cpu().numpy()

    def dtype_out_test(self, item):
        cpu_input, npu_input = create_common_tensor(item[0], 0, 100)
        cpu_out = self.cpu_dtype_out_exec(cpu_input, 2, [1, 2], True, torch.float)
        npu_out = self.npu_dtype_out_exec(npu_input, 2, [1, 2], True, torch.float)
        self.assertRtolEqual(cpu_out, npu_out)

        cpu_out = self.cpu_dtype_out_exec(cpu_input, 2, [1, 2], False, torch.float)
        npu_out = self.npu_dtype_out_exec(npu_input, 2, [1, 2], False, torch.float)
        self.assertRtolEqual(cpu_out, npu_out)

        cpu_out = self.cpu_dtype_out_exec(cpu_input, 1, [1, 2], False, torch.float)
        npu_out = self.npu_dtype_out_exec(npu_input, 1, [1, 2], False, torch.float)
        self.assertRtolEqual(cpu_out, npu_out)

        cpu_out = self.cpu_dtype_out_exec(cpu_input, 3, [1, 2], False, torch.float)
        npu_out = self.npu_dtype_out_exec(npu_input, 3, [1, 2], False, torch.float)
        self.assertRtolEqual(cpu_out, npu_out)

        cpu_out = self.cpu_dtype_out_exec(cpu_input, float("-inf"), [1, 2], False, torch.float)
        npu_out = self.npu_dtype_out_exec(npu_input, float("-inf"), [1, 2], False, torch.float)
        self.assertRtolEqual(cpu_out, npu_out)

    @unittest.skip("skip test_norm_shape_format now")
    def test_norm_shape_format(self):
        shape_format = [
            [[np.float32, 0, (64, 64, 64, 64)]],
        ]

        for item in shape_format:
            # norm.dtype_out
            self.dtype_out_test(item)


if __name__ == "__main__":
    run_tests()
