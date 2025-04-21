import torch
import numpy as np
import torch_npu
from torch_npu.testing.common_utils import SupportedDevices
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.decorator import Dtypes, instantiate_tests


@instantiate_tests
class TestNpuStrideAdd(TestCase):
    def _split_npu_stride_add(self, x1, x2, offset1, offset2, c1_len):
        x1 = x1[:, offset1:, :, :, :]
        x2 = x2[:, offset2:, :, :, :]
        x1_size = list(x1.size())
        x2_size = list(x2.size())
        x1_pad_size = [x1_size[0]] + [16 * c1_len - x1_size[1]] + x1_size[2:]
        x2_pad_size = [x2_size[0]] + [16 * c1_len - x2_size[1]] + x2_size[2:]
        x1_pad = torch.cat((x1, torch.zeros(*x1_pad_size, device='npu')), 1)
        x2_pad = torch.cat((x2, torch.zeros(*x2_pad_size, device='npu')), 1)
        return torch.add(x1_pad, x2_pad)

    def npu_op_exec(self, input1, input2, offset1, offset2, c1_len):
        output = torch_npu.npu_stride_add(input1, input2, offset1, offset2, c1_len)
        output = output.to("cpu")
        output = output.numpy()
        return output

    def split_npu_op_exec(self, input1, input2, offset1, offset2, c1_len):
        output = self._split_npu_stride_add(input1, input2, offset1, offset2, c1_len)
        output = output.to("cpu")
        output = output.numpy()
        return output

    @SupportedDevices(['Ascend910A', 'Ascend910P'])
    def test_StrideAdd(self):
        input1 = torch.tensor([[[[[1.]]]]]).npu()
        input2 = input1
        exoutput = self.npu_op_exec(input1, input2, 0, 0, 1)
        output = self.split_npu_op_exec(input1, input2, 0, 0, 1)
        self.assertRtolEqual(exoutput, output)


if __name__ == "__main__":
    run_tests()
