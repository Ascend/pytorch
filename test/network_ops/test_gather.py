import torch
import numpy as np
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor


class TestGather(TestCase):
    def cpu_op_exec(self, input1, dim, index):
        output = torch.gather(input1, dim, index)
        output = output.numpy()
        return output

    def npu_op_exec(self, input1, dim, index):
        output = torch.gather(input1, dim, index)
        output = output.to("cpu")
        output = output.numpy()
        return output

    def test_gather_shape_format(self, device="npu"):
        shape_format = [
            [[np.int32, 0, (4, 3)], 0, torch.LongTensor([[0, 1, 1], [2, 0, 1]])],
            [[np.int64, 0, (2, 3)], 1, torch.LongTensor([[0, 1, 1], [0, 0, 1]])],
            [[np.float16, 0, (2, 3, 5)], 2, torch.LongTensor([[[0, 1, 2, 0, 2], [0, 0, 0, 0, 0], [1, 1, 1, 1, 1]],
                                                              [[1, 1, 1, 1, 1], [0, 0, 0, 0, 0], [1, 1, 1, 1, 1]]])],
            [[np.float32, 0, (3, 3, 5)], -3, torch.LongTensor([[[0, 1, 2, 0, 2], [0, 0, 0, 0, 0], [1, 1, 1, 1, 1]],
                                                               [[1, 2, 2, 2, 2], [0, 0, 0, 0, 0], [2, 2, 2, 2, 2]]])]
        ]
        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item[0], -100, 100)
            cpu_output = self.cpu_op_exec(cpu_input1, item[1], item[2])
            npu_idx = item[2].to("npu")
            npu_output = self.npu_op_exec(npu_input1, item[1], npu_idx)
            self.assertRtolEqual(cpu_output, npu_output)


if __name__ == "__main__":
    run_tests()
