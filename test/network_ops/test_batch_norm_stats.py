import torch
import numpy as np
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor


class TestBatchNormStats(TestCase):
    def cuda_op_exec(self, *args):
        cpu_mean, cpu_invstd = torch.batch_norm_stats(*args)
        return cpu_mean.numpy(), cpu_invstd.numpy()

    def cuda_expect_result(self):
        cpu_output0 = np.array([5.401827, 5.444219, 5.7656665], dtype=np.float32)
        cpu_output1 = np.array([0.37123242, 0.38706362, 0.37435925], dtype=np.float32)
        return cpu_output0, cpu_output1

    def npu_op_exec(self, *args):
        npu_mean, npu_invstd = torch.batch_norm_stats(*args)
        out_mean = npu_mean.cpu().numpy()
        out_invstd = npu_invstd.cpu().numpy()
        return out_mean, out_invstd

    def test_batch_norm_stats(self):
        np.random.seed(1234)
        shape_format = [
            [[np.float16, -1, [2, 3, 12, 12]], 1e-5],
        ]
        for item in shape_format:
            cpu_input1, npu_inputfp16 = create_common_tensor(item[0], 1, 10)
            npu_input1fp32 = npu_inputfp16.float()
            if torch.cuda.is_available():
                cpu_output = self.cuda_op_exec(cpu_input1.cuda(), item[-1])
            else:
                cpu_output = self.cuda_expect_result()
            npu_outputfp16 = self.npu_op_exec(npu_inputfp16, item[-1])
            npu_outputfp32 = self.npu_op_exec(npu_inputfp16, item[-1])

            self.assertRtolEqual(cpu_output[0], npu_outputfp16[0])
            self.assertRtolEqual(cpu_output[1], npu_outputfp16[1], 1e-2)

            self.assertRtolEqual(cpu_output[0], npu_outputfp32[0])
            self.assertRtolEqual(cpu_output[1], npu_outputfp32[1], 1e-2)


if __name__ == "__main__":
    run_tests()
