import time
import numpy as np
import torch
import torch.nn as nn
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor, SupportedDevices
from torch_npu.contrib.module import FastBatchNorm1d, FastBatchNorm2d, FastBatchNorm3d


class TestBatchNormWithInt32Count(TestCase):
    def npu_slow_batchnorm1d_op_exec(self, num_features, input1):
        slow_batchnorm1d = nn.BatchNorm1d(num_features).npu()
        output = slow_batchnorm1d(input1)
        output.sum().backward()
        return output.cpu().detach().numpy()

    def npu_fast_batchnorm1d_op_exec(self, num_features, input1):
        fast_batchnorm1d = FastBatchNorm1d(num_features).npu()
        output = fast_batchnorm1d(input1)
        output.sum().backward()
        return output.cpu().detach().numpy()

    def npu_slow_batchnorm1d(self, num_features, input1):
        output = self.npu_slow_batchnorm1d_op_exec(num_features, input1)

        repeat_time = 100
        torch.npu.synchronize()
        t1 = time.time()
        for _ in range(repeat_time):
            self.npu_slow_batchnorm1d_op_exec(num_features, input1)
        torch.npu.synchronize()
        slow_time = (time.time() - t1) / repeat_time * 1000

        return output, slow_time

    def npu_fast_batchnorm1d(self, num_features, input1):
        output = self.npu_fast_batchnorm1d_op_exec(num_features, input1)

        repeat_time = 100
        torch.npu.synchronize()
        t2 = time.time()
        for _ in range(repeat_time):
            self.npu_fast_batchnorm1d_op_exec(num_features, input1)
        torch.npu.synchronize()
        fast_time = (time.time() - t2) / repeat_time * 1000

        return output, fast_time

    def npu_slow_batchnorm2d_op_exec(self, num_features, input1):
        slow_batchnorm2d = nn.BatchNorm2d(num_features).npu()
        output = slow_batchnorm2d(input1)
        output.sum().backward()
        return output.cpu().detach().numpy()

    def npu_fast_batchnorm2d_op_exec(self, num_features, input1):
        fast_batchnorm2d = FastBatchNorm2d(num_features).npu()
        output = fast_batchnorm2d(input1)
        output.sum().backward()
        return output.cpu().detach().numpy()

    def npu_slow_batchnorm2d(self, num_features, input1):
        output = self.npu_slow_batchnorm2d_op_exec(num_features, input1)

        repeat_time = 100
        torch.npu.synchronize()
        t1 = time.time()
        for _ in range(repeat_time):
            self.npu_slow_batchnorm2d_op_exec(num_features, input1)
        torch.npu.synchronize()
        slow_time = (time.time() - t1) / repeat_time * 1000

        return output, slow_time

    def npu_fast_batchnorm2d(self, num_features, input1):
        output = self.npu_fast_batchnorm2d_op_exec(num_features, input1)

        repeat_time = 100
        torch.npu.synchronize()
        t2 = time.time()
        for _ in range(repeat_time):
            self.npu_fast_batchnorm2d_op_exec(num_features, input1)
        torch.npu.synchronize()
        fast_time = (time.time() - t2) / repeat_time * 1000

        return output, fast_time

    def npu_slow_batchnorm3d_op_exec(self, num_features, input1):
        slow_batchnorm3d = nn.BatchNorm3d(num_features).npu()
        output = slow_batchnorm3d(input1)
        output.sum().backward()
        return output.cpu().detach().numpy()

    def npu_fast_batchnorm3d_op_exec(self, num_features, input1):
        fast_batchnorm3d = FastBatchNorm3d(num_features).npu()
        output = fast_batchnorm3d(input1)
        output.sum().backward()
        return output.cpu().detach().numpy()

    def npu_slow_batchnorm3d(self, num_features, input1):
        output = self.npu_slow_batchnorm3d_op_exec(num_features, input1)

        repeat_time = 100
        torch.npu.synchronize()
        t1 = time.time()
        for _ in range(repeat_time):
            self.npu_slow_batchnorm3d_op_exec(num_features, input1)
        torch.npu.synchronize()
        slow_time = (time.time() - t1) / repeat_time * 1000

        return output, slow_time

    def npu_fast_batchnorm3d(self, num_features, input1):
        output = self.npu_fast_batchnorm3d_op_exec(num_features, input1)

        repeat_time = 100
        torch.npu.synchronize()
        t2 = time.time()
        for _ in range(repeat_time):
            self.npu_fast_batchnorm3d_op_exec(num_features, input1)
        torch.npu.synchronize()
        fast_time = (time.time() - t2) / repeat_time * 1000

        return output, fast_time

    @SupportedDevices(['Ascend910A', 'Ascend910P'])
    def test_batchnorm1d_shape_format(self):
        shape_format = [
            [[np.float32, 2, [20, 100]], 100],
            [[np.float32, 3, [50, 100, 4]], 100],
            [[np.float16, 2, [20, 100]], 100],
            [[np.float16, 3, [50, 100, 4]], 100],
        ]
        for item in shape_format:
            _, input1 = create_common_tensor(item[0], -10, 10)
            input1.requires_grad_(True)
            num_features = item[1]
            slow_output, slow_time = \
                self.npu_slow_batchnorm1d(num_features, input1)
            fast_output, fast_time = \
                self.npu_fast_batchnorm1d(num_features, input1)

            self.assertRtolEqual(slow_output, fast_output)
            self.assertTrue(slow_time > fast_time)

    @SupportedDevices(['Ascend910A', 'Ascend910P'])
    def test_batchnorm2d_shape_format(self):
        shape_format = [
            [[np.float32, 0, [20, 100, 4, 5]], 100],
            [[np.float32, 3, [50, 100, 4, 8]], 100],
            [[np.float16, 0, [20, 5, 8, 3]], 5],
            [[np.float16, 3, [50, 5, 4, 7]], 5],
        ]
        for item in shape_format:
            _, input1 = create_common_tensor(item[0], -10, 10)
            input1.requires_grad_(True)
            num_features = item[1]
            slow_output, slow_time = \
                self.npu_slow_batchnorm2d(num_features, input1)
            fast_output, fast_time = \
                self.npu_fast_batchnorm2d(num_features, input1)

            self.assertRtolEqual(slow_output, fast_output)
            self.assertTrue(slow_time > fast_time)

    @SupportedDevices(['Ascend910A', 'Ascend910P'])
    def test_batchnorm3d_shape_format(self):
        shape_format = [
            [[np.float32, 30, [20, 100, 4, 5, 7]], 100],
            [[np.float32, 30, [50, 100, 4, 8, 4]], 100],
            [[np.float16, 30, [20, 5, 8, 3, 8]], 5],
            [[np.float16, 30, [50, 5, 4, 7, 9]], 5],
        ]
        for item in shape_format:
            _, input1 = create_common_tensor(item[0], -10, 10)
            input1.requires_grad_(True)
            num_features = item[1]
            slow_output, slow_time = \
                self.npu_slow_batchnorm3d(num_features, input1)
            fast_output, fast_time = \
                self.npu_fast_batchnorm3d(num_features, input1)

            self.assertRtolEqual(slow_output, fast_output)
            self.assertTrue(slow_time > fast_time)


if __name__ == "__main__":
    run_tests()
