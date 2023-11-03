import itertools

import numpy as np
import torch
import torch.nn.functional as F

import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests


class Testcdist(TestCase):

    def generate_data(self, min_n, max_n, shape_predict, shape_label, src_type):
        np.random.seed(10086)
        predict = np.random.uniform(min_n, max_n, shape_predict).astype(src_type)
        label = np.random.uniform(min_n, max_n, shape_label).astype(src_type)
        label[label < 0] = -1
        label[label >= 0] = 1
        return predict, label

    def op_exec(self, predict, label, reduction, device='cpu', beta=1):
        is_fp16 = predict.dtype == np.float16
        if device == 'cpu' and is_fp16:
            predict = predict.astype(np.float32)
            label = label.astype(np.float32)
        predict = torch.from_numpy(predict)
        label = torch.from_numpy(label)
        predict = predict.to(device)
        label = label.to(device)

        predict.requires_grad = True
        output_forward = F.smooth_l1_loss(predict, label, reduction=reduction, beta=beta)
        output_forward = output_forward.sum()
        output_forward.backward()

        gradient = predict.grad.cpu().numpy()
        if device == 'cpu' and is_fp16:
            gradient = gradient.astype(np.float16)
        return gradient

    def test_smooth_l1_loss_backward_float16_3(self):
        shape_format = [
            [-1, 1, [100], [100], np.float16],
            [-0.1, 0.1, [100, 200], [100, 200], np.float16],
            [-10, 10, [100, 20, 30], [100, 20, 1], np.float16],
            [-0.01, 0.01, [100, 20, 30], [100, 20, 30], np.float16],
            [-0.001, 0.001, [10, 20, 30, 4], [10, 20, 30, 4], np.float16],
            [-0.001, 0.001, [10, 20, 3, 4, 5], [10, 20, 3, 4, 5], np.float16],
        ]
        reduction_list = ['none', 'mean', 'sum']
        for item, reduction in itertools.product(shape_format, reduction_list):
            input1, input2 = self.generate_data(item[0], item[1], item[2], item[3], item[4])
            cpu_output1 = self.op_exec(input1, input2, reduction, 'cpu')
            npu_output1 = self.op_exec(input1, input2, reduction, 'npu')
            self.assertRtolEqual(cpu_output1, npu_output1)

    def test_smooth_l1_loss_backward_float32_3(self):
        shape_format = [
            [-1, 1, [100], [100], np.float32],
            [-0.1, 0.1, [100, 200], [100, 200], np.float32],
            [-10, 10, [100, 20, 30], [100, 20, 1], np.float32],
            [-0.01, 0.01, [100, 20, 30], [100, 20, 30], np.float32],
            [-0.001, 0.001, [10, 20, 30, 4], [10, 20, 30, 4], np.float32],
            [-0.001, 0.001, [10, 20, 3, 4, 5], [10, 20, 3, 4, 5], np.float32],
        ]
        reduction_list = ['none', 'mean', 'sum']
        for item, reduction in itertools.product(shape_format, reduction_list):
            input1, input2 = self.generate_data(item[0], item[1], item[2], item[3], item[4])
            cpu_output1 = self.op_exec(input1, input2, reduction, 'cpu')
            npu_output1 = self.op_exec(input1, input2, reduction, 'npu')
            self.assertRtolEqual(cpu_output1, npu_output1)

    def test_smooth_l1_loss_backward_beta(self):
        beta_list = [0.5, 1, 1.5, 2]
        reduction_list = ['none', 'mean', 'sum']
        input1, input2 = self.generate_data(-1, 1, 100, 100, np.float32)
        for beta, reduction in itertools.product(beta_list, reduction_list):
            cpu_output1 = self.op_exec(input1, input2, reduction, 'cpu', beta)
            npu_output1 = self.op_exec(input1, input2, reduction, 'npu', beta)
            self.assertRtolEqual(cpu_output1, npu_output1)


if __name__ == "__main__":
    run_tests()
