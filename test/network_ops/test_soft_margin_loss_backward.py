import torch
import numpy as np
import torch.nn.functional as F
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor


class Testcdist(TestCase):
    def generate_data(self, min_n, max_n, shape_predict, shape_label, src_type):
        np.random.seed(10086)
        predict = np.random.uniform(min_n, max_n,
                                    shape_predict).astype(src_type)
        label = np.random.uniform(min_n, max_n, shape_label).astype(src_type)
        label[label < 0] = -1
        label[label >= 0] = 1
        dout = np.ones(shape_predict).astype(src_type)
        return predict, label, dout

    def op_exec(self, predict, label, dout, reduction, device='cpu'):
        is_fp16 = predict.dtype == np.float16
        if device == 'cpu' and is_fp16:
            predict = predict.astype(np.float32)
            label = label.astype(np.float32)
            dout = dout.astype(np.float32)

        predict = torch.from_numpy(predict)
        label = torch.from_numpy(label)
        dout = torch.from_numpy(dout)

        predict = predict.to(device)
        label = label.to(device)
        dout = dout.to(device)

        predict.requires_grad = True

        output_forward = F.soft_margin_loss(predict, label, reduction=reduction)
        if reduction == 'none':
            output_forward.backward(dout)
        else:
            output_forward.backward()

        gradient = predict.grad.cpu().numpy()

        if device == 'cpu' and is_fp16:
            gradient = gradient.astype(np.float16)
        return gradient

    def test_soft_margin_loss_backward_float16_1(self):
        input1, input2, input3 = self.generate_data(-1, 1,
                                                    (100,), (100,), np.float16)
        cpu_output = self.op_exec(input1, input2, input3, 'none', 'cpu')
        npu_output = self.op_exec(input1, input2, input3, 'none', 'npu')
        self.assertRtolEqual(cpu_output, npu_output)

    def test_soft_margin_loss_backward_float16_2(self):
        input1, input2, input3 = self.generate_data(-1, 1,
                                                    (100,), (100,), np.float16)
        cpu_output = self.op_exec(input1, input2, input3, 'mean', 'cpu')
        npu_output = self.op_exec(input1, input2, input3, 'mean', 'npu')
        self.assertRtolEqual(cpu_output, npu_output)

    def test_soft_margin_loss_backward_float16_3(self):
        input1, input2, input3 = self.generate_data(-1, 1,
                                                    (100,), (100,), np.float16)
        cpu_output = self.op_exec(input1, input2, input3, 'sum', 'cpu')
        npu_output = self.op_exec(input1, input2, input3, 'sum', 'npu')
        self.assertRtolEqual(cpu_output, npu_output)

    def test_soft_margin_loss_backward_float16_4(self):
        input1, input2, input3 = self.generate_data(-0.1, 0.1,
                                                    (100, 200), (100, 200),
                                                    np.float16)
        cpu_output = self.op_exec(input1, input2, input3, 'none', 'cpu')
        npu_output = self.op_exec(input1, input2, input3, 'none', 'npu')
        self.assertRtolEqual(cpu_output, npu_output)

    def test_soft_margin_loss_backward_float16_5(self):
        input1, input2, input3 = self.generate_data(-0.1, 0.1,
                                                    (100, 200), (100, 200),
                                                    np.float16)
        cpu_output = self.op_exec(input1, input2, input3, 'mean', 'cpu')
        npu_output = self.op_exec(input1, input2, input3, 'mean', 'npu')
        self.assertRtolEqual(cpu_output, npu_output)

    def test_soft_margin_loss_backward_float16_6(self):
        input1, input2, input3 = self.generate_data(-0.1, 0.1,
                                                    (100, 200), (100, 200),
                                                    np.float16)
        cpu_output = self.op_exec(input1, input2, input3, 'sum', 'cpu')
        npu_output = self.op_exec(input1, input2, input3, 'sum', 'npu')
        self.assertRtolEqual(cpu_output, npu_output)

    def test_soft_margin_loss_backward_float16_7(self):
        input1, input2, input3 = self.generate_data(-10, 10,
                                                    (100, 20, 30),
                                                    (100, 20, 1),
                                                    np.float16)
        cpu_output = self.op_exec(input1, input2, input3, 'none', 'cpu')
        npu_output = self.op_exec(input1, input2, input3, 'none', 'npu')
        self.assertRtolEqual(cpu_output, npu_output)

    def test_soft_margin_loss_backward_float16_8(self):
        input1, input2, input3 = self.generate_data(-10, 10,
                                                    (100, 20, 30),
                                                    (100, 20, 1),
                                                    np.float16)
        cpu_output = self.op_exec(input1, input2, input3, 'mean', 'cpu')
        npu_output = self.op_exec(input1, input2, input3, 'mean', 'npu')
        self.assertRtolEqual(cpu_output, npu_output)

    def test_soft_margin_loss_backward_float16_9(self):
        input1, input2, input3 = self.generate_data(-10, 10,
                                                    (100, 20, 30),
                                                    (100, 20, 1),
                                                    np.float16)
        cpu_output = self.op_exec(input1, input2, input3, 'sum', 'cpu')
        npu_output = self.op_exec(input1, input2, input3, 'sum', 'npu')
        self.assertRtolEqual(cpu_output, npu_output)

    def test_soft_margin_loss_backward_float16_10(self):
        input1, input2, input3 = self.generate_data(-0.01, 0.01,
                                                    (100, 20, 30),
                                                    (100, 20, 30),
                                                    np.float16)
        cpu_output = self.op_exec(input1, input2, input3, 'none', 'cpu')
        npu_output = self.op_exec(input1, input2, input3, 'none', 'npu')
        self.assertRtolEqual(cpu_output, npu_output)

    def test_soft_margin_loss_backward_float16_11(self):
        input1, input2, input3 = self.generate_data(-0.01, 0.01,
                                                    (100, 20, 30),
                                                    (100, 20, 30),
                                                    np.float16)
        cpu_output = self.op_exec(input1, input2, input3, 'mean', 'cpu')
        npu_output = self.op_exec(input1, input2, input3, 'mean', 'npu')
        self.assertRtolEqual(cpu_output, npu_output)

    def test_soft_margin_loss_backward_float16_12(self):
        input1, input2, input3 = self.generate_data(-0.01, 0.01,
                                                    (100, 20, 30),
                                                    (100, 20, 30),
                                                    np.float16)
        cpu_output = self.op_exec(input1, input2, input3, 'sum', 'cpu')
        npu_output = self.op_exec(input1, input2, input3, 'sum', 'npu')
        self.assertRtolEqual(cpu_output, npu_output)

    def test_soft_margin_loss_backward_float16_13(self):
        input1, input2, input3 = self.generate_data(-0.001, 0.001,
                                                    (10, 20, 30, 4),
                                                    (10, 20, 30, 4),
                                                    np.float16)
        cpu_output = self.op_exec(input1, input2, input3, 'none', 'cpu')
        npu_output = self.op_exec(input1, input2, input3, 'none', 'npu')
        self.assertRtolEqual(cpu_output, npu_output)

    def test_soft_margin_loss_backward_float16_14(self):
        input1, input2, input3 = self.generate_data(-0.001, 0.001,
                                                    (10, 20, 30, 4),
                                                    (10, 20, 30, 4),
                                                    np.float16)
        cpu_output = self.op_exec(input1, input2, input3, 'mean', 'cpu')
        npu_output = self.op_exec(input1, input2, input3, 'mean', 'npu')
        self.assertRtolEqual(cpu_output, npu_output)

    def test_soft_margin_loss_backward_float16_15(self):
        input1, input2, input3 = self.generate_data(-0.001, 0.001,
                                                    (10, 20, 30, 4),
                                                    (10, 20, 30, 4),
                                                    np.float16)
        cpu_output = self.op_exec(input1, input2, input3, 'sum', 'cpu')
        npu_output = self.op_exec(input1, input2, input3, 'sum', 'npu')
        self.assertRtolEqual(cpu_output, npu_output)

    def test_soft_margin_loss_backward_float16_16(self):
        input1, input2, input3 = self.generate_data(-0.001, 0.001,
                                                    (100, 20, 3, 4, 5),
                                                    (100, 20, 3, 4, 5),
                                                    np.float16)
        cpu_output = self.op_exec(input1, input2, input3, 'none', 'cpu')
        npu_output = self.op_exec(input1, input2, input3, 'none', 'npu')
        self.assertRtolEqual(cpu_output, npu_output)

    def test_soft_margin_loss_backward_float16_17(self):
        input1, input2, input3 = self.generate_data(-0.001, 0.001,
                                                    (100, 20, 3, 4, 5),
                                                    (100, 20, 3, 4, 5),
                                                    np.float16)
        cpu_output = self.op_exec(input1, input2, input3, 'mean', 'cpu')
        npu_output = self.op_exec(input1, input2, input3, 'mean', 'npu')
        self.assertRtolEqual(cpu_output, npu_output)

    def test_soft_margin_loss_backward_float16_18(self):
        input1, input2, input3 = self.generate_data(-0.001, 0.001,
                                                    (10, 20, 3, 4, 5),
                                                    (10, 20, 3, 4, 5),
                                                    np.float16)
        cpu_output = self.op_exec(input1, input2, input3, 'sum', 'cpu')
        npu_output = self.op_exec(input1, input2, input3, 'sum', 'npu')
        self.assertRtolEqual(cpu_output, npu_output)

    def test_soft_margin_loss_backward_float32_1(self):
        input1, input2, input3 = self.generate_data(-1, 1,
                                                    (100,), (100,), np.float32)
        cpu_output = self.op_exec(input1, input2, input3, 'none', 'cpu')
        npu_output = self.op_exec(input1, input2, input3, 'none', 'npu')
        self.assertRtolEqual(cpu_output, npu_output)

    def test_soft_margin_loss_backward_float32_2(self):
        input1, input2, input3 = self.generate_data(-1, 1,
                                                    (100,), (100,), np.float32)
        cpu_output = self.op_exec(input1, input2, input3, 'mean', 'cpu')
        npu_output = self.op_exec(input1, input2, input3, 'mean', 'npu')
        self.assertRtolEqual(cpu_output, npu_output)

    def test_soft_margin_loss_backward_float32_3(self):
        input1, input2, input3 = self.generate_data(-1, 1,
                                                    (100,), (100,), np.float32)
        cpu_output = self.op_exec(input1, input2, input3, 'sum', 'cpu')
        npu_output = self.op_exec(input1, input2, input3, 'sum', 'npu')
        self.assertRtolEqual(cpu_output, npu_output)

    def test_soft_margin_loss_backward_float32_4(self):
        input1, input2, input3 = self.generate_data(-0.1, 0.1,
                                                    (100, 200), (100, 200),
                                                    np.float32)
        cpu_output = self.op_exec(input1, input2, input3, 'none', 'cpu')
        npu_output = self.op_exec(input1, input2, input3, 'none', 'npu')
        self.assertRtolEqual(cpu_output, npu_output)

    def test_soft_margin_loss_backward_float32_5(self):
        input1, input2, input3 = self.generate_data(-0.1, 0.1,
                                                    (100, 200), (100, 200),
                                                    np.float16)
        cpu_output = self.op_exec(input1, input2, input3, 'mean', 'cpu')
        npu_output = self.op_exec(input1, input2, input3, 'mean', 'npu')
        self.assertRtolEqual(cpu_output, npu_output)

    def test_soft_margin_loss_backward_float32_6(self):
        input1, input2, input3 = self.generate_data(-0.1, 0.1,
                                                    (100, 200), (100, 200),
                                                    np.float32)
        cpu_output = self.op_exec(input1, input2, input3, 'sum', 'cpu')
        npu_output = self.op_exec(input1, input2, input3, 'sum', 'npu')
        self.assertRtolEqual(cpu_output, npu_output)

    def test_soft_margin_loss_backward_float32_7(self):
        input1, input2, input3 = self.generate_data(-10, 10,
                                                    (100, 20, 30),
                                                    (100, 20, 30),
                                                    np.float32)
        cpu_output = self.op_exec(input1, input2, input3, 'none', 'cpu')
        npu_output = self.op_exec(input1, input2, input3, 'none', 'npu')
        self.assertRtolEqual(cpu_output, npu_output)

    def test_soft_margin_loss_backward_float32_8(self):
        input1, input2, input3 = self.generate_data(-10, 10,
                                                    (100, 20, 30),
                                                    (100, 20, 30),
                                                    np.float32)
        cpu_output = self.op_exec(input1, input2, input3, 'mean', 'cpu')
        npu_output = self.op_exec(input1, input2, input3, 'mean', 'npu')
        self.assertRtolEqual(cpu_output, npu_output)

    def test_soft_margin_loss_backward_float32_9(self):
        input1, input2, input3 = self.generate_data(-10, 10,
                                                    (100, 20, 30),
                                                    (100, 20, 30),
                                                    np.float32)
        cpu_output = self.op_exec(input1, input2, input3, 'sum', 'cpu')
        npu_output = self.op_exec(input1, input2, input3, 'sum', 'npu')
        self.assertRtolEqual(cpu_output, npu_output)

    def test_soft_margin_loss_backward_float32_10(self):
        input1, input2, input3 = self.generate_data(-0.01, 0.01,
                                                    (100, 20, 30),
                                                    (100, 20, 30),
                                                    np.float32)
        cpu_output = self.op_exec(input1, input2, input3, 'none', 'cpu')
        npu_output = self.op_exec(input1, input2, input3, 'none', 'npu')
        self.assertRtolEqual(cpu_output, npu_output)

    def test_soft_margin_loss_backward_float32_11(self):
        input1, input2, input3 = self.generate_data(-0.01, 0.01,
                                                    (100, 20, 30),
                                                    (100, 20, 30),
                                                    np.float32)
        cpu_output = self.op_exec(input1, input2, input3, 'mean', 'cpu')
        npu_output = self.op_exec(input1, input2, input3, 'mean', 'npu')
        self.assertRtolEqual(cpu_output, npu_output)

    def test_soft_margin_loss_backward_float32_12(self):
        input1, input2, input3 = self.generate_data(-0.01, 0.01,
                                                    (100, 20, 30),
                                                    (100, 20, 30),
                                                    np.float32)
        cpu_output = self.op_exec(input1, input2, input3, 'sum', 'cpu')
        npu_output = self.op_exec(input1, input2, input3, 'sum', 'npu')
        self.assertRtolEqual(cpu_output, npu_output)

    def test_soft_margin_loss_backward_float32_13(self):
        input1, input2, input3 = self.generate_data(-0.001, 0.001,
                                                    (10, 20, 30, 4),
                                                    (10, 20, 30, 4),
                                                    np.float32)
        cpu_output = self.op_exec(input1, input2, input3, 'none', 'cpu')
        npu_output = self.op_exec(input1, input2, input3, 'none', 'npu')
        self.assertRtolEqual(cpu_output, npu_output)

    def test_soft_margin_loss_backward_float32_14(self):
        input1, input2, input3 = self.generate_data(-0.001, 0.001,
                                                    (10, 20, 30, 4),
                                                    (10, 20, 30, 4),
                                                    np.float32)
        cpu_output = self.op_exec(input1, input2, input3, 'mean', 'cpu')
        npu_output = self.op_exec(input1, input2, input3, 'mean', 'npu')
        self.assertRtolEqual(cpu_output, npu_output)

    def test_soft_margin_loss_backward_float32_15(self):
        input1, input2, input3 = self.generate_data(-0.001, 0.001,
                                                    (10, 20, 30, 4),
                                                    (10, 20, 30, 4),
                                                    np.float32)
        cpu_output = self.op_exec(input1, input2, input3, 'sum', 'cpu')
        npu_output = self.op_exec(input1, input2, input3, 'sum', 'npu')
        self.assertRtolEqual(cpu_output, npu_output)

    def test_soft_margin_loss_backward_float32_16(self):
        input1, input2, input3 = self.generate_data(-0.001, 0.001,
                                                    (100, 20, 3, 4, 5),
                                                    (100, 20, 3, 4, 5),
                                                    np.float32)
        cpu_output = self.op_exec(input1, input2, input3, 'none', 'cpu')
        npu_output = self.op_exec(input1, input2, input3, 'none', 'npu')
        self.assertRtolEqual(cpu_output, npu_output)

    def test_soft_margin_loss_backward_float32_17(self):
        input1, input2, input3 = self.generate_data(-0.001, 0.001,
                                                    (100, 20, 3, 4, 5),
                                                    (100, 20, 3, 4, 5),
                                                    np.float32)
        cpu_output = self.op_exec(input1, input2, input3, 'mean', 'cpu')
        npu_output = self.op_exec(input1, input2, input3, 'mean', 'npu')
        self.assertRtolEqual(cpu_output, npu_output)

    def test_soft_margin_loss_backward_float32_18(self):
        input1, input2, input3 = self.generate_data(-0.001, 0.001,
                                                    (100, 20, 3, 4, 5),
                                                    (100, 20, 3, 4, 5),
                                                    np.float32)
        cpu_output = self.op_exec(input1, input2, input3, 'sum', 'cpu')
        npu_output = self.op_exec(input1, input2, input3, 'sum', 'npu')
        self.assertRtolEqual(cpu_output, npu_output)


if __name__ == "__main__":
    run_tests()
