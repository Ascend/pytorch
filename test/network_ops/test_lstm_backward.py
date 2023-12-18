import copy
import unittest
import torch
import numpy as np

import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor


class TestLstmBackward(TestCase):

    def test_lstm_backward(self):
        # 注：shape_format:[[dtype, (num_step, batch_size, input_size)], input_size, hidden_size, is_training]
        shape_format = [
            [[np.float32, (16, 32, 64)], 64, 32, True],
            [[np.float32, (5, 32, 64)], 64, 32, True],
        ]

        for item in shape_format:
            cpu_lstm = torch.nn.LSTM(input_size=item[1], hidden_size=item[2],
                                     num_layers=1, bidirectional=False, bias=False)
            cpu_lstm.training = item[3]
            npu_lstm = copy.deepcopy(cpu_lstm).npu()

            input1 = np.random.uniform(0, 1, item[0][1]).astype(np.float16).astype(np.float32)
            cpu_input1 = torch.from_numpy(input1)
            cpu_input1.requires_grad_(True)
            cpu_output_y, (cpu_output_h, cpu_output_c) = cpu_lstm(cpu_input1)

            npu_input1 = torch.from_numpy(input1.astype(item[0][0])).npu()
            npu_input1.requires_grad_(True)
            npu_output_y, (npu_output_h, npu_output_c) = npu_lstm(npu_input1)

            self.assertRtolEqual(cpu_output_y.detach().numpy(),
                                 npu_output_y.cpu().to(torch.float).detach().numpy(), prec=1.e-3)
            self.assertRtolEqual(cpu_output_h.detach().numpy(),
                                 npu_output_h.cpu().to(torch.float).detach().numpy(), prec=1.e-3)
            self.assertRtolEqual(cpu_output_c.detach().numpy(),
                                 npu_output_c.cpu().to(torch.float).detach().numpy(), prec=2.e-3)

            cpu_input1.retain_grad()
            cpu_output_y.backward(torch.ones(cpu_output_y.size(), dtype=torch.float))
            cpu_dx = cpu_input1.grad

            npu_input1.retain_grad()
            npu_output_y.backward(torch.ones(npu_output_y.size(), dtype=torch.float).npu())
            npu_dx = npu_input1.grad

            self.assertRtolEqual(cpu_dx.numpy(), npu_dx.cpu().to(torch.float).numpy(), prec=1.e-3)

    def test_lstm_bidirection_backward(self):
        # 注：shape_format:[[dtype, (num_step, batch_size, input_size)], input_size, hidden_size, is_training]
        shape_format = [
            [[np.float32, (16, 32, 64)], 64, 32, True],
            [[np.float32, (5, 32, 64)], 64, 32, True],
        ]

        for item in shape_format:
            cpu_lstm = torch.nn.LSTM(input_size=item[1],
                                     hidden_size=item[2], num_layers=1, bidirectional=True, bias=False)
            cpu_lstm.training = item[3]
            npu_lstm = copy.deepcopy(cpu_lstm).npu()

            input1 = np.random.uniform(0, 1, item[0][1]).astype(np.float16).astype(np.float32)

            cpu_input1 = torch.from_numpy(input1)
            cpu_input1.requires_grad_(True)
            cpu_output_y, (cpu_output_h, cpu_output_c) = cpu_lstm(cpu_input1)

            npu_input1 = torch.from_numpy(input1.astype(item[0][0])).npu()
            npu_input1.requires_grad_(True)
            npu_output_y, (npu_output_h, npu_output_c) = npu_lstm(npu_input1)

            self.assertRtolEqual(cpu_output_y.detach().numpy(),
                                 npu_output_y.cpu().to(torch.float).detach().numpy(), prec=1.e-3)
            self.assertRtolEqual(cpu_output_h.detach().numpy(),
                                 npu_output_h.cpu().to(torch.float).detach().numpy(), prec=1.e-3)
            self.assertRtolEqual(cpu_output_c.detach().numpy(),
                                 npu_output_c.cpu().to(torch.float).detach().numpy(), prec=2.e-3)

            cpu_input1.retain_grad()
            cpu_output_y.backward(torch.ones(cpu_output_y.size(), dtype=torch.float))
            cpu_dx = cpu_input1.grad

            npu_input1.retain_grad()
            npu_output_y.backward(torch.ones(npu_output_y.size(), dtype=torch.float).npu())
            npu_dx = npu_input1.grad

            self.assertRtolEqual(cpu_dx.numpy(), npu_dx.cpu().to(torch.float).numpy(), prec=1.e-3)

    def test_lstm_backward_output_shape(self):
        cpu_lstm = torch.nn.LSTM(input_size=512, hidden_size=512, num_layers=2, bias=True,
                                 batch_first=True, bidirectional=False)
        cpu_lstm.training = True
        npu_lstm = copy.deepcopy(cpu_lstm).npu()
        shape_format = [
            [np.float16, 2, (8, 1, 512)],  # the operator uses fp16  for calculation.
            [np.float16, 2, (2, 8, 512)],
        ]
        cpu_input1, npu_input1 = create_common_tensor(shape_format[0], 0, 1)
        cpu_input2, npu_input2 = create_common_tensor(shape_format[1], 0, 1)
        cpu_input3, npu_input3 = create_common_tensor(shape_format[1], 0, 1)
        cpu_input1.requires_grad = True
        cpu_input2.requires_grad = True
        cpu_input3.requires_grad = True
        npu_input1.requires_grad = True
        npu_input2.requires_grad = True
        npu_input3.requires_grad = True

        cpu_output_y, (cpu_output_h, cpu_output_c) = cpu_lstm(cpu_input1.type(torch.float32),
                                                              (cpu_input2.type(torch.float32), cpu_input3.type(torch.float32)))
        npu_output_y, (npu_output_h, npu_output_c) = npu_lstm(npu_input1.type(torch.float32),
                                                              (npu_input2.type(torch.float32), npu_input3.type(torch.float32)))
        cpu_output_y.sum().backward()
        npu_output_y.sum().backward()
        self.assertRtolEqual(npu_input1.grad.cpu().numpy(), cpu_input1.grad.numpy(), prec=1.e-3)


if __name__ == "__main__":
    run_tests()
