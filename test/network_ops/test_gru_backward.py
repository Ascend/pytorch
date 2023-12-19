import copy
import torch
import numpy as np

import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests


class TestGruBackward(TestCase):

    def test_gru_backward(self):
        shape_format = [
            [[np.float16, (16, 32, 64)], 64, 32],
            [[np.float16, (5, 32, 64)], 64, 32],
            [[np.float32, (5, 32, 64)], 64, 32],
            [[np.float32, (5, 32, 64)], 64, 64],
            [[np.float32, (5, 1, 64)], 64, 64],
        ]

        for item in shape_format:
            cpu_gru = torch.nn.GRU(input_size=item[1], hidden_size=item[2], num_layers=1, bidirectional=False)
            npu_gru = copy.deepcopy(cpu_gru).npu()
            if item[0][0] == np.float16:
                npu_gru = npu_gru.half()

            input1 = np.random.uniform(0, 1, item[0][1]).astype(item[0][0])
            cpu_input1 = torch.from_numpy(input1.astype(np.float32))
            cpu_input1.requires_grad_(True)
            npu_input1 = torch.from_numpy(input1).npu()
            npu_input1.requires_grad_(True)

            cpu_output_y, cpu_output_h = cpu_gru(cpu_input1)
            npu_output_y, npu_output_h = npu_gru(npu_input1)

            self.assertRtolEqual(cpu_output_y.detach().numpy(), npu_output_y.cpu().detach().numpy().astype(np.float32),
                                 prec=1.e-1)
            self.assertRtolEqual(cpu_output_h.detach().numpy(), npu_output_h.cpu().detach().numpy().astype(np.float32),
                                 prec=1.e-1)

            cpu_input1.retain_grad()
            cpu_output_y.backward(torch.ones(cpu_output_y.size(), dtype=torch.float))
            cpu_dx = cpu_input1.grad
            cpu_dw_ih = cpu_gru.weight_ih_l0.grad
            cpu_dw_hh = cpu_gru.weight_hh_l0.grad
            cpu_db_ih = cpu_gru.bias_ih_l0.grad
            cpu_db_hh = cpu_gru.bias_hh_l0.grad

            npu_input1.retain_grad()
            npu_output_y.backward(torch.ones(npu_output_y.size(), dtype=torch.float).npu())
            npu_dx = npu_input1.grad
            npu_dw_ih = npu_gru.weight_ih_l0.grad
            npu_dw_hh = npu_gru.weight_hh_l0.grad
            npu_db_ih = npu_gru.bias_ih_l0.grad
            npu_db_hh = npu_gru.bias_hh_l0.grad

            self.assertRtolEqual(cpu_dx.numpy(), npu_dx.cpu().numpy().astype(np.float32), prec=1.e-1)
            self.assertRtolEqual(cpu_dw_ih.numpy(), npu_dw_ih.cpu().numpy().astype(np.float32), prec=1.e-1)
            self.assertRtolEqual(cpu_dw_hh.numpy(), npu_dw_hh.cpu().numpy().astype(np.float32), prec=1.e-1)
            self.assertRtolEqual(cpu_db_ih.numpy(), npu_db_ih.cpu().numpy().astype(np.float32), prec=1.e1)
            self.assertRtolEqual(cpu_db_hh.numpy(), npu_db_hh.cpu().numpy().astype(np.float32), prec=1.e1)


if __name__ == "__main__":
    run_tests()
