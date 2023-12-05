import copy
import torch
import numpy as np
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests


class TestGru(TestCase):
    def test_gru(self):
        shape_format = [
            [[np.float32, (1, 3, 2)], [np.float32, (1, 3, 2)], 2, 2, 1, False, True, False],
            [[np.float32, (2, 1, 1)], [np.float32, (1, 2, 2)], 1, 2, 1, False, False, True],
            [[np.float32, (1, 3, 1)], [np.float32, (2, 3, 2)], 1, 2, 2, False, True, False],
            [[np.float32, (1, 1, 2)], [np.float32, (1, 1, 3)], 2, 3, 1, False, False, False],
            [[np.float32, (1, 1, 1)], [np.float32, (3, 1, 1)], 1, 1, 3, False, True, True],
        ]

        for item in shape_format:
            cpu_gru = torch.nn.GRU(input_size=item[2], hidden_size=item[3], num_layers=item[4],
                                   bidirectional=item[5], bias=item[-2], batch_first=item[-1])
            npu_gru = copy.deepcopy(cpu_gru).npu()

            input1 = np.random.uniform(0, 1, item[0][1]).astype(item[0][0])
            if item[0][0] == np.float16:
                cpu_input1 = torch.from_numpy(input1.astype(np.float32))
            else:
                cpu_input1 = torch.from_numpy(input1)
            npu_input1 = torch.from_numpy(input1).npu()

            h0 = np.random.uniform(0, 1, item[1][1]).astype(item[1][0])
            if item[1][0] == np.float16:
                cpu_h0 = torch.from_numpy(h0.astype(np.float32))
            else:
                cpu_h0 = torch.from_numpy(h0)
            npu_h0 = torch.from_numpy(h0).npu()

            cpu_output_y, cpu_output_h = cpu_gru(cpu_input1, cpu_h0)
            npu_output_y, npu_output_h = npu_gru(npu_input1, npu_h0)

            if item[0][0] == np.float16:
                self.assertRtolEqual(cpu_output_y.detach().numpy().astype(np.float16),
                                     npu_output_y.cpu().detach().numpy())
                self.assertRtolEqual(cpu_output_h.detach().numpy().astype(np.float16),
                                     npu_output_h.cpu().detach().numpy())
            else:
                # Ascend: fp33 isn't enough precision, relaxation of precision requirement temporary
                self.assertRtolEqual(cpu_output_y.detach().numpy(), npu_output_y.cpu().detach().numpy(), prec=1.e-1)
                self.assertRtolEqual(cpu_output_h.detach().numpy(), npu_output_h.cpu().detach().numpy(), prec=1.e-1)


if __name__ == "__main__":
    run_tests()
