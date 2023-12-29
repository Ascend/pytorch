import copy
import torch
import numpy as np
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests


class TestGru(TestCase):
    def test_gru(self):
        shape_format = [
            [[np.float32, (2, 3, 2)], [np.float32, (2, 2, 1)], 2, 1, 1, True, False, True],
            [[np.float32, (1, 1, 1)], [np.float32, (6, 1, 1)], 1, 1, 3, True, True, False],
            [[np.float32, (2, 1, 1)], [np.float32, (4, 1, 1)], 1, 1, 2, True, True, False],
            [[np.float32, (1, 2, 3)], [np.float32, (4, 1, 2)], 3, 2, 2, True, False, True],
            [[np.float32, (2, 2, 1)], [np.float32, (2, 2, 2)], 1, 2, 1, True, True, False],
            [[np.float32, (1, 2, 1)], [np.float32, (4, 1, 2)], 1, 2, 2, True, False, True],
        ]

        for item in shape_format:
            cpu_gru = torch.nn.GRU(input_size=item[2], hidden_size=item[3], num_layers=item[4],
                                   bidirectional=item[5], bias=item[-2], batch_first=item[-1])
            npu_gru = copy.deepcopy(cpu_gru).npu()

            input2 = np.random.uniform(0, 1, item[0][1]).astype(item[0][0])
            if item[0][0] == np.float16:
                cpu_input2 = torch.from_numpy(input2.astype(np.float32))
            else:
                cpu_input2 = torch.from_numpy(input2)
            npu_input2 = torch.from_numpy(input2).npu()

            h0 = np.random.uniform(0, 1, item[1][1]).astype(item[1][0])
            if item[1][0] == np.float16:
                cpu_h0 = torch.from_numpy(h0.astype(np.float32))
            else:
                cpu_h0 = torch.from_numpy(h0)
            npu_h0 = torch.from_numpy(h0).npu()

            cpu_output_y1, cpu_output_h1 = cpu_gru(cpu_input2, cpu_h0)
            npu_output_y1, npu_output_h1 = npu_gru(npu_input2, npu_h0)

            if item[0][0] == np.float16:
                self.assertRtolEqual(cpu_output_y1.detach().numpy().astype(np.float16),
                                     npu_output_y1.cpu().detach().numpy())
                self.assertRtolEqual(cpu_output_h1.detach().numpy().astype(np.float16),
                                     npu_output_h1.cpu().detach().numpy())
            else:
                # Ascend: fp33 isn't enough precision, relaxation of precision requirement temporary
                self.assertRtolEqual(cpu_output_y1.detach().numpy(), npu_output_y1.cpu().detach().numpy(), prec=1.e-1)
                self.assertRtolEqual(cpu_output_h1.detach().numpy(), npu_output_h1.cpu().detach().numpy(), prec=1.e-1)


if __name__ == "__main__":
    np.random.seed(1234)
    run_tests()
