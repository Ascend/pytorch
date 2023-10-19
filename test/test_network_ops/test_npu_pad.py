import torch
import numpy as np
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests


class TestNpuPad(TestCase):
    def test_npu_pad(self, device="npu"):
        npu_input = torch.ones(2, 2).npu()
        pads = (1, 1, 1, 1)
        benchmark = torch.tensor([[0., 0., 0., 0.],
                                  [0., 1., 1., 0.],
                                  [0., 1., 1., 0.],
                                  [0., 0., 0., 0.]])
        npu_output = torch_npu.npu_pad(npu_input, pads)
        npu_output = npu_output.cpu().detach()
        self.assertRtolEqual(benchmark, npu_output)


if __name__ == "__main__":
    run_tests()
