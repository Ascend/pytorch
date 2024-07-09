import torch
import torch.nn as nn

import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests


class TestInit(TestCase):
    def test_xavier_uniform_(self):
        m = nn.Conv2d(3, 3, 1)
        n = m.npu()
        nn.init.xavier_uniform_(m.weight, gain=True)
        nn.init.xavier_uniform_(n.weight, gain=True)
        self.assertEqual(m.weight.requires_grad, n.weight.requires_grad)

        m = nn.Conv2d(3, 3, 1)
        n = m.npu()
        nn.init.xavier_uniform_(m.weight, gain=False)
        nn.init.xavier_uniform_(n.weight, gain=False)
        self.assertEqual(m.weight.requires_grad, n.weight.requires_grad)

        m = nn.Conv2d(3, 3, 1)
        n = m.npu()
        nn.init.xavier_normal_(m.weight, gain=True)
        nn.init.xavier_normal_(n.weight, gain=True)
        self.assertEqual(m.weight.requires_grad, n.weight.requires_grad)


if __name__ == "__main__":
    run_tests()
