import torch
import torch.nn as nn
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests


class TestSparseLayers(TestCase):
    def test_Embedding(self):
        embedding = nn.Embedding(10, 3).npu()
        input1 = torch.LongTensor([[1, 2, 4, 5], [4, 3, 2, 9]]).npu()
        output = embedding(input1)
        self.assertEqual(output is not None, True)


if __name__ == "__main__":
    run_tests()
