import torch
from torch.testing._internal.common_utils import run_tests
from testutils import TestUtils
import torch_npu


class Test_issue70(TestUtils):
    def op_forward(self, x):
        return x.mean(-1)

    def test_issue70(self):
        compiled_net = torch.compile(self.op_forward, backend="inductor")

        arg = torch.randn((1, 1, 7168)).npu()

        output = self.op_forward(arg)
        output1 = compiled_net(arg)
        self.assertEqual(output, output1, atol=1e-3, rtol=1e-3)


if __name__ == "__main__":
    run_tests()
