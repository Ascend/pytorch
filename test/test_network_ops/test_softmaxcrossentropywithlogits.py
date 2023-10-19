import torch
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests


class TestSoftmaxCrossentropyWithLogits(TestCase):
    def npu_op_exec(self, input1, label):
        output = torch_npu.npu_softmax_cross_entropy_with_logits(input1, label)
        output = output.to("cpu")
        output = output.numpy()
        return output

    def test_softmaxcross(self, device="npu"):
        input1 = torch.tensor([[1., 2., 3., 4.]]).npu()
        label = torch.tensor([[1., 2., 3., 4.]]).npu()
        exresult = torch.tensor([14.4019])
        output = self.npu_op_exec(input1, label)
        self.assertRtolEqual(exresult.numpy(), output)


if __name__ == "__main__":
    run_tests()
