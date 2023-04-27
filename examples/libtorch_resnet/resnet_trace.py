import os
import torch
import torchvision

from torch_npu.testing.testcase import TestCase, run_tests


class TestJitTrace(TestCase):

    def test_jit_trace(self):
        model = torchvision.models.resnet18()
        example_input = torch.rand(1, 3, 244, 244)

        resnet_model = torch.jit.trace(model, example_input)
        torch.jit.save(resnet_model, 'resnet_model.pt')
        assert os.path.isfile('./resnet_model.pt')


if __name__ == '__main__':
    run_tests()
