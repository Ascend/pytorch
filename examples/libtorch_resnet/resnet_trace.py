import os
import torch
import torchvision

from torch_npu.testing.testcase import TestCase, run_tests


class TestJitTrace(TestCase):

    def test_jit_trace_load_script(self):
        model = torchvision.models.resnet18().eval().to('npu:0')
        example_input = torch.rand(1, 3, 244, 244).to('npu:0')

        output = model(example_input)

        resnet_model = torch.jit.trace(model, example_input)
        torch.jit.save(resnet_model, 'resnet_model.pt')
        assert os.path.isfile('./resnet_model.pt')

        trace_model = torch.jit.load('./resnet_model.pt')
        trace_output = trace_model(example_input)
        self.assertRtolEqual(trace_output, output)

        script_model = torch.jit.script(model)
        script_output = script_model(example_input)
        self.assertRtolEqual(script_output, output)


if __name__ == '__main__':
    run_tests()
