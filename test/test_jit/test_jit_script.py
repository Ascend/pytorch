import torch

import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests


class TestJitTrace(TestCase):
    def test_script_npu_max(self):
        class NpuModel(torch.nn.Module):
            def __init__(self):
                super(NpuModel, self).__init__()

            def forward(self, x):
                x = torch_npu.npu_max(x, dim=1)
                return x

        example_input = torch.rand(2, 8).npu()
        model = NpuModel().to("npu")
        output1 = model(example_input)

        script_model = torch.jit.script(model)
        output2 = script_model(example_input)
        self.assertRtolEqual(output1, output2)


if __name__ == '__main__':
    run_tests()
