import torch
import torch.nn.functional as F
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests


class TestVisionFunctions(TestCase):
    def test_pixel_shuffle(self):
        input1 = torch.randn(1, 9, 4, 4)

        npu_input = input1.npu()

        cpu_output = F.pixel_shuffle(input1, 3)
        npu_output = F.pixel_shuffle(npu_input, 3)

        self.assertRtolEqual(cpu_output.detach().numpy(), npu_output.detach().cpu().numpy())

    def test_pad(self):
        input1 = torch.empty(3, 3, 4, 2)
        p1d = (1, 1)
        npu_input = input1.npu()

        cpu_output = F.pad(input1, p1d, "constant", 0)
        npu_output = F.pad(npu_input, p1d, "constant", 0)

        self.assertRtolEqual(cpu_output.detach().numpy(), npu_output.detach().cpu().numpy())

    def test_interpolate(self):
        input1 = torch.empty(3, 3, 4, 2)
        npu_input = input1.npu()

        cpu_output = F.interpolate(input1, 4)
        npu_output = F.interpolate(npu_input, 4)

        self.assertRtolEqual(cpu_output.detach().numpy(), npu_output.detach().cpu().numpy())

    def test_upsample(self):
        input1 = torch.empty(3, 3, 4, 2)
        npu_input = input1.npu()

        cpu_output = F.upsample(input1, 4)
        npu_output = F.upsample(npu_input, 4)

        self.assertRtolEqual(cpu_output.detach().numpy(), npu_output.detach().cpu().numpy())

    def test_upsample_nearest(self):
        input1 = torch.empty(3, 3, 4, 2)
        npu_input = input1.npu()

        cpu_output = F.upsample_nearest(input1, 4)
        npu_output = F.upsample_nearest(npu_input, 4)

        self.assertRtolEqual(cpu_output.detach().numpy(), npu_output.detach().cpu().numpy())

    def test_grid_sample(self):
        input1 = torch.empty(1, 1, 2, 2)
        grid = torch.empty(1, 1, 1, 2)

        npu_input = input1.npu()
        npu_grid = grid.npu()

        cpu_output = F.grid_sample(input1, grid)
        npu_output = F.grid_sample(npu_input, npu_grid)

        self.assertRtolEqual(cpu_output.detach().numpy(), npu_output.detach().cpu().numpy())

    def test_affine_grid(self):
        '''
        Because of the limitation of NPU op, the NPU op will automatically convert the input 
        fp32 to fp16 for calculation, so the input must be passed data within the representable
        range of fp16.
        '''
        input1 = torch.arange(1., 7).view(1, 2, 3)
        size = torch.Size([1, 1, 2, 2])

        npu_input = input1.npu()

        cpu_output = F.affine_grid(input1, size)
        npu_output = F.affine_grid(npu_input, size)

        self.assertRtolEqual(cpu_output.detach().numpy(), npu_output.detach().cpu().numpy())


if __name__ == "__main__":
    run_tests()
