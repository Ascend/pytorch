import numpy as np
import torch

import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor


class TestScatterUpdate(TestCase):

    def supported_scatter_update_exec(self, var, updates, start, length, axis=-2):
        var_input = var.clone()
        var_input[:, :, start: start + length, :] = updates
        return var_input

    def custom_scatter_update_exec(self, var, indices, updates, axis=-2):
        output = torch_npu.scatter_update(var, indices, updates, axis)
        return output.cpu().detach()


    def supported_scatter_update__exec(self, var, updates, start, length, axis=-2):
        var[:, :, start: start + length, :] = updates
        return var

    def custom_scatter_update__exec(self, var, indices, updates, axis=-2):
        torch_npu.scatter_update_(var, indices, updates, axis)
        return var.cpu().detach()

    def custom_scatter_update__exec_return(self, var, indices, updates, axis=-2):
        result = torch_npu.scatter_update_(var, indices, updates, axis)
        return result.cpu().detach()


    def test_scatter_update(self, device="npu"):
        # input_dtype, slice_start, slice_length
        items = [
            [torch.float16, 0, 1],
            [torch.float16, 2, 1],
            [torch.float16, 4, 8],
            [torch.float16, 0, 16],
            [torch.float32, 0, 1],
            [torch.float32, 2, 1],
            [torch.float32, 4, 8],
            [torch.float32, 0, 16],
        ]

        for item in items:
            in_self_cpu = torch.randn(4, 8, 16, 64, dtype=item[0])
            in_self_npu = in_self_cpu.npu()
            in_update_cpu = torch.randn(4, 8, item[2], 64, dtype=item[0])
            in_update_npu = in_update_cpu.npu()
            in_indices_npu = torch.tensor([item[1], item[1], item[1], item[1]]).npu()

            supported_output = self.supported_scatter_update_exec(in_self_cpu, in_update_cpu, item[1], item[2])
            custom_output = self.custom_scatter_update_exec(in_self_npu, in_indices_npu, in_update_npu)
            self.assertRtolEqual(supported_output, custom_output)
            # check whether the custom operator modifies the input_self
            self.assertRtolEqual(in_self_cpu, in_self_npu.cpu().detach())


    def test_scatter_update_(self, device="npu"):
        # input_dtype, slice_start, slice_length
        items = [
            [torch.float16, 0, 1],
            [torch.float16, 4, 1],
            [torch.float16, 8, 32],
            [torch.float16, 0, 64],
            [torch.float32, 0, 1],
            [torch.float32, 4, 1],
            [torch.float32, 8, 32],
            [torch.float32, 0, 64],
        ]
        for item in items:
            in_self_cpu = torch.randn(4, 8, 64, 128, dtype=item[0])
            in_self_npu = in_self_cpu.npu()
            in_update_cpu = torch.randn(4, 8, item[2], 128, dtype=item[0])
            in_update_npu = in_update_cpu.npu()
            in_indices_npu = torch.tensor([item[1], item[1], item[1], item[1]]).npu()

            supported_output = self.supported_scatter_update__exec(in_self_cpu, in_update_cpu, item[1], item[2])
            custom_output = self.custom_scatter_update__exec(in_self_npu, in_indices_npu, in_update_npu)
            self.assertRtolEqual(supported_output, custom_output)
            # check whether the custom operator modifies the input_self
            self.assertRtolEqual(in_self_cpu, in_self_npu.cpu().detach())


    def test_scatter_update__return(self, device="npu"):
        # input_dtype, slice_start, slice_length
        items = [
            [torch.float16, 0, 1],
            [torch.float16, 8, 1],
            [torch.float16, 16, 8],
            [torch.float16, 0, 32],
            [torch.float32, 0, 1],
            [torch.float32, 8, 1],
            [torch.float32, 16, 8],
            [torch.float32, 0, 32],
        ]
        for item in items:
            in_self_cpu = torch.randn(12, 2, 64, 128, dtype=item[0])
            in_self_npu = in_self_cpu.npu()
            in_update_cpu = torch.randn(12, 2, item[2], 128, dtype=item[0])
            in_update_npu = in_update_cpu.npu()
            in_indices_npu = torch.tensor([item[1], item[1], item[1], item[1], item[1], item[1],
                                           item[1], item[1], item[1], item[1], item[1], item[1]]).npu()

            supported_output = self.supported_scatter_update__exec(in_self_cpu, in_update_cpu, item[1], item[2])
            custom_output = self.custom_scatter_update__exec(in_self_npu, in_indices_npu, in_update_npu)
            self.assertRtolEqual(supported_output, custom_output)
            # check whether the custom operator modifies the input_self
            self.assertRtolEqual(in_self_cpu, in_self_npu.cpu().detach())


if __name__ == "__main__":
    run_tests()
