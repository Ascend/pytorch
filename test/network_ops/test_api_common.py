import unittest
import torch
import numpy as np

import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor, SupportedDevices


DEVICE_NAME = torch_npu.npu.get_device_name(0)[:10]
torch_npu.npu.set_compile_mode(jit_compile=False)


class TestApiCommon(TestCase):
    # test input tensor
    def cpu_op_out_exec_add(self, input1, input2, output):
        torch.add(input1, input2, alpha=1, out=output)
        output = output.numpy()
        return output

    def npu_op_out_exec_add(self, input1, input2, output):
        torch.add(input1, input2, alpha=1, out=output)
        output = output.to("cpu").numpy()
        return output

    def test_add_out_result(self):
        cpuout = torch.randn(18)
        npuout = torch.randn(18).to("npu")
        item = [np.float32, 0, [18]]
        cpu_input1, npu_input1 = create_common_tensor(item, 0, 100)
        cpu_input2, npu_input2 = create_common_tensor(item, 0, 100)

        cpu_output = self.cpu_op_out_exec_add(cpu_input1, cpu_input2, cpuout)
        npu_output = self.npu_op_out_exec_add(npu_input1, npu_input2, npuout)

        self.assertRtolEqual(cpu_output, npu_output)

    # test input scalar
    def cpu_op_scalar_exec_add(self, input1, scalar):
        output = torch.add(input1, scalar, alpha=1)
        output = output.numpy()
        return output

    def npu_op_scalar_exec_add(self, input1, scalar):
        output = torch.add(input1, scalar, alpha=1)
        output = output.to("cpu")
        output = output.numpy()
        return output

    def test_add_saclar_alpha_result(self):
        format_list = [0, 3]
        scalar_list = [0, 1]
        shape_format = [
            [[np.float16, i, [18]], k] for i in format_list for k in scalar_list
        ]
        for item in shape_format:
            cpu_input, npu_input = create_common_tensor(item[0], 0, 100)
            if cpu_input.dtype == torch.float16:
                cpu_input = cpu_input.to(torch.float32)
            cpu_output = self.cpu_op_scalar_exec_add(cpu_input, item[1])
            npu_output = self.npu_op_scalar_exec_add(npu_input, item[1])
            cpu_output = cpu_output.astype(npu_output.dtype)

            self.assertRtolEqual(cpu_output, npu_output)
    
    # test input IntArray
    def cpu_op_exec_AdaptiveAvgPool2d(self, input_x, input_grad):
        input_x.requires_grad_(True)
        m = torch.nn.AdaptiveAvgPool2d(input_grad)
        output = m(input_x)
        output.backward(output)
        out = output.detach(), input_x.grad
        return out

    def npu_op_exec_AdaptiveAvgPool2d(self, input_x, input_grad):
        input_x.requires_grad_(True)
        m = torch.nn.AdaptiveAvgPool2d(input_grad)
        output = m(input_x)
        output.backward(output)
        out = output.detach().cpu(), input_x.grad.cpu()
        return out

    def test_adaptiveAvgPool2d_backward(self):
        torch.manual_seed(123)
        cpu_input = torch.randn((1, 8, 9), dtype=torch.float32)
        npu_input = cpu_input.npu()
        output_size = np.array((2, 3))
        cpu_output = self.cpu_op_exec_AdaptiveAvgPool2d(cpu_input, output_size)
        npu_output = self.npu_op_exec_AdaptiveAvgPool2d(npu_input, output_size)
        self.assertRtolEqual(cpu_output[0], npu_output[0], prec=1e-3)
        self.assertRtolEqual(cpu_output[1], npu_output[1], prec=1e-3)

    # test input TensorList
    @SupportedDevices(['Ascend910B'])
    def test_foreach_add(self, device="npu"):
        input1 = torch.randn((1, 8, 9), dtype=torch.float32).npu()
        input2 = torch.randn((1, 8, 9), dtype=torch.float32).npu()
        npu_input_list = (input1, input2)
        npu_input_list2 = (input1, input2)
        scalar1 = torch.tensor(0.5, dtype=torch.float32).npu()
        npu_output = torch._foreach_add(npu_input_list, scalar1)

    # test input ScalarList    
    # test input c10::ArrayRef<at::Scalar>
    @SupportedDevices(['Ascend910B'])
    def test_foreach_add_ScalarList(self, device="npu"):
        input1 = torch.randn((1, 8, 9), dtype=torch.float32).npu()
        input2 = torch.randn((1, 8, 9), dtype=torch.float32).npu()
        npu_input_list = (input1, input2)
        scalar1 = torch.tensor(0.5, dtype=torch.float32).npu()
        scalar2 = torch.tensor(0.5, dtype=torch.float32).npu()
        scalar_list = (scalar1, scalar2)
        npu_output = torch._foreach_add(npu_input_list, scalar_list)

    # test c10::optional<at::Tensor>
    def test_batch_norm_backward_elemt_4d(self):
        grad_output = torch.ones([2, 3, 1, 4]).npu()
        input1 = torch.ones([2, 3, 1, 4]).npu()
        mean = torch.tensor([8., 5., 9.]).npu()
        invstd = torch.tensor([2., 1., 2.]).npu()
        weight = torch.tensor([1., 1., 4.]).npu()
        mean_dy = torch.tensor([2., 2., 6.]).npu()
        mean_dy_xmn = torch.tensor([2., 3., 11.]).npu()
        count_tensor = torch.tensor([5, 5, 5], dtype=torch.int32).npu()

        grad_input = torch.batch_norm_backward_elemt(grad_output, input1, mean, invstd,
                                                     weight, mean_dy, mean_dy_xmn, count_tensor)
        cuda_expect_out = torch.tensor([[[[9.2000, 9.2000, 9.2000, 9.2000]],
                                         [[1.6667, 1.6667, 1.6667, 1.6667]],
                                         [[192.5333, 192.5333, 192.5333, 192.5333]]],
                                        [[[9.2000, 9.2000, 9.2000, 9.2000]],
                                         [[1.6667, 1.6667, 1.6667, 1.6667]],
                                         [[192.5333, 192.5333, 192.5333, 192.5333]]]])
        self.assertRtolEqual(grad_input.cpu(), cuda_expect_out)

    # test c10::optional<at::IntArrayRef>、at::OptionalIntArrayRef
    def cpu_op_exec_mean(self, input1, dtype):
        output = torch.mean(input1, [2, 3], keepdim=True, dtype=dtype)
        output = output.numpy()
        return output

    def npu_op_exec_mean(self, input1, dtype):
        input1 = input1.to("npu")
        output = torch.mean(input1, [2, 3], keepdim=True, dtype=dtype)
        output = output.to("cpu")
        output = output.numpy()
        return output

    def test_mean_shape_format(self):
        item = [[np.float32, 3, (256, 1280, 7, 7)], torch.float32]
        cpu_input, npu_input = create_common_tensor(item[0], 0, 100)
        cpu_output = self.cpu_op_exec_mean(cpu_input, dtype=item[-1])
        npu_output = self.npu_op_exec_mean(npu_input, dtype=item[-1])
        self.assertRtolEqual(cpu_output, npu_output)

    # test String
    # test c10::optional<at::Scalar>
    def test_clamp(self):
        item = [[np.float32, 0, (4, 3)], [np.float32, 0, (4, 3)], [np.float32, 0, (4, 3)]]
        input_cpu, input_npu = create_common_tensor(item[0], 0, 10)
        min_cpu, min_npu = create_common_tensor(item[1], 1, 50)
        max_cpu, max_npu = create_common_tensor(item[2], 50, 100)
        _, out_npu = create_common_tensor(item[0], 1, 100)

        cpu_output = torch.clamp(input_cpu, min_cpu, max_cpu).numpy()
        npu_output = torch.clamp(input_npu, min_npu, max_npu).cpu().numpy()
        self.assertRtolEqual(cpu_output, npu_output)


if __name__ == "__main__":
    run_tests()
