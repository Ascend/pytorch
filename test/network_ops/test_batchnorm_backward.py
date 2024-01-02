import copy
import torch
import numpy as np

import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor


class TestBatchNorm(TestCase):
    def cpu_op_exec(self, input1, bn):
        input1 = input1.float()
        input1.requires_grad_(True)
        bn.requires_grad_(True)

        # get forward results
        output = bn(input1)
        running_mean = bn.running_mean
        running_var = bn.running_var

        # get backward results
        output.backward(torch.ones_like(output))
        input_grad = input1.grad
        weight_grad = bn.weight.grad
        bias_grad = bn.bias.grad

        # results to numpy
        output = output.detach().numpy()
        running_mean = running_mean.detach().numpy()
        running_var = running_var.detach().numpy()
        input_grad = input_grad.detach().numpy()
        weight_grad = weight_grad.detach().numpy()
        bias_grad = bias_grad.detach().numpy()

        return output, running_mean, running_var, input_grad, weight_grad, bias_grad

    def npu_op_exec(self, input1, bn):
        input1.requires_grad_(True)
        bn.requires_grad_(True)

        # get forward results
        output = bn(input1)
        running_mean = bn.running_mean
        running_var = bn.running_var

        # get backward results
        output.backward(torch.ones_like(output))
        input_grad = input1.grad
        weight_grad = bn.weight.grad
        bias_grad = bn.bias.grad

        # results to numpy
        output = output.detach().cpu().float().numpy()
        running_mean = running_mean.detach().cpu().float().numpy()
        running_var = running_var.detach().cpu().float().numpy()
        input_grad = input_grad.detach().cpu().float().numpy()
        weight_grad = weight_grad.detach().cpu().float().numpy()
        bias_grad = bias_grad.detach().cpu().float().numpy()

        return output, running_mean, running_var, input_grad, weight_grad, bias_grad

    def test_BatchNorm1D_float32(self):
        np.random.seed(1234)
        format_list = [-1]
        shape_list = [[256, 672], [58, 28, 16]]
        shape_format = [
            [np.float32, i, j] for i in format_list for j in shape_list
        ]
        for item in shape_format:
            # create inputs and params
            cpu_input1, npu_input1 = create_common_tensor(item, -1, 1)

            dim_c = item[2][1]
            params_item = [np.float32, -1, [dim_c]]
            weight, _ = create_common_tensor(params_item, -1, 1)
            bias, _ = create_common_tensor(params_item, -1, 1)
            running_mean, _ = create_common_tensor(params_item, -1, 1)
            running_var, _ = create_common_tensor(params_item, -1, 1)

            # create BatchNorm2D
            cpu_bn1d = torch.nn.BatchNorm1d(dim_c)
            cpu_bn1d.weight.data = weight
            cpu_bn1d.bias.data = bias
            cpu_bn1d.running_mean.data = running_mean
            cpu_bn1d.running_var.data = running_var

            npu_bn1d = copy.deepcopy(cpu_bn1d).to("npu")

            # run exec
            cpu_output, cpu_running_mean, cpu_running_var, cpu_input_grad, cpu_weight_grad, cpu_bias_grad \
                = self.cpu_op_exec(cpu_input1, cpu_bn1d)

            npu_output, npu_running_mean, npu_running_var, npu_input_grad, npu_weight_grad, npu_bias_grad \
                = self.npu_op_exec(npu_input1, npu_bn1d)

            # comparison results
            self.assertRtolEqual(cpu_output, npu_output)
            self.assertRtolEqual(cpu_running_mean, npu_running_mean)
            self.assertRtolEqual(cpu_running_var, npu_running_var)
            self.assertRtolEqual(cpu_input_grad, npu_input_grad)
            self.assertRtolEqual(cpu_weight_grad, npu_weight_grad, 1e-2)
            self.assertRtolEqual(cpu_bias_grad, npu_bias_grad)

    def test_BatchNorm1D_float16(self):
        np.random.seed(1234)
        format_list = [-1]
        shape_list = [[256, 672], [58, 28, 16]]
        shape_format = [
            [np.float16, i, j] for i in format_list for j in shape_list
        ]
        for item in shape_format:
            # create inputs and params
            cpu_input1, npu_input1 = create_common_tensor(item, -1, 1)

            dim_c = item[2][1]
            params_item = [np.float32, -1, [dim_c]]
            weight, _ = create_common_tensor(params_item, -1, 1)
            bias, _ = create_common_tensor(params_item, -1, 1)
            running_mean, _ = create_common_tensor(params_item, -1, 1)
            running_var, _ = create_common_tensor(params_item, -1, 1)

            # create BatchNorm2D
            cpu_bn1d = torch.nn.BatchNorm1d(dim_c)
            cpu_bn1d.weight.data = weight
            cpu_bn1d.bias.data = bias
            cpu_bn1d.running_mean.data = running_mean
            cpu_bn1d.running_var.data = running_var

            npu_bn1d = copy.deepcopy(cpu_bn1d).to("npu")

            # run exec
            cpu_output, cpu_running_mean, cpu_running_var, cpu_input_grad, cpu_weight_grad, cpu_bias_grad \
                = self.cpu_op_exec(cpu_input1, cpu_bn1d)

            npu_output, npu_running_mean, npu_running_var, npu_input_grad, npu_weight_grad, npu_bias_grad \
                = self.npu_op_exec(npu_input1, npu_bn1d)

            # comparison results
            self.assertRtolEqual(cpu_output, npu_output, 1e-3)
            self.assertRtolEqual(cpu_running_mean, npu_running_mean)
            self.assertRtolEqual(cpu_running_var, npu_running_var)
            self.assertRtolEqual(cpu_input_grad, npu_input_grad)
            self.assertRtolEqual(cpu_weight_grad, npu_weight_grad, 1e-2)
            self.assertRtolEqual(cpu_bias_grad, npu_bias_grad)

    def test_BatchNorm2D_float32(self):
        np.random.seed(1234)
        format_list = [-1]
        shape_list = [[256, 672, 7, 7], [1024, 58, 28, 28]]
        shape_format = [
            [np.float32, i, j] for i in format_list for j in shape_list
        ]
        for item in shape_format:
            # create inputs and params
            cpu_input1, npu_input1 = create_common_tensor(item, -1, 1)

            dim_c = item[2][1]
            params_item = [np.float32, -1, [dim_c]]
            weight, _ = create_common_tensor(params_item, -1, 1)
            bias, _ = create_common_tensor(params_item, -1, 1)
            running_mean, _ = create_common_tensor(params_item, -1, 1)
            running_var, _ = create_common_tensor(params_item, -1, 1)

            # create BatchNorm2D
            cpu_bn2d = torch.nn.BatchNorm2d(dim_c)
            cpu_bn2d.weight.data = weight
            cpu_bn2d.bias.data = bias
            cpu_bn2d.running_mean.data = running_mean
            cpu_bn2d.running_var.data = running_var

            npu_bn2d = copy.deepcopy(cpu_bn2d).to("npu")

            # run exec
            cpu_output, cpu_running_mean, cpu_running_var, cpu_input_grad, cpu_weight_grad, cpu_bias_grad \
                = self.cpu_op_exec(cpu_input1, cpu_bn2d)

            npu_output, npu_running_mean, npu_running_var, npu_input_grad, npu_weight_grad, npu_bias_grad \
                = self.npu_op_exec(npu_input1, npu_bn2d)

            # comparison results
            self.assertRtolEqual(cpu_output, npu_output)
            self.assertRtolEqual(cpu_running_mean, npu_running_mean)
            self.assertRtolEqual(cpu_running_var, npu_running_var)
            self.assertRtolEqual(cpu_input_grad, npu_input_grad)
            self.assertRtolEqual(cpu_weight_grad, npu_weight_grad, 1e-2)
            self.assertRtolEqual(cpu_bias_grad, npu_bias_grad)

    def test_BatchNorm2D_float16(self):
        np.random.seed(1234)
        format_list = [-1]
        shape_list = [[256, 672, 7, 7], [1024, 58, 28, 28]]
        shape_format = [
            [np.float16, i, j] for i in format_list for j in shape_list
        ]
        for item in shape_format:
            # create inputs and params
            cpu_input1, npu_input1 = create_common_tensor(item, -1, 1)

            dim_c = item[2][1]
            params_item = [np.float32, -1, [dim_c]]
            weight, _ = create_common_tensor(params_item, -1, 1)
            bias, _ = create_common_tensor(params_item, -1, 1)
            running_mean, _ = create_common_tensor(params_item, -1, 1)
            running_var, _ = create_common_tensor(params_item, -1, 1)

            # create BatchNorm2D
            cpu_bn2d = torch.nn.BatchNorm2d(dim_c)
            cpu_bn2d.weight.data = weight
            cpu_bn2d.bias.data = bias
            cpu_bn2d.running_mean.data = running_mean
            cpu_bn2d.running_var.data = running_var

            npu_bn2d = copy.deepcopy(cpu_bn2d).to("npu")

            # run exec
            cpu_output, cpu_running_mean, cpu_running_var, cpu_input_grad, cpu_weight_grad, cpu_bias_grad \
                = self.cpu_op_exec(cpu_input1, cpu_bn2d)

            npu_output, npu_running_mean, npu_running_var, npu_input_grad, npu_weight_grad, npu_bias_grad \
                = self.npu_op_exec(npu_input1, npu_bn2d)

            # comparison results
            self.assertRtolEqual(cpu_output, npu_output, 1e-3)
            self.assertRtolEqual(cpu_running_mean, npu_running_mean)
            self.assertRtolEqual(cpu_running_var, npu_running_var)
            self.assertRtolEqual(cpu_input_grad, npu_input_grad)
            self.assertRtolEqual(cpu_weight_grad, npu_weight_grad, 1e-1)
            self.assertRtolEqual(cpu_bias_grad, npu_bias_grad)

    def test_BatchNorm3D_float32(self):
        np.random.seed(1234)
        format_list = [-1]
        shape_list = [[8, 512, 4, 28, 28], [8, 256, 8, 56, 56]]
        shape_format = [
            [np.float32, i, j] for i in format_list for j in shape_list
        ]
        for item in shape_format:
            # create inputs and params
            cpu_input1, npu_input1 = create_common_tensor(item, -1, 1)

            dim_c = item[2][1]
            params_item = [np.float32, -1, [dim_c]]
            weight, _ = create_common_tensor(params_item, -1, 1)
            bias, _ = create_common_tensor(params_item, -1, 1)
            running_mean, _ = create_common_tensor(params_item, -1, 1)
            running_var, _ = create_common_tensor(params_item, -1, 1)

            # create BatchNorm2D
            cpu_bn3d = torch.nn.BatchNorm3d(dim_c)
            cpu_bn3d.weight.data = weight
            cpu_bn3d.bias.data = bias
            cpu_bn3d.running_mean.data = running_mean
            cpu_bn3d.running_var.data = running_var

            npu_bn3d = copy.deepcopy(cpu_bn3d).to("npu")

            # run exec
            cpu_output, cpu_running_mean, cpu_running_var, cpu_input_grad, cpu_weight_grad, cpu_bias_grad \
                = self.cpu_op_exec(cpu_input1, cpu_bn3d)

            npu_output, npu_running_mean, npu_running_var, npu_input_grad, npu_weight_grad, npu_bias_grad \
                = self.npu_op_exec(npu_input1, npu_bn3d)

            # comparison results
            self.assertRtolEqual(cpu_output, npu_output)
            self.assertRtolEqual(cpu_running_mean, npu_running_mean)
            self.assertRtolEqual(cpu_running_var, npu_running_var)
            self.assertRtolEqual(cpu_input_grad, npu_input_grad)
            self.assertRtolEqual(cpu_weight_grad, npu_weight_grad, 1e-2)
            self.assertRtolEqual(cpu_bias_grad, npu_bias_grad)

    def test_BatchNorm3D_float16(self):
        np.random.seed(1234)
        format_list = [-1]
        shape_list = [[8, 512, 4, 28, 28], [2, 256, 8, 16, 56]]
        shape_format = [
            [np.float16, i, j] for i in format_list for j in shape_list
        ]
        for item in shape_format:
            # create inputs and params
            cpu_input1, npu_input1 = create_common_tensor(item, -1, 1)

            dim_c = item[2][1]
            params_item = [np.float32, -1, [dim_c]]
            weight, _ = create_common_tensor(params_item, -1, 1)
            bias, _ = create_common_tensor(params_item, -1, 1)
            running_mean, _ = create_common_tensor(params_item, -1, 1)
            running_var, _ = create_common_tensor(params_item, -1, 1)

            # create BatchNorm2D
            cpu_bn3d = torch.nn.BatchNorm3d(dim_c)
            cpu_bn3d.weight.data = weight
            cpu_bn3d.bias.data = bias
            cpu_bn3d.running_mean.data = running_mean
            cpu_bn3d.running_var.data = running_var

            npu_bn3d = copy.deepcopy(cpu_bn3d).to("npu")

            # run exec
            cpu_output, cpu_running_mean, cpu_running_var, cpu_input_grad, cpu_weight_grad, cpu_bias_grad \
                = self.cpu_op_exec(cpu_input1, cpu_bn3d)

            npu_output, npu_running_mean, npu_running_var, npu_input_grad, npu_weight_grad, npu_bias_grad \
                = self.npu_op_exec(npu_input1, npu_bn3d)

            # comparison results
            self.assertRtolEqual(cpu_output, npu_output, 1e-3)
            self.assertRtolEqual(cpu_running_mean, npu_running_mean)
            self.assertRtolEqual(cpu_running_var, npu_running_var)
            self.assertRtolEqual(cpu_input_grad, npu_input_grad)
            self.assertRtolEqual(cpu_weight_grad, npu_weight_grad, 1e-2)
            self.assertRtolEqual(cpu_bias_grad, npu_bias_grad)


if __name__ == "__main__":
    run_tests()
