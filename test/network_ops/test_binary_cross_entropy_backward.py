import copy
import torch
import numpy as np
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests


class TestBinaryCrossEntropyBackward(TestCase):

    def generate_data(self, min_val, max_val, shape, dtype):
        np.random.seed(1234)
        x = np.random.uniform(min_val, max_val, shape).astype(dtype)
        x = torch.from_numpy(x)
        return x

    def cpu_op_exec(self, input1, target, weight, reduction="mean"):
        float16flag = False
        if input1.dtype == torch.float16:
            input1 = input1.to(torch.float32)
            target = target.to(torch.float32)
            float16flag = True
            if weight is not None:
                weight = weight.to(torch.float32)
        input1.requires_grad_(True)
        cpu_output = torch.nn.functional.binary_cross_entropy(input1, target,
                                                              weight=weight, size_average=None,
                                                              reduce=None, reduction=reduction)
        input_cpu = cpu_output.detach().numpy()
        if reduction == 'none':
            w = torch.ones_like(input1)
            cpu_output.backward(w)
        else:
            cpu_output.backward()
        res = input1.grad
        res = res.numpy()
        if float16flag:
            input_cpu = input_cpu.astype(np.float16)
            res = res.astype(np.float16)
        return input_cpu, res

    def npu_op_exec(self, input1, target, weight, reduction="mean"):
        input1 = input1.npu()
        target = target.npu()
        if weight is not None:
            weight = weight.npu()
        input1.requires_grad_(True)
        npu_output = torch.nn.functional.binary_cross_entropy(input1, target,
                                                              weight=weight, size_average=None,
                                                              reduce=None, reduction=reduction)
        npu_input = npu_output.cpu()
        npu_input = npu_input.detach().numpy()
        if reduction == 'none':
            w = torch.ones_like(input1)
            npu_output.backward(w)
        else:
            npu_output.backward()
        res = input1.grad.cpu()
        res = res.numpy()
        return npu_input, res

    def test_binary_cross_entropy_backward_float16(self):
        shape_list = [(10, 64)]
        reduction_list = ["none", "mean", "sum"]
        shape_format = [
            [np.float16, i, j] for i in shape_list for j in reduction_list
        ]
        for item in shape_format:
            input1 = self.generate_data(0, 1, item[1], item[0])
            target = self.generate_data(0, 1, item[1], item[0])
            cpu_input1 = copy.deepcopy(input1)
            cpu_target = copy.deepcopy(target)
            weight = None
            cpu_output, cpu_grad = self.cpu_op_exec(cpu_input1, cpu_target, weight, reduction=item[2])
            npu_output, npu_grad = self.npu_op_exec(input1, target, weight, reduction=item[2])
            self.assertRtolEqual(cpu_output, npu_output)
            self.assertRtolEqual(cpu_grad, npu_grad)

    def test_binary_cross_entropy_backward_float32(self):
        shape_list = [(10, 64)]
        reduction_list = ["none", "mean", "sum"]
        shape_format = [
            [np.float32, i, j] for i in shape_list for j in reduction_list
        ]
        for item in shape_format:
            input1 = self.generate_data(0, 1, item[1], item[0])
            target = self.generate_data(0, 1, item[1], item[0]).int().to(torch.float32)
            cpu_input1 = copy.deepcopy(input1)
            cpu_target = copy.deepcopy(target)
            weight = None
            cpu_output, cpu_grad = self.cpu_op_exec(cpu_input1, cpu_target, weight, reduction=item[2])
            npu_output, npu_grad = self.npu_op_exec(input1, target, weight, reduction=item[2])
            self.assertRtolEqual(cpu_output, npu_output)
            self.assertRtolEqual(cpu_grad, npu_grad)

    def test_binary_cross_entropy_backward_with_weight_float16(self):
        shape_list = [(10, 64)]
        reduction_list = ["none", "mean", "sum"]
        shape_format = [
            [np.float16, i, j] for i in shape_list for j in reduction_list
        ]
        for item in shape_format:
            input1 = self.generate_data(0, 1, item[1], item[0])
            target = self.generate_data(0, 1, item[1], item[0])
            weight = self.generate_data(0, 1, item[1], item[0])
            cpu_input1 = copy.deepcopy(input1)
            cpu_target = copy.deepcopy(target)
            cpu_weight = copy.deepcopy(weight)
            cpu_output, cpu_grad = self.cpu_op_exec(cpu_input1, cpu_target, cpu_weight, reduction=item[2])
            npu_output, npu_grad = self.npu_op_exec(input1, target, weight, reduction=item[2])
            self.assertRtolEqual(cpu_output, npu_output)
            self.assertRtolEqual(cpu_grad, npu_grad)

    def test_binary_cross_entropy_backward_with_weight_float32(self):
        shape_list = [(10, 64)]
        reduction_list = ["none", "mean", "sum"]
        shape_format = [
            [np.float32, i, j] for i in shape_list for j in reduction_list
        ]
        for item in shape_format:
            input1 = self.generate_data(0, 1, item[1], item[0])
            target = self.generate_data(0, 1, item[1], item[0])
            weight = self.generate_data(0, 1, item[1], item[0])
            cpu_input1 = copy.deepcopy(input1)
            cpu_target = copy.deepcopy(target)
            cpu_weight = copy.deepcopy(weight)
            cpu_output, cpu_grad = self.cpu_op_exec(cpu_input1, cpu_target, cpu_weight, reduction=item[2])
            npu_output, npu_grad = self.npu_op_exec(input1, target, weight, reduction=item[2])
            self.assertRtolEqual(cpu_output, npu_output)
            self.assertRtolEqual(cpu_grad, npu_grad)


if __name__ == "__main__":
    run_tests()
