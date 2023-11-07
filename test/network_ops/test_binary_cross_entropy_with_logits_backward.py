import copy
import torch
import torch.nn as nn
import numpy as np
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests


def generate_data(min1, max1, shape, dtype):
    input1 = np.random.uniform(min1, max1, shape).astype(dtype)
    # modify from numpy.ndarray to torch.tensor
    output = torch.from_numpy(input1)
    # generate target: target.size == input1.size
    label = torch.randint(shape[1], size=(shape[0],), dtype=torch.long)
    target = torch.zeros(shape[0], shape[1])
    target[range(target.shape[0]), label] = 1
    target = target.to(output.dtype)
    return output, target


class TestBinaryCrossEntropyWithLogitsBackward(TestCase):
    def generate_one_input(self, lower, upper, shape, dtype):
        x = np.random.uniform(lower, upper, shape).astype(dtype)
        npu_input = torch.from_numpy(x)
        return npu_input

    def cpu_op_exec(self, input1, target):
        input1.requires_grad_(True)
        output = torch.nn.functional.binary_cross_entropy_with_logits(input1, target)
        input_cpu = output.detach().numpy()
        output.backward()
        res = input1.grad
        res = res.numpy()
        return input_cpu, res

    def npu_op_exec(self, input1, target):
        target = target.to("npu")
        input1 = input1.to("npu")
        input1.requires_grad_(True)
        output = torch.nn.functional.binary_cross_entropy_with_logits(input1, target)
        input_npu = output.cpu()
        input_npu = input_npu.detach().numpy()
        output.backward()
        res = input1.grad.cpu()
        res = res.numpy()
        return input_npu, res

    def cpu_op_exec_module(self, input1, target, weight=None, pos_weight=None, reduction="mean"):
        input1.requires_grad_(True)
        criterion = torch.nn.BCEWithLogitsLoss(weight=weight, pos_weight=pos_weight,
                                               reduction=reduction)
        res = criterion(input1, target)
        res.sum().backward()
        return res.detach().numpy(), input1.grad.numpy()

    def npu_op_exec_module(self, input1, target, weight=None, pos_weight=None, reduction="mean"):
        input1 = input1.to("npu")
        target = target.to("npu")
        input1.requires_grad_(True)
        if weight is not None:
            weight = weight.to("npu")
        if pos_weight is not None:
            pos_weight = pos_weight.to("npu")

        criterion = torch.nn.BCEWithLogitsLoss(weight=weight, pos_weight=pos_weight,
                                               reduction=reduction)
        criterion = criterion.to("npu")
        res = criterion(input1, target)
        res.sum().backward()
        res = res.to("cpu")
        return res.detach().numpy(), input1.grad.cpu().numpy()

    def test_binary_cross_entropy_with_logits_backward_fp32(self):
        npu_input1, npu_target = generate_data(0, 100, (5, 3), np.float32)
        cpu_input1 = copy.deepcopy(npu_input1)
        cpu_target = copy.deepcopy(npu_target)
        cpu_output, cpu_grad_output = self.cpu_op_exec(cpu_input1, cpu_target)
        npu_output, npu_grad_output = self.npu_op_exec(npu_input1, npu_target)
        self.assertRtolEqual(cpu_output, npu_output)
        self.assertRtolEqual(cpu_grad_output, npu_grad_output)

    def test_binary_cross_entropy_with_logits_backward_fp16(self):
        npu_input1, npu_target = generate_data(0, 100, (5, 3), np.float16)
        cpu_input1 = copy.deepcopy(npu_input1)
        cpu_target = copy.deepcopy(npu_target)
        cpu_input1 = cpu_input1.to(torch.float32)
        cpu_target = cpu_target.to(torch.float32)
        cpu_output, cpu_grad_output = self.cpu_op_exec(cpu_input1, cpu_target)
        npu_output, npu_grad_output = self.npu_op_exec(npu_input1, npu_target)
        cpu_output = cpu_output.astype(npu_output.dtype)
        cpu_grad_output = cpu_grad_output.astype(npu_grad_output.dtype)
        self.assertRtolEqual(cpu_output, npu_output)
        self.assertRtolEqual(cpu_grad_output, npu_grad_output)

    def test_binary_cross_with_logits_backward_module_float32(self):
        for shape, weight_shape, pos_weight_shape, reduction in [
            ((10, 64), None, None, "mean"),
            ((10, 64), (10, 1), None, "mean"),
            ((10, 64), None, (64,), "mean"),
            ((10, 64), None, None, "none"),
            ((10, 64), (10, 1), None, "none"),
            ((10, 64), None, (64,), "none"),
            ((10, 64), None, None, "sum"),
            ((10, 64), (10, 1), None, "sum"),
            ((10, 64), None, (64,), "sum"),
            ((10, 64), (10, 64), (10, 64), "sum"),
            ((10, 64), (10, 64), (10, 64), "mean"),
            ((10, 64), (10, 64), (10, 64), "none")
        ]:
            input1 = self.generate_one_input(0, 10, shape, np.float32)
            target = torch.empty(shape, dtype=torch.float32).random_(2)
            weight = None
            pos_weight = None

            if weight_shape is not None:
                weight = self.generate_one_input(0, 10, weight_shape, np.float32)
            if pos_weight_shape is not None:
                pos_weight = self.generate_one_input(0, 10, pos_weight_shape, np.float32)

            npu_output, npu_grad = self.npu_op_exec_module(input1, target,
                                                           weight=weight,
                                                           pos_weight=pos_weight,
                                                           reduction=reduction)
            cpu_output, cpu_grad = self.cpu_op_exec_module(input1, target,
                                                           weight=weight,
                                                           pos_weight=pos_weight,
                                                           reduction=reduction)
            self.assertRtolEqual(cpu_output, npu_output)
            self.assertRtolEqual(cpu_grad, npu_grad)

    def test_binary_cross_with_logits_backward_module_float16(self):
        for shape, weight_shape, pos_weight_shape, reduction in [
            ((10, 64), None, None, "mean"),
            ((10, 64), (10, 1), None, "mean"),
            ((10, 64), None, (64,), "mean"),
            ((10, 64), None, None, "none"),
            ((10, 64), (10, 1), None, "none"),
            ((10, 64), None, (64,), "none"),
            ((10, 64), None, None, "sum"),
            ((10, 64), (10, 1), None, "sum"),
            ((10, 64), None, (64,), "sum"),
            ((10, 64), (10, 64), (10, 64), "sum"),
            ((10, 64), (10, 64), (10, 64), "mean"),
            ((10, 64), (10, 64), (10, 64), "none")
        ]:
            input1 = self.generate_one_input(0, 10, shape, np.float16)
            target = torch.empty(shape, dtype=torch.float16).random_(2)
            input_32 = input1.type(torch.float32)
            target_32 = target.type(torch.float32)
            weight = None
            weight_32 = None
            pos_weight = None
            pos_weight_32 = None

            if weight_shape is not None:
                weight = self.generate_one_input(0, 10, weight_shape, np.float16)
                weight_32 = weight.type(torch.float32)
            if pos_weight_shape is not None:
                pos_weight = self.generate_one_input(0, 10, pos_weight_shape, np.float16)
                pos_weight_32 = pos_weight.type(torch.float32)

            npu_output, npu_grad = self.npu_op_exec_module(input1, target,
                                                           weight=weight,
                                                           pos_weight=pos_weight,
                                                           reduction=reduction)
            cpu_output, cpu_grad = self.cpu_op_exec_module(input_32, target_32,
                                                           weight=weight_32,
                                                           pos_weight=pos_weight_32,
                                                           reduction=reduction)
            cpu_output = cpu_output.astype(np.float16)
            cpu_grad = cpu_grad.astype(np.float16)
            self.assertRtolEqual(cpu_output, npu_output)
            self.assertRtolEqual(cpu_grad, npu_grad)


if __name__ == "__main__":
    run_tests()
