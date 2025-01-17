import torch
import numpy as np
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests


class TestBinaryCrossEntropyWithLogits(TestCase):

    def generate_two_input(self, lower, upper, shape, dtype):
        x = np.random.uniform(lower, upper, shape).astype(dtype)
        y = np.random.uniform(lower, upper, shape).astype(dtype)

        npu_input = torch.from_numpy(x)
        target_input = torch.from_numpy(y)

        return npu_input, target_input

    def generate_one_input(self, lower, upper, shape, dtype):
        x = np.random.uniform(lower, upper, shape).astype(dtype)
        npu_input = torch.from_numpy(x)
        return npu_input

    def cpu_op_exec(self, input1, target, weight=None, pos_weight=None, reduction="mean"):
        criterion = torch.nn.BCEWithLogitsLoss(weight=weight, pos_weight=pos_weight,
                                               reduction=reduction)
        res = criterion(input1, target)
        return res.numpy()

    def npu_op_exec(self, input1, target, weight=None, pos_weight=None, reduction="mean"):
        input1 = input1.to("npu")
        target = target.to("npu")
        if weight is not None:
            weight = weight.to("npu")
        if pos_weight is not None:
            pos_weight = pos_weight.to("npu")

        criterion = torch.nn.BCEWithLogitsLoss(weight=weight, pos_weight=pos_weight,
                                               reduction=reduction)
        criterion = criterion.to("npu")
        res = criterion(input1, target)
        res = res.to("cpu")
        return res.numpy()

    def cpu_op_func_exec(self, input1, target, weight=None, pos_weight=None, reduction="mean"):
        res = torch.nn.functional.binary_cross_entropy_with_logits(input1, target, weight=weight, pos_weight=pos_weight,
                                                                   reduction=reduction)
        return res.numpy()

    def npu_op_func_exec(self, input1, target, weight=None, pos_weight=None, reduction="mean"):
        input1 = input1.to("npu")
        target = target.to("npu")
        if weight is not None:
            weight = weight.to("npu")
        if pos_weight is not None:
            pos_weight = pos_weight.to("npu")

        res = torch.nn.functional.binary_cross_entropy_with_logits(input1, target, weight=weight, pos_weight=pos_weight,
                                                                   reduction=reduction)
        res = res.to("cpu")
        return res.numpy()

    def test_binary_cross_with_logits_float32(self, device="npu"):
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
            ((10, 64), (10, 64), (10, 64), "mean"),
            ((10, 64), (10, 64), (10, 64), "sum"),
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
            cpu_output = self.cpu_op_exec(input1, target, weight=weight, pos_weight=pos_weight, reduction=reduction)
            npu_output = self.npu_op_exec(input1, target, weight=weight, pos_weight=pos_weight, reduction=reduction)
            self.assertRtolEqual(cpu_output, npu_output)

    def test_binary_cross_with_logits_float16(self, device="npu"):
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

            npu_output = self.npu_op_exec(input1, target, weight=weight, pos_weight=pos_weight, reduction=reduction)
            cpu_output = self.cpu_op_exec(input_32, target_32, weight=weight_32, pos_weight=pos_weight_32,
                                          reduction=reduction)
            cpu_output = cpu_output.astype(np.float16)
            self.assertRtolEqual(cpu_output, npu_output, 1.e-3)

    def test_binary_cross_with_logits_function_float32(self, device="npu"):
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
            ((10, 64), (10, 64), (10, 64), "mean"),
            ((10, 64), (10, 64), (10, 64), "sum"),
            ((10, 64), (10, 64), (10, 64), "none")
        ]:
            input1 = self.generate_one_input(0, 2, shape, np.float32)
            target = torch.empty(shape, dtype=torch.float32).random_(2)
            weight = None
            pos_weight = None
            if weight_shape is not None:
                weight = self.generate_one_input(0, 2, weight_shape, np.float32)
            if pos_weight_shape is not None:
                pos_weight = self.generate_one_input(0, 2, pos_weight_shape, np.float32)
            cpu_output = self.cpu_op_func_exec(input1, target, weight=weight,
                                               pos_weight=pos_weight, reduction=reduction)
            npu_output = self.npu_op_func_exec(input1, target, weight=weight,
                                               pos_weight=pos_weight, reduction=reduction)
            self.assertRtolEqual(cpu_output, npu_output)

    def test_binary_cross_with_logits_function_float16(self, device="npu"):
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
            input1 = self.generate_one_input(0, 2, shape, np.float16)
            target = torch.empty(shape, dtype=torch.float16).random_(2)
            input_32 = input1.type(torch.float32)
            target_32 = target.type(torch.float32)
            weight = None
            weight_32 = None
            pos_weight = None
            pos_weight_32 = None

            if weight_shape is not None:
                weight = self.generate_one_input(0, 2, weight_shape, np.float16)
                weight_32 = weight.type(torch.float32)
            if pos_weight_shape is not None:
                pos_weight = self.generate_one_input(0, 2, pos_weight_shape, np.float16)
                pos_weight_32 = pos_weight.type(torch.float32)

            npu_output = self.npu_op_func_exec(input1, target, weight=weight,
                                               pos_weight=pos_weight, reduction=reduction)
            cpu_output = self.cpu_op_func_exec(input_32, target_32, weight=weight_32,
                                               pos_weight=pos_weight_32, reduction=reduction)

            cpu_output = cpu_output.astype(np.float16)
            self.assertRtolEqual(cpu_output, npu_output)


if __name__ == "__main__":
    run_tests()
