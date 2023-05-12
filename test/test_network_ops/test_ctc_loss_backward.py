import copy
import torch
import numpy as np
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests


class TestCtcLossBackward(TestCase):
    def generate_data(self, item):
        T = item[0][0]
        C = item[0][1]
        N = item[0][2]
        S = item[0][3]
        S_min = item[0][4]
        dtype = item[1]
        reduction_str = item[2]

        log_probs = np.random.uniform(-10, 10, (T, N, C)).astype(dtype)
        targets = torch.randint(1, C, (N, S), dtype=torch.long)
        input_lengths = torch.full((N,), T, dtype=torch.long)
        target_lengths = torch.randint(S_min, S, (N,), dtype=torch.long)

        # modify from numpy.ndarray to torch.tensor
        log_probs = torch.from_numpy(log_probs)

        ctc_loss = torch.nn.CTCLoss(zero_infinity=True, reduction=reduction_str)

        return ctc_loss, log_probs, targets, input_lengths, target_lengths

    def generate_data_1d(self, item):
        T = item[0][0]
        C = item[0][1]
        N = item[0][2]
        S = item[0][3]
        S_min = item[0][4]
        dtype = item[1]
        reduction_str = item[2]

        log_probs = np.random.uniform(-10, 10, (T, N, C)).astype(dtype)
        input_lengths = torch.full((N,), T, dtype=torch.long)
        target_lengths = torch.randint(S_min, S, (N,), dtype=torch.long)

        target_lengths_sum = torch.sum(target_lengths)
        target_length = target_lengths_sum.item()
        targets = torch.randint(1, C, (target_length,), dtype=torch.long)

        # modify from numpy.ndarray to torch.tensor
        log_probs = torch.from_numpy(log_probs)

        ctc_loss = torch.nn.CTCLoss(zero_infinity=True, reduction=reduction_str)
        return ctc_loss, log_probs, targets, input_lengths, target_lengths

    def cpu_op_exec(self, ctc_loss, log_probs, targets, input_lengths, target_lengths):
        if log_probs.dtype == torch.float16:
            log_probs = log_probs.to(torch.float32)

        log_probs.requires_grad_(True)
        log_probs.retain_grad()

        neg_log_likelihood = ctc_loss(log_probs.log_softmax(2), targets, input_lengths, target_lengths)
        neg_log_likelihood.backward()
        grad = log_probs.grad

        grad = grad.numpy()

        return grad

    def npu_op_exec(self, ctc_loss, log_probs, targets, input_lengths, target_lengths):
        log_probs = copy.deepcopy(log_probs).npu()
        targets = targets.npu()
        log_probs.requires_grad_(True)
        log_probs.retain_grad()

        neg_log_likelihood = ctc_loss(log_probs.log_softmax(2), targets, input_lengths.npu(), target_lengths.npu())
        neg_log_likelihood.backward()
        grad = log_probs.grad

        if grad.dtype == torch.float16:
            grad = grad.to(torch.float32)

        grad = grad.cpu().numpy()

        return grad

    def test_ctc_loss_backward(self):
        sizes_list = [[50, 20, 16, 30, 10], [26, 37, 2560, 18, 10]]
        para_reduction = ["sum", "mean"]
        dtype = [np.float32]  # Insufficient accuracy when use fp16 data
        shape_format = [
            [i, j, k] for i in sizes_list for j in dtype for k in para_reduction
        ]

        for item in shape_format:
            ctc_loss, log_probs, targets, input_lengths, target_lengths = self.generate_data(item)

            grad_cpu = self.cpu_op_exec(ctc_loss, log_probs, targets, input_lengths, target_lengths)
            grad_npu = self.npu_op_exec(ctc_loss, log_probs, targets, input_lengths, target_lengths)

            self.assertRtolEqual(grad_cpu, grad_npu, 1e-3)

    def test_ctc_loss_backward_1d(self):
        sizes_list = [[50, 20, 16, 30, 10], [26, 37, 2560, 18, 10]]
        para_reduction = ["sum", "mean"]
        dtype = [np.float32]  # Insufficient accuracy when use fp16 data
        shape_format = [
            [i, j, k] for i in sizes_list for j in dtype for k in para_reduction
        ]

        for item in shape_format:
            ctc_loss, log_probs, targets, input_lengths, target_lengths = self.generate_data_1d(item)

            grad_cpu = self.cpu_op_exec(ctc_loss, log_probs, targets, input_lengths, target_lengths)
            grad_npu = self.npu_op_exec(ctc_loss, log_probs, targets, input_lengths, target_lengths)

            self.assertRtolEqual(grad_cpu, grad_npu, 1e-3)


if __name__ == "__main__":
    run_tests()
