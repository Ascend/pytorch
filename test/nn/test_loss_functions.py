import copy
import unittest

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests


class TestLossFunctions(TestCase):
    def test_L1Loss(self):
        loss = nn.L1Loss().npu()
        input1 = torch.randn(3, 5).npu()
        input1.requires_grad_(True)
        target = torch.randn(3, 5).npu()
        output = loss(input1, target)
        output.backward()
        self.assertEqual(input1.grad is not None, True)

    def test_MSELoss(self):
        loss = nn.MSELoss().npu()
        input1 = torch.randn(3, 5).npu()
        input1.requires_grad_(True)
        target = torch.randn(3, 5).npu()
        output = loss(input1, target)
        output.backward()
        self.assertEqual(input1.grad is not None, True)

    def test_CrossEntropyLoss(self):
        loss = nn.CrossEntropyLoss().npu()
        input1 = torch.randn(3, 5).npu()
        input1.requires_grad_(True)
        target = torch.empty(3, dtype=torch.long).random_(5).npu()
        output = loss(input1, target)
        output.backward()
        self.assertEqual(input1.grad is not None, True)

    def test_CTCLoss(self):
        T = 50      # Input sequence length
        C = 20      # Number of classes (including blank)
        N = 16      # Batch size
        S = 30      # Target sequence length of longest target in batch (padding length)
        S_min = 10  # Minimum target length, for demonstration purposes

        # Initialize random batch of input vectors, for *size = (T,N,C)
        input1 = torch.randn(T, N, C).npu().log_softmax(2).detach()
        cinput = input1.cpu()
        input1.requires_grad_(True)
        cinput.requires_grad_(True)
        # Initialize random batch of targets (0 = blank, 1:C = classes)
        target = torch.randint(low=1, high=C, size=(N, S), dtype=torch.int32).npu()

        input_lengths = torch.full(size=(N,), fill_value=T, dtype=torch.int32).npu()
        target_lengths = torch.randint(low=S_min, high=S, size=(N,), dtype=torch.int32).npu()
        ctc_loss = nn.CTCLoss().npu()
        loss = ctc_loss(input1, target, input_lengths, target_lengths)
        loss.backward(torch.ones_like(loss))

        ctarget = target.cpu().long()
        cinput_lengths = input_lengths.cpu().long()
        ctarget_lengths = target_lengths.cpu().long()
        cctc_loss = ctc_loss.cpu()
        closs = cctc_loss(cinput, ctarget, cinput_lengths, ctarget_lengths)
        closs.backward(torch.ones_like(closs))
        self.assertRtolEqual(loss.detach().cpu().numpy(), closs.detach().numpy())
        self.assertRtolEqual(input1.grad.cpu().numpy(), cinput.grad.numpy())

    def test_NLLLoss(self):
        m = nn.LogSoftmax(dim=1)
        # input1 is of size N x C = 3 x 5
        x = torch.randn(3, 5)
        input1 = m(x)
        ninput = input1.npu()
        input1.requires_grad_(True)
        ninput.requires_grad_(True)
        # each element in target has to have 0 <= value < C
        target = torch.tensor([1, 0, 4])
        loss = nn.NLLLoss()
        output = loss(input1, target)
        output.backward(torch.ones_like(output))

        ntarget = target.npu()
        nloss = loss.npu()
        noutput = nloss(ninput, ntarget)
        noutput.backward(torch.ones_like(noutput))
        self.assertRtolEqual(output.detach().numpy(), noutput.detach().cpu().numpy())
        self.assertRtolEqual(input1.grad.numpy(), ninput.grad.cpu().numpy())

    def test_PoissonNLLLoss(self):
        loss = nn.PoissonNLLLoss().npu()
        log_input = torch.randn(5, 2).npu()
        log_input.requires_grad_(True)
        target = torch.randn(5, 2).npu()
        output = loss(log_input, target)
        output.backward()
        self.assertEqual(log_input.grad is not None, True)

    def test_KLDivLoss(self):
        input_shape = (2, 5)
        log_prob1 = F.log_softmax(torch.randn(input_shape), 1).npu()
        prob2 = F.softmax(torch.randn(input_shape), 1).npu()

        loss = nn.KLDivLoss(reduction='batchmean').npu()
        output = loss(log_prob1, prob2)

        loss_none_reduce = nn.KLDivLoss(reduction='sum').npu()(log_prob1, prob2)
        expected = loss_none_reduce / input_shape[0]

        self.assertEqual(output, expected)

    def test_BCELoss(self):
        m = nn.Sigmoid().npu()
        loss = nn.BCELoss().npu()
        input1 = torch.randn(3).npu()
        input1.requires_grad_(True)
        target = torch.empty(3).random_(2).npu()
        output = loss(m(input1), target)
        output.backward()
        self.assertEqual(input1.grad is not None, True)

    def test_BCEWithLogitsLoss(self):
        loss = nn.BCEWithLogitsLoss().npu()
        input1 = torch.randn(3).npu()
        input1.requires_grad_(True)
        target = torch.empty(3).random_(2).npu()
        output = loss(input1, target)
        output.backward()
        self.assertEqual(input1.grad is not None, True)

    def test_MarginRankingLoss(self):
        loss = nn.MarginRankingLoss().npu()
        input1 = torch.randn(3).npu()
        input2 = torch.randn(3).npu()
        input1.requires_grad_(True)
        input2.requires_grad_(True)
        target = torch.randn(3).sign().npu()
        output = loss(input1, input2, target)
        output.backward()
        self.assertEqual(input1.grad is not None, True)
        self.assertEqual(input2.grad is not None, True)

    def test_HingeEmbeddingLoss(self):

        torch.manual_seed(20)
        hinge_loss = nn.HingeEmbeddingLoss(margin=0.2).npu()
        a = torch.randn(100, 128).npu()
        b = torch.randn(100, 128).npu()
        a.requires_grad_(True)
        b.requires_grad_(True)
        x = 1 - torch.cosine_similarity(a, b)
        y = 2 * torch.empty(100).random_(2) - 1
        output = hinge_loss(x, y.npu())
        self.assertEqual(output.item() is not None, True)

    def test_MultiLabelMarginLoss(self):

        loss = nn.MultiLabelMarginLoss().npu()
        x = torch.FloatTensor([[0.1, 0.2, 0.4, 0.8]]).npu()
        # for target y, only consider labels 3 and 0, not after label -1
        y = torch.LongTensor([[3, 0, -1, 1]]).npu().int()
        output = loss(x, y)
        output.item()
        self.assertEqual(output is not None, True)

    def test_SmoothL1Loss(self):
        loss = nn.SmoothL1Loss().npu()
        input1 = torch.Tensor([1, 5, 3, 0.5, 0.9]).npu()
        targt = torch.Tensor([4, 1, 0, 0.4, 0.2]).npu()
        output = loss(input1, targt)
        self.assertEqual(output is not None, True)

    def test_SoftMarginLoss(self):
        loss = nn.SoftMarginLoss().npu()
        input1 = torch.Tensor([1, 5, 3, 0.5, 0.9]).npu()
        targt = torch.Tensor([4, 1, 0, 0.4, 0.2]).npu()
        output = loss(input1, targt)
        self.assertEqual(output is not None, True)

    def test_CosineEmbeddingLoss(self):
        loss = torch.nn.CosineEmbeddingLoss(reduction="mean").npu()
        input1 = torch.randn(3, 5).npu()
        input2 = torch.randn(3, 5).npu()
        targt = torch.randint(5, (3,)).npu().int()
        output = loss(input1, input2, targt)
        self.assertEqual(output is not None, True)

    def test_TripletMarginLoss(self):
        triplet_loss = nn.TripletMarginLoss(margin=1.0, p=2).npu()
        anchor = torch.randn(100, 128).npu()
        positive = torch.randn(100, 128).npu()
        negative = torch.randn(100, 128).npu()
        anchor.requires_grad_(True)
        positive.requires_grad_(True)
        negative.requires_grad_(True)
        output = triplet_loss(anchor, positive, negative)
        output.backward()
        self.assertEqual(anchor.grad is not None, True)

    def test_binary_cross_entropy(self):
        input1 = torch.rand(2, 3)
        target = torch.rand(2, 3)
        npu_input = copy.deepcopy(input1).npu()
        npu_target = copy.deepcopy(target).npu()

        cpu_output = F.binary_cross_entropy(input1, target)
        npu_output = F.binary_cross_entropy(npu_input, npu_target)

        self.assertRtolEqual(cpu_output.detach().numpy(), npu_output.detach().cpu().numpy())

    def test_binary_cross_entropy_with_logits(self):
        input1 = torch.rand(2, 3)
        target = torch.rand(2, 3)
        npu_input = copy.deepcopy(input1).npu()
        npu_target = copy.deepcopy(target).npu()

        cpu_output = F.binary_cross_entropy_with_logits(input1, target)
        npu_output = F.binary_cross_entropy_with_logits(npu_input, npu_target)

        self.assertRtolEqual(cpu_output.detach().numpy(), npu_output.detach().cpu().numpy())

    def test_poisson_nll_loss(self):
        input1 = torch.rand(2, 3)
        target = torch.rand(2, 3)
        npu_input = copy.deepcopy(input1).npu()
        npu_target = copy.deepcopy(target).npu()

        cpu_output = F.poisson_nll_loss(input1, target)
        npu_output = F.poisson_nll_loss(npu_input, npu_target)

        self.assertRtolEqual(cpu_output.detach().numpy(), npu_output.detach().cpu().numpy())

    def test_cosine_embedding_loss(self):
        input1 = torch.randn(2, 3)
        input2 = torch.randn(2, 3)
        npu_input1 = copy.deepcopy(input1).npu()
        npu_input2 = copy.deepcopy(input2).npu()
        target = torch.rand(2)
        npu_target = copy.deepcopy(target).npu()

        cpu_output = F.cosine_embedding_loss(input1, input2, target)
        npu_output = F.cosine_embedding_loss(npu_input1, npu_input2, npu_target)

        self.assertRtolEqual(cpu_output.detach().numpy(), npu_output.detach().cpu().numpy())

    def test_ctc_loss(self):
        log_probs = torch.randn(50, 16, 20).npu().log_softmax(2)
        targets = torch.randint(1, 20, (16, 30), dtype=torch.int32).npu()
        input_lengths = torch.full((16,), 50, dtype=torch.int32).npu()
        target_lengths = torch.randint(10, 30, (16,), dtype=torch.int32).npu()

        cpu_log_probs = copy.deepcopy(log_probs).cpu()
        cpu_targets = copy.deepcopy(targets).cpu().long()
        cpu_input_lengths = copy.deepcopy(input_lengths).cpu().long()
        cpu_target_lengths = copy.deepcopy(target_lengths).cpu().long()

        npu_output = F.ctc_loss(log_probs, targets, input_lengths, target_lengths)
        cpu_output = F.ctc_loss(cpu_log_probs, cpu_targets, cpu_input_lengths, cpu_target_lengths)

        self.assertRtolEqual(cpu_output.numpy(), npu_output.cpu().numpy())

    def test_hinge_embedding_loss(self):
        input1 = torch.randn(5, 3)
        targets = torch.randint(1, 20, (5, 3), dtype=torch.long)

        npu_input = copy.deepcopy(input1).npu()
        npu_targets = copy.deepcopy(targets).npu().int()

        cpu_output = F.hinge_embedding_loss(input1, targets)
        npu_output = F.hinge_embedding_loss(npu_input, npu_targets)

        self.assertRtolEqual(cpu_output.detach().numpy(), npu_output.detach().cpu().numpy())
    
    @unittest.skip("skip test_kl_div now")
    def test_kl_div(self):
        input1 = torch.randn(5, 3)
        targets = torch.randn(5, 3)

        npu_input = copy.deepcopy(input1).npu()
        npu_targets = copy.deepcopy(targets).npu()

        cpu_output = F.kl_div(input1, targets)
        npu_output = F.kl_div(npu_input, npu_targets)

        self.assertRtolEqual(cpu_output.detach().numpy(), npu_output.detach().cpu().numpy())

    def test_l1_loss(self):
        input1 = torch.randn(5, 3)
        targets = torch.randn(5, 3)

        npu_input = copy.deepcopy(input1).npu()
        npu_targets = copy.deepcopy(targets).npu()

        cpu_output = F.l1_loss(input1, targets)
        npu_output = F.l1_loss(npu_input, npu_targets)

        self.assertRtolEqual(cpu_output.detach().numpy(), npu_output.detach().cpu().numpy())

    def test_mse_loss(self):
        input1 = torch.randn(5, 3)
        targets = torch.randn(5, 3)

        npu_input = copy.deepcopy(input1).npu()
        npu_targets = copy.deepcopy(targets).npu()

        cpu_output = F.mse_loss(input1, targets)
        npu_output = F.mse_loss(npu_input, npu_targets)

        self.assertRtolEqual(cpu_output.detach().numpy(), npu_output.detach().cpu().numpy())

    def test_margin_ranking_loss(self):
        input1 = torch.randn(3)
        input2 = torch.randn(3)
        targets = torch.randn(3)

        npu_input1 = copy.deepcopy(input1).npu()
        npu_input2 = copy.deepcopy(input2).npu()
        npu_targets = copy.deepcopy(targets).npu()

        cpu_output = F.margin_ranking_loss(input1, input2, targets)
        npu_output = F.margin_ranking_loss(npu_input1, npu_input2, npu_targets)

        self.assertRtolEqual(cpu_output.detach().numpy(), npu_output.detach().cpu().numpy())

    def test_multilabel_margin_loss(self):
        input1 = torch.Tensor([[0.1, 0.2, 0.4, 0.8], [0.1, 0.2, 0.4, 0.8]]).to(torch.float32)
        targets = torch.Tensor([[3, 0, -1, 1], [0, 1, 3, -1]]).to(torch.int64)

        npu_input = copy.deepcopy(input1).npu()
        npu_targets = copy.deepcopy(targets).npu().int()

        cpu_output = F.multilabel_margin_loss(input1, targets)
        npu_output = F.multilabel_margin_loss(npu_input, npu_targets)

        self.assertRtolEqual(cpu_output.detach().numpy(), npu_output.detach().cpu().numpy())

    def test_nll_loss(self):
        input1 = torch.Tensor([[0.1, 0.2, 0.4, 0.8], [0.1, 0.2, 0.4, 0.8]]).to(torch.float32)
        targets = torch.Tensor([3, 2]).to(torch.int64)

        npu_input = copy.deepcopy(input1).npu()
        npu_targets = copy.deepcopy(targets).npu().int()

        cpu_output = F.nll_loss(input1, targets)
        npu_output = F.nll_loss(npu_input, npu_targets)

        self.assertRtolEqual(cpu_output.detach().numpy(), npu_output.detach().cpu().numpy())

    def test_smooth_l1_loss(self):
        input1 = torch.Tensor([[0.1, 0.2, 0.4, 0.8], [0.1, 0.2, 0.4, 0.8]]).to(torch.float32)
        targets = torch.Tensor([[3, 0, -1, 1], [0, 1, 3, -1]]).to(torch.float32)

        npu_input = copy.deepcopy(input1).npu()
        npu_targets = copy.deepcopy(targets).npu()

        cpu_output = F.smooth_l1_loss(input1, targets)
        npu_output = F.smooth_l1_loss(npu_input, npu_targets)

        self.assertRtolEqual(cpu_output.detach().numpy(), npu_output.detach().cpu().numpy())

    def test_soft_margin_loss(self):
        input1 = torch.Tensor([[0.1, 0.2, 0.4, 0.8], [0.1, 0.2, 0.4, 0.8]]).to(torch.float32)
        targets = torch.Tensor([[3, 0, -1, 1], [0, 1, 3, -1]]).to(torch.float32)

        npu_input = copy.deepcopy(input1).npu()
        npu_targets = copy.deepcopy(targets).npu()

        cpu_output = F.soft_margin_loss(input1, targets)
        npu_output = F.soft_margin_loss(npu_input, npu_targets)

        self.assertRtolEqual(cpu_output.detach().numpy(), npu_output.detach().cpu().numpy())

    def test_triplet_margin_loss(self):
        input1 = torch.randn(5, 10)
        input2 = torch.randn(5, 10)
        input3 = torch.randn(5, 10)

        npu_input1 = copy.deepcopy(input1).npu()
        npu_input2 = copy.deepcopy(input2).npu()
        npu_input3 = copy.deepcopy(input3).npu()

        cpu_output = F.triplet_margin_loss(input1, input2, input3)
        npu_output = F.triplet_margin_loss(npu_input1, npu_input2, npu_input3)

        self.assertRtolEqual(cpu_output.detach().numpy(), npu_output.detach().cpu().numpy())


if __name__ == "__main__":
    run_tests()
