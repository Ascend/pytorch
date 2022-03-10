# Copyright (c) 2020, Huawei Technologies.All rights reserved.
#
# Licensed under the BSD 3-Clause License  (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://opensource.org/licenses/BSD-3-Clause
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import torch
import torch_npu
import torch.nn as nn
import torch.nn.functional as F

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
        input1 = torch.randn(T, N, C).log_softmax(2).detach()
        ninput = input1.npu()
        input1.requires_grad_(True)
        ninput.requires_grad_(True)
         # Initialize random batch of targets (0 = blank, 1:C = classes)
        target = torch.randint(low=1, high=C, size=(N, S), dtype=torch.long)

        input_lengths = torch.full(size=(N,), fill_value=T, dtype=torch.long)
        target_lengths = torch.randint(low=S_min, high=S, size=(N,), dtype=torch.long)
        ctc_loss = nn.CTCLoss()
        loss = ctc_loss(input1, target, input_lengths, target_lengths)
        loss.backward(torch.ones_like(loss))

        ntarget = target.npu()
        ninput_lengths = input_lengths.npu()
        ntarget_lengths = target_lengths.npu()
        nctc_loss = ctc_loss.npu()
        nloss = nctc_loss(ninput, ntarget, ninput_lengths, ntarget_lengths)
        nloss.backward(torch.ones_like(nloss))
        self.assertRtolEqual(loss.detach().numpy(), nloss.detach().cpu().numpy())
        self.assertRtolEqual(input1.grad.numpy(), ninput.grad.cpu().numpy())

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
        l = loss(log_prob1, prob2)

        loss_none_reduce = nn.KLDivLoss(reduction='sum').npu()(log_prob1, prob2)
        expected = loss_none_reduce / input_shape[0]

        self.assertEqual(l, expected)

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
        input1 = torch.Tensor([1,5,3,0.5,0.9]).npu()
        targt = torch.Tensor([4,1,0,0.4,0.2]).npu()
        output = loss(input1, targt)
        self.assertEqual(output is not None, True)

    def test_SoftMarginLoss(self):
        loss = nn.SoftMarginLoss().npu()
        input1 = torch.Tensor([1,5,3,0.5,0.9]).npu()
        targt = torch.Tensor([4,1,0,0.4,0.2]).npu()
        output = loss(input1, targt)
        self.assertEqual(output is not None, True)

    def test_CosineEmbeddingLoss(self):
        loss = torch.nn.CosineEmbeddingLoss(reduction = "mean").npu()
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


if __name__ == "__main__":
    torch.npu.set_device(0)
    run_tests()