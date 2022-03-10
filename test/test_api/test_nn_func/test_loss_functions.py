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
import copy

import torch
import torch_npu
import torch.nn.functional as F


from torch_npu.testing.testcase import TestCase, run_tests


class TestLossFunctions(TestCase):
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
        log_probs = torch.randn(50, 16, 20).log_softmax(2).detach().requires_grad_()
        targets = torch.randint(1, 20, (16, 30), dtype=torch.long)
        input_lengths = torch.full((16,), 50, dtype=torch.long)
        target_lengths = torch.randint(10,30,(16,), dtype=torch.long)

        npu_log_probs = copy.deepcopy(log_probs).npu()
        npu_targets = copy.deepcopy(targets).npu().int()
        npu_input_lengths = copy.deepcopy(input_lengths).npu().int()
        npu_target_lengths = copy.deepcopy(target_lengths).npu().int()
        
        cpu_output = F.ctc_loss(log_probs, targets, input_lengths, target_lengths)
        npu_output = F.ctc_loss(npu_log_probs, npu_targets, npu_input_lengths, npu_target_lengths)

        self.assertRtolEqual(cpu_output.detach().numpy(), npu_output.detach().cpu().numpy())

    def test_hinge_embedding_loss(self):
        input1 = torch.randn(5, 3)
        targets = torch.randint(1, 20, (5, 3), dtype=torch.long)

        npu_input = copy.deepcopy(input1).npu()
        npu_targets = copy.deepcopy(targets).npu().int()
        
        cpu_output = F.hinge_embedding_loss(input1, targets)
        npu_output = F.hinge_embedding_loss(npu_input, npu_targets)

        self.assertRtolEqual(cpu_output.detach().numpy(), npu_output.detach().cpu().numpy())

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
    torch.npu.set_device(0)
    run_tests()