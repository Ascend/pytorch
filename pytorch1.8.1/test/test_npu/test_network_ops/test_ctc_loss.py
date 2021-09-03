# Copyright (c) 2020 Huawei Technologies Co., Ltd
# Copyright (c) 2019, Facebook CORPORATION.
# All rights reserved.
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
import numpy as np
import sys
import copy
from common_utils import TestCase, run_tests
from common_device_type import dtypes, instantiate_device_type_tests
from util_test import create_common_tensor


class TestCtcLoss(TestCase):
    def generate_data(self, item):
        T = item[0][0]
        C = item[0][1]
        N = item[0][2]
        S = item[0][3]
        S_min = item[0][4]
        dtype = item[1]
        reduction_str = item[2] 

        log_probs = np.random.uniform(-10, 10, (T, N, C)).astype(dtype)
        targets = torch.randint(1, C, (N, S), dtype = torch.long)
        input_lengths = torch.full((N,), T, dtype=torch.long)
        target_lengths = torch.randint(S_min, S, (N,), dtype=torch.long)

        # modify from numpy.ndarray to torch.tensor
        log_probs = torch.from_numpy(log_probs)
        
        ctc_loss = torch.nn.CTCLoss(zero_infinity=True, reduction=reduction_str)

        return ctc_loss, log_probs, targets, input_lengths, target_lengths

    def cpu_op_exec(self, ctc_loss, log_probs, targets, input_lengths, target_lengths):
        if log_probs.dtype == torch.float16:
            log_probs = log_probs.to(torch.float32)
    
        neg_log_likelihood = ctc_loss(log_probs.log_softmax(2), targets, input_lengths, target_lengths)

        neg_log_likelihood = neg_log_likelihood.numpy()

        return neg_log_likelihood

    def npu_op_exec(self, ctc_loss, log_probs, targets, input_lengths, target_lengths):
        log_probs = log_probs.npu()
        targets = targets.npu()
        input_lengths = input_lengths.npu()
        target_lengths = target_lengths.npu()
        
        neg_log_likelihood = ctc_loss(log_probs.log_softmax(2), targets, input_lengths, target_lengths)
                
        if neg_log_likelihood.dtype == torch.float16:
            neg_log_likelihood = neg_log_likelihood.to(torch.float32)

        neg_log_likelihood = neg_log_likelihood.cpu().numpy()

        return neg_log_likelihood

    def test_ctc_loss(self, device):
        sizes_list = [[50, 20, 16, 30, 10], [26, 37, 256, 18, 10]]
        para_reduction = ["sum", "mean", "none"]
        dtype = [np.float32, np.float16]
        shape_format = [
            [i, j, k] for i in sizes_list for j in dtype for k in para_reduction
        ]

        for item in shape_format:
            ctc_loss, log_probs, targets, input_lengths, target_lengths = self.generate_data(item)

            neg_log_likelihood_cpu = self.cpu_op_exec(ctc_loss, log_probs, targets, input_lengths, target_lengths)
            neg_log_likelihood_npu = self.npu_op_exec(ctc_loss, log_probs, targets, input_lengths, target_lengths)
            
            self.assertRtolEqual(neg_log_likelihood_cpu, neg_log_likelihood_npu, 1e-3)



instantiate_device_type_tests(TestCtcLoss, globals(), except_for='cpu')
if __name__ == "__main__":
    run_tests()
