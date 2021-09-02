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
import random
import numpy as np
import sys
from common_utils import TestCase, run_tests
from common_device_type import dtypes, instantiate_device_type_tests
from util_test import create_common_tensor

class TestCudnnRnnBackward(TestCase):
    def generate_bool(self):
        scalar = random.randint(1, 2)
        return scalar == 1

    def generate_int(self, min_d, max_d):
        scalar = random.randint(min_d, max_d)
        return scalar

    def cpu_op_exec(self, input1, vocab_size, num_hiddens, num_steps, batch_size, first, drop, bid):
        input1.requires_grad_(True)
        m = torch.nn.RNN(input_size=vocab_size, hidden_size=num_hiddens, batch_first=first, dropout=drop, bidirectional=bid)
        state = None
        output, _ = m(input1, state)
        w = torch.ones_like(output)
        output = output.backward(w)
        return input1.grad

    def npu_op_exec(self, input1, vocab_size, num_hiddens, num_steps, batch_size, first, drop, bid):
        input1.requires_grad_(True)
        m = torch.nn.RNN(input_size=vocab_size, hidden_size=num_hiddens, batch_first=first, dropout=drop, bidirectional=bid)
        m = m.npu()
        state = None
        output, _ = m(input1, state)
        w = torch.ones_like(output)
        output = output.backward(w)
        out = input1.grad
        out = out.to("cpu")
        return out

    def test_cudnn_rnn_backward_common_shape_format(self, device):
        npu_vocab_size = self.generate_int(1, 10)
        npu_num_hiddens = self.generate_int(1, 10)
        npu_num_step = self.generate_int(1, 10)
        npu_batch_size = self.generate_int(1, 10)
        first = self.generate_bool()
        drop = self.generate_int(0, 1)
        bid = self.generate_bool()
        shape_format = [
            [[np.float32, -1, (npu_num_step, npu_batch_size, npu_vocab_size)]],
            [[np.float32, 0, (npu_num_step, npu_batch_size, npu_vocab_size)]],
            [[np.float32, 3, (npu_num_step, npu_batch_size, npu_vocab_size)]],
        ]
        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item[0], 1, 100)
            cpu_result = self.cpu_op_exec(cpu_input1, npu_vocab_size, npu_num_hiddens, npu_num_step, npu_batch_size, first, drop, bid)
            npu_result = self.npu_op_exec(npu_input1, npu_vocab_size, npu_num_hiddens, npu_num_step, npu_batch_size, first, drop, bid)
            self.assertRtolEqual(cpu_result.numpy(), npu_result.numpy());

    def test_cudnn_rnn_backward_float16_shape_format(self, device):
        npu_vocab_size = self.generate_int(1, 10)
        npu_num_hiddens = self.generate_int(1, 10)
        npu_num_step = self.generate_int(1, 10)
        npu_batch_size = self.generate_int(1, 10)
        first = self.generate_bool()
        drop = self.generate_int(0, 1)
        bid = self.generate_bool()
        shape_format = [
            [[np.float16, -1, (npu_num_step, npu_batch_size, npu_vocab_size)]],
            [[np.float16, 0, (npu_num_step, npu_batch_size, npu_vocab_size)]],
            [[np.float16, 3, (npu_num_step, npu_batch_size, npu_vocab_size)]],
        ]

        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item[0], 10, 100)
            cpu_input1 = cpu_input1.to(torch.float32)
            cpu_result = self.cpu_op_exec(cpu_input1, npu_vocab_size, npu_num_hiddens, npu_num_step, npu_batch_size, first, drop, bid)
            npu_result = self.npu_op_exec(npu_input1, npu_vocab_size, npu_num_hiddens, npu_num_step, npu_batch_size, first, drop, bid)
            self.assertRtolEqual(cpu_result.numpy().astype(np.float16), npu_result.numpy().astype(np.float16));

instantiate_device_type_tests(TestCudnnRnnBackward, globals(), except_for="cpu")
if __name__ == "__main__":
    torch.npu.set_device("npu:5")
    run_tests()
