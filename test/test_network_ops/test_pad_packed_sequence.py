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

import unittest
import torch
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests


class TestPadPackedSequence(TestCase):
    @unittest.skip("skip test_pad_packed_sequence_fp32 now")
    def test_pad_packed_sequence_fp32(self, device="npu"):
        data = torch.tensor([4, 1, 3, 5, 2, 6], dtype=torch.float32)
        batch_sizes = torch.tensor([3, 2, 1], dtype=torch.int64)
        cpu_out, cpu_lengths = torch._pad_packed_sequence(data, batch_sizes, False, 0, 6)
        npu_out, npu_lengths = torch._pad_packed_sequence(data.npu(), batch_sizes, False, 0, 6)
        self.assertRtolEqual(cpu_out, npu_out.cpu())
        self.assertRtolEqual(cpu_lengths, npu_lengths.cpu())

    @unittest.skip("skip test_pad_packed_sequence_fp16 now")
    def test_pad_packed_sequence_fp16(self, device="npu"):
        data = torch.tensor([4, 1, 3, 5, 2, 6], dtype=torch.float16)
        batch_sizes = torch.tensor([3, 2, 1], dtype=torch.int64)
        cpu_out, cpu_lengths = torch._pad_packed_sequence(data, batch_sizes, False, 0, 6)
        npu_out, npu_lengths = torch._pad_packed_sequence(data.npu(), batch_sizes, False, 0, 6)
        self.assertRtolEqual(cpu_out, npu_out.cpu())
        self.assertRtolEqual(cpu_lengths, npu_lengths.cpu())


if __name__ == "__main__":
    run_tests()
