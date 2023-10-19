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

from torch_npu.testing.testcase import TestCase, run_tests


class TestPackPaddedSequence(TestCase):
    def test_pack_padded_sequence_fp32(self, device="npu"):
        data = torch.randn(6, 3, 2, dtype=torch.float32).npu()
        lengths = torch.tensor([6, 5, 3], dtype=torch.int64)
        expect_dim2 = data.view(18, 2).cpu()
        expect_batch_sizes = torch.tensor([3, 3, 3, 2, 2, 1], dtype=torch.int64)
        out_dim2, batch_sizes = torch._pack_padded_sequence(data, lengths, False)
        self.assertRtolEqual(expect_dim2, out_dim2.cpu())
        self.assertRtolEqual(expect_batch_sizes, batch_sizes.cpu())

    def test_pack_padded_sequence_fp16(self, device="npu"):
        data = torch.randn(6, 3, 2, dtype=torch.float16).npu()
        lengths = torch.tensor([6, 5, 3], dtype=torch.int64)
        expect_dim2 = data.view(18, 2).cpu()
        expect_batch_sizes = torch.tensor([3, 3, 3, 2, 2, 1], dtype=torch.int64)
        out_dim2, batch_sizes = torch._pack_padded_sequence(data, lengths, False)
        self.assertRtolEqual(expect_dim2, out_dim2.cpu())
        self.assertRtolEqual(expect_batch_sizes, batch_sizes.cpu())


if __name__ == "__main__":
    run_tests()
