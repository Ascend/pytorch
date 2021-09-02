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
from common_utils import TestCase, run_tests
from common_device_type import instantiate_device_type_tests

class TestPadPackedSequence(TestCase):
    def test_pad_packed_sequence_fp32(self, device):
        data = torch.randn(6, 3, 16, dtype = torch.float32).npu()
        batch_sizes = torch.tensor([3, 3, 3, 2, 2, 1], dtype = torch.int64)
        expect_lengths = torch.tensor([6, 5, 3], dtype = torch.int64)
        out, lengths = torch._pad_packed_sequence(data, batch_sizes, False, 0, 6)
        self.assertRtolEqual(data.cpu(), out.cpu())
        self.assertRtolEqual(expect_lengths, lengths.cpu())

instantiate_device_type_tests(TestPadPackedSequence, globals(), except_for="cpu")
if __name__ == "__main__":
    run_tests()
