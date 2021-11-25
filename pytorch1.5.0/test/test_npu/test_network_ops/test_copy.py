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
import numpy as np
from common_utils import TestCase, run_tests
from common_device_type import dtypes, instantiate_device_type_tests
from util_test import create_common_tensor

class TestCopy(TestCase):
    def test_copy_transpose(self, device):
        inputs = torch.randn(2, 3, 5)
        cpu_out = inputs.transpose(2, 0) + 1
        inputs = inputs.to("npu")
        npu_out = inputs.transpose(2, 0) + 1
        self.assertRtolEqual(cpu_out.detach().numpy(), npu_out.cpu().detach().numpy())

    def test_copy_permute_nd(self, device):
        inputs = torch.randn(2, 5, 6, 9)
        cpu_out = inputs.permute(2, 3, 0, 1) + 1
        inputs = inputs.to("npu").npu_format_cast(0)
        npu_out = inputs.permute(2, 3, 0, 1) + 1
        self.assertRtolEqual(cpu_out.detach().numpy(), npu_out.cpu().detach().numpy())

    def test_copy_permute_nd_optimize(self, device):
        inputs = torch.randn(32, 64, 15, 20, 1)
        cpu_out = inputs.permute(2, 3, 0, 1, 4) + 1
        inputs = inputs.to("npu").npu_format_cast(2)
        npu_out = inputs.permute(2, 3, 0, 1, 4) + 1
        self.assertRtolEqual(cpu_out.detach().numpy(), npu_out.cpu().detach().numpy())

    def test_copy_permute_5hd(self, device):
        inputs = torch.from_numpy(np.random.randn(2560,512,1,26).astype(np.float32))
        cpu_out = inputs.permute(2,3,0,1) + 1
        inputs = inputs.to("npu").npu_format_cast(3)
        npu_out = inputs.permute(2,3,0,1) + 1
        self.assertRtolEqual(cpu_out.detach().numpy(), npu_out.cpu().detach().numpy())

    def test_copy_squeeze_permute_nd(self, device):
        inputs = torch.from_numpy(np.random.randn(2560,512,1,26).astype(np.float32))
        cpu_out = inputs.squeeze(2).permute(1,2,0) + 1
        inputs = inputs.to("npu").npu_format_cast(0)
        npu_out = inputs.squeeze(2).permute(1,2,0) + 1
        self.assertRtolEqual(cpu_out.detach().numpy(), npu_out.cpu().detach().numpy())

    def test_copy_squeeze_unsqueeze_permute_5hd(self, device):
        inputs = torch.from_numpy(np.random.randn(1,512,1,26).astype(np.float32))
        cpu_out = inputs.squeeze().unsqueeze(1).unsqueeze(3).permute(1,3,2,0) + 1
        inputs = inputs.to("npu").npu_format_cast(3)
        npu_out = inputs.squeeze().unsqueeze(1).unsqueeze(3).permute(1,3,2,0) + 1
        self.assertRtolEqual(cpu_out.detach().numpy(), npu_out.cpu().detach().numpy())

    def test_copy_transpose_squeeze_permute_nd(self, device):
        inputs = torch.from_numpy(np.random.randn(16,512,1,26).astype(np.float32))
        cpu_out = inputs.transpose(1,3).squeeze().permute(2,1,0) + 1
        inputs = inputs.to("npu").npu_format_cast(0)
        npu_out = inputs.transpose(1,3).squeeze().permute(2,1,0) + 1
        self.assertRtolEqual(cpu_out.detach().numpy(), npu_out.cpu().detach().numpy())

    def test_copy_view_permute_nd(self, device):
        inputs = torch.from_numpy(np.random.randn(16,512,1,26).astype(np.float32))
        cpu_out = inputs.view(32,256,1,26).permute(2,1,0,3) + 1
        inputs = inputs.to("npu").npu_format_cast(0)
        npu_out = inputs.view(32,256,1,26).permute(2,1,0,3) + 1
        self.assertRtolEqual(cpu_out.detach().numpy(), npu_out.cpu().detach().numpy())

    def test_copy_narrow_5hd(self, device):
        inputs = torch.from_numpy(np.random.randn(256,256,28,28).astype(np.float32))
        cpu_out = torch.narrow(inputs, 1, 224, 32) + 1
        inputs = inputs.to("npu").npu_format_cast(3)
        npu_out = torch.narrow(inputs, 1, 224, 32) + 1
        self.assertRtolEqual(cpu_out.detach().numpy(), npu_out.cpu().detach().numpy())

    def test_copy_narrow_nd(self, device):
        inputs = torch.from_numpy(np.random.randn(256,256,28,28).astype(np.float32))
        narrow_1 = torch.narrow(inputs, 1, 224, 32)
        cpu_out = torch.narrow(narrow_1, 2, 14, 14) + 1
        inputs = inputs.to("npu").npu_format_cast(0)
        narrow_1 = torch.narrow(inputs, 1, 224, 32)
        npu_out = torch.narrow(narrow_1, 2, 14, 14) + 1
        self.assertRtolEqual(cpu_out.detach().numpy(), npu_out.cpu().detach().numpy())

    def test_copy_index_nd(self, device):
        inputs = torch.from_numpy(np.random.randn(256,256,28,28).astype(np.float32))
        narrow_1 = torch.narrow(inputs, 1, 32, 192)
        cpu_out = narrow_1[0:64, 32:128, 16:24, :] + 1
        inputs = inputs.to("npu").npu_format_cast(0)
        narrow_1 = torch.narrow(inputs, 1, 32, 192)
        npu_out = narrow_1[0:64, 32:128, 16:24, :] + 1
        self.assertRtolEqual(cpu_out.detach().numpy(), npu_out.cpu().detach().numpy())

    def test_copy_index_step_nd(self, device):
        inputs = torch.from_numpy(np.random.randn(256,256,28,28).astype(np.float32))
        cpu_out = inputs[0:64:2, 32:128:4, :, 6:22] + 1
        inputs = inputs.to("npu").npu_format_cast(0)
        npu_out = inputs[0:64:2, 32:128:4, :, 6:22] + 1
        self.assertRtolEqual(cpu_out.detach().numpy(), npu_out.cpu().detach().numpy())

    def test_copy_chunk(self, device):
        inputs = torch.from_numpy(np.random.randn(256,256,28,28).astype(np.float32))
        cpu_out = torch.chunk(inputs, 2, 1)
        chunk1_cpu = cpu_out[0] + 1
        chunk2_cpu = cpu_out[1] * 2
        inputs = inputs.to("npu")
        npu_out= torch.chunk(inputs, 2, 1)
        chunk1_npu = npu_out[0] + 1
        chunk2_npu = npu_out[1] * 2
        self.assertRtolEqual(chunk1_cpu.detach().numpy(), chunk1_npu.cpu().detach().numpy())
        self.assertRtolEqual(chunk2_cpu.detach().numpy(), chunk2_npu.cpu().detach().numpy())

    def test_copy_split(self, device):
        inputs = torch.from_numpy(np.random.randn(256,256,28,28).astype(np.float32))
        cpu_out = torch.chunk(inputs, 12, 2)
        chunk1_cpu = cpu_out[0] + 1
        chunk2_cpu = cpu_out[1] * 2
        chunk3_cpu = cpu_out[2].contiguous()
        inputs = inputs.to("npu")
        npu_out= torch.chunk(inputs, 12, 2)
        chunk1_npu = npu_out[0] + 1
        chunk2_npu = npu_out[1] * 2
        chunk3_npu = npu_out[2].contiguous()
        self.assertRtolEqual(chunk1_cpu.detach().numpy(), chunk1_npu.cpu().detach().numpy())
        self.assertRtolEqual(chunk2_cpu.detach().numpy(), chunk2_npu.cpu().detach().numpy())
        self.assertRtolEqual(chunk3_cpu.detach().numpy(), chunk3_npu.cpu().detach().numpy())

    def test_cross_device_copy_check(self, device):
        device_count = torch.npu.device_count()
        if device_count < 2:
            return
        
        inputs = torch.randn(2, 3, 5).to("npu")
        target_device = torch.npu.current_device() + 1
        if target_device >= device_count:
            target_device = 0
        with self.assertRaisesRegex(RuntimeError, "Cross-device copy is not supported."):
            inputs.to("npu:" + str(target_device))

instantiate_device_type_tests(TestCopy, globals(), except_for='cpu')
if __name__ == "__main__":
    run_tests()
