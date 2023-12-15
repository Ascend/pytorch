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

import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor


class TestSignBitsUnpack(TestCase):

    def sign_unpack(self, in_data, size, dtype):
        unpack_data = np.unpackbits(in_data, bitorder="little")
        unpack_data = unpack_data.astype(dtype)
        unpack_data = (unpack_data - 0.5) * 2.0
        return unpack_data.reshape(size, unpack_data.shape[0] // size)

    def cpu_op_exec(self, cpu_input, destype, size):
        cup_out = self.sign_unpack(cpu_input, size, destype)
        return cup_out

    def npu_op_exec(self, npu_input, destype, size):
        if(destype == "float16"):
            nup_out = torch_npu.npu_sign_bits_unpack(npu_input, size, torch.float16)
        elif destype == "float32":
            nup_out = torch_npu.npu_sign_bits_unpack(npu_input, size, torch.float32)

        nup_out = nup_out.to("cpu").numpy()
        return nup_out

    def test_sign_bits_unpack(self, device="npu"):
        shape = np.random.uniform(1, 10**5, 1)
        shape = shape // (10 ** int(np.random.uniform(0, int(np.log10(shape) + 1), 1)))
        shape = max(int(shape), 1)
        size = int(np.random.uniform(1, 100))
        shape = shape * size

        shape_format = [np.uint8, 2, [shape]]
        cpu_input, npu_input = create_common_tensor(shape_format, 0, 255)
        dest_dtype = ["float16", "float32"]
        for destype in dest_dtype:
            cpu_output = self.cpu_op_exec(cpu_input, destype, size)
            npu_output = self.npu_op_exec(npu_input, destype, size)
            self.assertRtolEqual(cpu_output, npu_output)


if __name__ == "__main__":
    run_tests()
