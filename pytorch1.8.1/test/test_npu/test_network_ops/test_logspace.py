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

class TestLogSpace(TestCase):

    def cpu_op_exec(self, start, end, steps, base):
        output = torch.logspace(start=start, end=end, steps=steps, base=base)
        output = output.numpy()
        return output

    def npu_op_exec(self, start, end, steps, base):
        output = torch.logspace(start=start, end=end, steps=steps, base=base, device="npu")
        output = output.to("cpu")
        output = output.numpy()
        return output    

    def npu_op_exec_out(self, start, end, steps, base, dtype):
        output = torch.randn(steps)
        torch.logspace(start=start, end=end, steps=steps, base=base, dtype=dtype, out=output)
        output = output.to("cpu")
        output = output.numpy()
        return output

    def test_logspace_common_shape_format(self, device):
        shape_format = [
                [0.0, 1.0, 10, 0.2, torch.float32],
                [2.0, 3.0, 10, 0.05, torch.float32],
                [10.0, 10.5, 11, 0.2, torch.float32],
                [10.0, 10.5, 110, 0.2, torch.float32],
                [0.0, 0.1, 20, 1.2, torch.float32],
                [0.5, 1.0, 50, 8.0, torch.float32],
                [1.0, 2.0, 2, -0.5, torch.float32],
                [0.0, 0.0, 1, 0.0, torch.float32],
                [1.0, 1.0, 1, 0.0, torch.float32],
                [1.0, 1.0, 0, 0.0, torch.float32],
                [1.0, 2.0, 9, 0.0, torch.float32]
        ] 

        for item in shape_format:
            cpu_output = self.cpu_op_exec(item[0], item[1], item[2], item[3])
            npu_output = self.npu_op_exec(item[0], item[1], item[2], item[3])
            self.assertRtolEqual(cpu_output, npu_output)
            npu_out_output = self.npu_op_exec_out(item[0], item[1], item[2], item[3], item[4])
            self.assertRtolEqual(cpu_output, npu_out_output)
    def test_logspace_float16_shape_format(self, device):
        def cpu_op_exec_fp16(start, end, steps, base, dtype):
            output = torch.logspace(start=start, end=end, steps=steps, base=base, dtype=torch.float32)
            output = output.numpy()
            output = output.astype(np.float16)
            return output

        def npu_op_exec(start, end, steps, base, dtype):
            output = torch.logspace( start=start, end=end, steps=steps, base=base, dtype=dtype, device="npu" )
            output = output.to("cpu")
            output = output.numpy()
            return output

        shape_format = [
                [-2.0, 2.0, 32, 32, torch.float16],
                [0.0, 1.0, 10, 0.2, torch.float16],
                [2.0, 3.0, 10, 0.05, torch.float16],
                [0.0, 0.1, 20, 1.2, torch.float16],
                [0.5, 1.0, 50, 8.0, torch.float16],
                [1.0, 2.0, 2, -0.5, torch.float16],
                [0.0, 0.0, 1, 0.0, torch.float16]
        ] 

        for item in shape_format:
            cpu_output = cpu_op_exec_fp16(item[0], item[1], item[2], item[3], item[4])
            npu_output = npu_op_exec(item[0], item[1], item[2], item[3], item[4])
            self.assertRtolEqual(cpu_output, npu_output)


instantiate_device_type_tests(TestLogSpace, globals(), except_for='cpu')
if __name__ == "__main__":
    run_tests()
