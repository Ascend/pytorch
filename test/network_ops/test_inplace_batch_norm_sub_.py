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
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor


# affine = False,目前测试报错。所以本UT未做affine=False测试


class TestBatchNormSub_(TestCase):
    def cpu_op_exec(self, input1, num_features, affine):
        m = torch.nn.BatchNorm2d(num_features, affine=affine)
        m2 = torch.nn.BatchNorm2d(num_features, affine=affine)

        out1 = m.running_mean + m.running_mean
        out2 = m.running_var / 2
        output = m(input1)

        out3 = m2.running_var * 3
        output = m2(output)

        out4 = m2.running_mean - 6
        m2.running_mean.sub_(100)

        out1 = out1.detach().numpy()
        out2 = out2.detach().numpy()
        out3 = out3.detach().numpy()
        out4 = out4.detach().numpy()
        return out1, out2, out3, out4

    def npu_op_exec_new(self, input1, num_features, affine):
        m = torch.nn.BatchNorm2d(num_features, affine=affine)
        m2 = torch.nn.BatchNorm2d(num_features, affine=affine)
        input1 = input1.npu()
        m = m.npu()
        m2 = m2.npu()

        out1 = m.running_mean + m.running_mean
        out2 = m.running_var / 2
        output = m(input1)

        out3 = m2.running_var * 3
        output = m2(output)

        out4 = m2.running_mean - 6
        m2.running_mean.sub_(100)

        out1 = out1.to("cpu").detach().numpy()
        out2 = out2.to("cpu").detach().numpy()
        out3 = out3.to("cpu").detach().numpy()
        out4 = out4.to("cpu").detach().numpy()
        return out1, out2, out3, out4

    def test_batchnorm_shape_format(self):
        shape_format = [[[np.float32, -1, (10, 100, 14, 14)], True]]

        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item[0], 0, 10)
            cpu_output1, cpu_output2, cpu_output3, cpu_output4 = \
                self.cpu_op_exec(cpu_input1, item[0][2][1], item[1])
            npu_output1, npu_output2, npu_output3, npu_output4 = \
                self.npu_op_exec_new(npu_input1, item[0][2][1], item[1])
            self.assertRtolEqual(cpu_output1, npu_output1)
            self.assertRtolEqual(cpu_output2, npu_output2)
            self.assertRtolEqual(cpu_output3, npu_output3)
            self.assertRtolEqual(cpu_output4, npu_output4)


if __name__ == "__main__":
    run_tests()
