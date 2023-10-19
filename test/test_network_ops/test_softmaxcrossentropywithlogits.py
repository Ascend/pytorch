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


class TestSoftmaxCrossentropyWithLogits(TestCase):
    def npu_op_exec(self, input1, label):
        output = torch_npu.npu_softmax_cross_entropy_with_logits(input1, label)
        output = output.to("cpu")
        output = output.numpy()
        return output

    def test_softmaxcross(self, device="npu"):
        input1 = torch.tensor([[1., 2., 3., 4.]]).npu()
        label = torch.tensor([[1., 2., 3., 4.]]).npu()
        exresult = torch.tensor([14.4019])
        output = self.npu_op_exec(input1, label)
        self.assertRtolEqual(exresult.numpy(), output)


if __name__ == "__main__":
    run_tests()
