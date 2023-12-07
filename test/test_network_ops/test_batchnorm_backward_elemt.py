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


class TestBatchNormBackwardElemt(TestCase):

    def test_batch_norm_backward_elemt_4d(self):
        grad_output = torch.ones([2, 3, 1, 4]).npu()
        input1 = torch.ones([2, 3, 1, 4]).npu()
        mean = torch.tensor([8., 5., 9.]).npu()
        invstd = torch.tensor([2., 1., 2.]).npu()
        weight = torch.tensor([1., 1., 4.]).npu()
        mean_dy = torch.tensor([2., 2., 6.]).npu()
        mean_dy_xmn = torch.tensor([2., 3., 11.]).npu()
        count_tensor = torch.tensor([5, 5, 5], dtype=torch.int32).npu()

        grad_input = torch.batch_norm_backward_elemt(grad_output, input1, mean, invstd,
                                                     weight, mean_dy, mean_dy_xmn, count_tensor)
        cuda_expect_out = torch.tensor([[[[9.2000, 9.2000, 9.2000, 9.2000]],
                                         [[1.6667, 1.6667, 1.6667, 1.6667]],
                                         [[192.5333, 192.5333, 192.5333, 192.5333]]],
                                        [[[9.2000, 9.2000, 9.2000, 9.2000]],
                                         [[1.6667, 1.6667, 1.6667, 1.6667]],
                                         [[192.5333, 192.5333, 192.5333, 192.5333]]]])
        self.assertRtolEqual(grad_input.cpu(), cuda_expect_out)

    def test_batch_norm_backward_elemt_2d(self):
        grad_output = torch.ones([2, 3]).npu()
        input1 = torch.ones([2, 3]).npu()
        mean = torch.tensor([8., 5., 9.]).npu()
        invstd = torch.tensor([2., 1., 2.]).npu()
        weight = torch.tensor([1., 1., 4.]).npu()
        mean_dy = torch.tensor([2., 2., 6.]).npu()
        mean_dy_xmn = torch.tensor([2., 3., 11.]).npu()
        count_tensor = torch.tensor([5, 5, 5], dtype=torch.int32).npu()

        grad_input = torch.batch_norm_backward_elemt(grad_output, input1, mean, invstd,
                                                     weight, mean_dy, mean_dy_xmn, count_tensor)
        cuda_expect_out = torch.tensor([[9.2000, 1.6667, 192.5333],
                                        [9.2000, 1.6667, 192.5333]])
        self.assertRtolEqual(grad_input.cpu(), cuda_expect_out)

    def test_batch_norm_backward_elemt_2d_fp(self):
        grad_output = torch.ones([2, 3]).npu()
        input1 = torch.ones([2, 3]).npu()
        mean = torch.tensor([8.123456, 5.147125, 9.365778]).npu()
        invstd = torch.tensor([2.65485, 1.36541, 2.25879]).npu()
        weight = torch.tensor([1.36987, 1.36944, 4.25774]).npu()
        mean_dy = torch.tensor([2., 2., 6.]).npu()
        mean_dy_xmn = torch.tensor([2., 3., 11.]).npu()
        count_tensor = torch.tensor([5, 5, 5], dtype=torch.int32).npu()

        grad_input = torch.batch_norm_backward_elemt(grad_output, input1, mean, invstd,
                                                     weight, mean_dy, mean_dy_xmn, count_tensor)
        cuda_expect_out = torch.tensor([[27.4980, 4.5119, 306.8037],
                                        [27.4980, 4.5119, 306.8037]])
        self.assertRtolEqual(grad_input.cpu(), cuda_expect_out)


if __name__ == "__main__":
    run_tests()
