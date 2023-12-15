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

import sys
import torch
import numpy as np
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests


class TestFormatCast(TestCase):
    def create_single_npu_tensor(self, item, minvalue, maxvalue):
        dtype = item[0]
        format1 = item[1]
        shape = item[2]
        input1 = np.random.uniform(minvalue, maxvalue, shape).astype(dtype)
        npu_input = torch.from_numpy(input1).to("npu")
        if format1 != -1:
            npu_input = torch_npu.npu_format_cast(npu_input, format1)
        return npu_input

    def check_result(self, expectValue, retTensor):
        if torch_npu.get_npu_format(retTensor) != expectValue:
            raise RuntimeError(f"expectValue: {expectValue},  resultValue: {torch_npu.get_npu_format(retTensor)}")

    def test_format_cast_backward(self, device="npu"):
        a = torch.rand(2, 3).npu()
        a.requires_grad = True
        b = torch_npu.npu_format_cast(a, 29)
        if b.requires_grad is not True:
            raise RuntimeError("the output.requires_grad of npu_format_cast should be same with input, but not so.")

    def test_format_cast_tensor(self, device="npu"):
        src_shape_format = [
            [np.float16, 0, (2, 2, 4, 4)],
            [np.float16, 2, (2, 2, 4, 4)]
        ]
        dst_shape_format = [
            [np.float16, 3, (2, 2, 4, 4)],
            [np.float16, 4, (2, 2, 4, 4)],
            [np.float16, 29, (2, 2, 4, 4)],
            [np.float16, 30, (2, 2, 2, 4, 4)],
        ]

        for i in src_shape_format:
            src_tensor = self.create_single_npu_tensor(i, 1, 5)
            for j in dst_shape_format:
                dst_tensor = self.create_single_npu_tensor(j, 3, 6)
                result_tensor = torch_npu.npu_format_cast(src_tensor, dst_tensor)
                self.check_result(torch_npu.get_npu_format(dst_tensor), result_tensor)

    def test_format_cast(self, device="npu"):
        shape_format = [np.float16, -1, (2, 2, 4, 4)]
        npu_tensor = self.create_single_npu_tensor(shape_format, 1, 5)

        npu_tensor = torch_npu.npu_format_cast(npu_tensor, 2)
        self.check_result(2, npu_tensor)
        npu_tensor = torch_npu.npu_format_cast(npu_tensor, 3)
        self.check_result(3, npu_tensor)
        npu_tensor = torch_npu.npu_format_cast(npu_tensor, 0)
        self.check_result(0, npu_tensor)
        npu_tensor = torch_npu.npu_format_cast(npu_tensor, 3)
        self.check_result(3, npu_tensor)
        npu_tensor = torch_npu.npu_format_cast(npu_tensor, 2)
        self.check_result(0, npu_tensor)

        npu_format_list = [0, 2, 0, 4, 0, 29, 0, 2, 4, 0, 2, 29, 0]
        for npu_format in npu_format_list:
            npu_tensor = torch_npu.npu_format_cast(npu_tensor, npu_format)
            self.check_result(npu_format, npu_tensor)

        npu_tensor = npu_tensor.view(2, 2, 2, 2, 4).clone()

        npu_format_list = [30, 33, 30, 2, 33, 2, 30, 32, 30, 2, 32, 2]
        for npu_format in npu_format_list:
            npu_tensor = torch_npu.npu_format_cast(npu_tensor, npu_format)
            self.check_result(npu_format, npu_tensor)

    def test_format_cast_inplace(self, device="npu"):
        shape_format = [np.float16, -1, (2, 2, 4, 4)]
        npu_tensor = self.create_single_npu_tensor(shape_format, 1, 5)

        npu_tensor = torch_npu.npu_format_cast_(npu_tensor, 2)
        self.check_result(2, npu_tensor)
        npu_tensor = torch_npu.npu_format_cast_(npu_tensor, 3)
        self.check_result(3, npu_tensor)
        npu_tensor = torch_npu.npu_format_cast_(npu_tensor, 0)
        self.check_result(0, npu_tensor)
        npu_tensor = torch_npu.npu_format_cast_(npu_tensor, 3)
        self.check_result(3, npu_tensor)
        npu_tensor = torch_npu.npu_format_cast_(npu_tensor, 2)
        self.check_result(0, npu_tensor)

        npu_format_list = [0, 2, 0, 4, 0, 29, 0, 2, 4, 0, 2, 29, 0]
        for npu_format in npu_format_list:
            npu_tensor = torch_npu.npu_format_cast_(npu_tensor, npu_format)
            self.check_result(npu_format, npu_tensor)

        npu_tensor = npu_tensor.view(2, 2, 2, 2, 4).clone()

        npu_format_list = [30, 33, 30, 2, 33, 2, 30, 32, 30, 2, 32, 2]
        for npu_format in npu_format_list:
            npu_tensor = torch_npu.npu_format_cast_(npu_tensor, npu_format)
            self.check_result(npu_format, npu_tensor)

    # UT for view + transdata scene
    def test_format_cast_val(self, device="npu"):
        shape_format = [np.float32, -1, (10, 4)]
        npu_tensor = self.create_single_npu_tensor(shape_format, 1, 5)
        npu_tensor = torch_npu.npu_format_cast(npu_tensor, 3)
        a = torch_npu.npu_format_cast(npu_tensor[1], 0).contiguous()
        b = torch_npu.npu_format_cast(npu_tensor, 0)[1].contiguous()
        a = a.to("cpu")
        b = b.to("cpu")
        self.assertRtolEqual(a, b)

    def test_format_cast_equal(self):
        a = torch.randn(2, 3).npu()
        a.requires_grad = True
        ori_format = torch_npu.get_npu_format(a)
        b = torch_npu.npu_format_cast(a, ori_format)
        res = b.sum()
        res.backward()


if __name__ == "__main__":
    run_tests()
