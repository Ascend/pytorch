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
from common_utils import TestCase, run_tests
from common_device_type import dtypes, instantiate_device_type_tests
 
class TestFormatCast(TestCase):
    def create_single_npu_tensor(self, item, minvalue, maxvalue):
        dtype = item[0]
        format = item[1]
        shape = item[2]
        input1 = np.random.uniform(minvalue, maxvalue, shape).astype(dtype)
        npu_input = torch.from_numpy(input1).to("npu")
        if format != -1:
            npu_input = npu_input.npu_format_cast(format)
        return npu_input
    
    def check_result(self, expectValue, retTensor):
        if retTensor.get_npu_format() != expectValue:
            print("expectValue: ", expectValue, " resultValue: ", retTensor.get_npu_format())
            sys.exit(-1)

    def test_format_cast_tensor(self, device):
        src_shape_format = [
            [np.float16, 0, (2, 2, 4, 4)],
            [np.float16, 2, (2, 2, 4, 4)]
        ]
        dst_shape_format = [
            [np.float16, 3, (2, 2, 4, 4)],
            [np.float16, 4, (2, 2, 4, 4)],
            [np.float16, 29, (2, 2, 4, 4)], 
            [np.float16, 30, (2, 2 ,2 , 4, 4)],
        ]

        for i in src_shape_format:
            src_tensor = self.create_single_npu_tensor(i, 1, 5)
            for j in dst_shape_format:
                dst_tensor = self.create_single_npu_tensor(j, 3, 6)
                result_tensor = src_tensor.npu_format_cast(dst_tensor)
                self.check_result(dst_tensor.get_npu_format(), result_tensor)

    def test_format_cast(self, device):
        shape_format = [np.float16, -1, (2, 2, 4, 4)]
        npu_tensor = self.create_single_npu_tensor(shape_format, 1, 5)
        npu_tensor = npu_tensor.npu_format_cast(2)
        self.check_result(2, npu_tensor)
        npu_tensor = npu_tensor.npu_format_cast(3)
        self.check_result(3, npu_tensor)
        npu_tensor = npu_tensor.npu_format_cast(0)
        self.check_result(0, npu_tensor)
        npu_tensor = npu_tensor.npu_format_cast(3)
        self.check_result(3, npu_tensor)
        npu_tensor = npu_tensor.npu_format_cast(2)
        self.check_result(0, npu_tensor)
        npu_tensor = npu_tensor.npu_format_cast(0)
        self.check_result(0, npu_tensor)
        npu_tensor = npu_tensor.npu_format_cast(2)
        self.check_result(2, npu_tensor)

        npu_tensor = npu_tensor.npu_format_cast(0)
        self.check_result(0, npu_tensor)
        npu_tensor = npu_tensor.npu_format_cast(4)
        self.check_result(4, npu_tensor)
        npu_tensor = npu_tensor.npu_format_cast(0)
        self.check_result(0, npu_tensor)
        npu_tensor = npu_tensor.npu_format_cast(29)
        self.check_result(29, npu_tensor)
        npu_tensor = npu_tensor.npu_format_cast(0)
        self.check_result(0, npu_tensor)
        npu_tensor = npu_tensor.npu_format_cast(2)
        self.check_result(2, npu_tensor)
        npu_tensor = npu_tensor.npu_format_cast(4)
        self.check_result(4, npu_tensor)
        npu_tensor = npu_tensor.npu_format_cast(0)
        self.check_result(0, npu_tensor)
        npu_tensor = npu_tensor.npu_format_cast(2)
        self.check_result(2, npu_tensor)
        npu_tensor = npu_tensor.npu_format_cast(29)
        self.check_result(29, npu_tensor)
        npu_tensor = npu_tensor.npu_format_cast(0)
        self.check_result(0, npu_tensor)

        npu_tensor = npu_tensor.view(2,2,2,2,4).clone()
        npu_tensor = npu_tensor.npu_format_cast(30)
        self.check_result(30, npu_tensor)
        npu_tensor = npu_tensor.npu_format_cast(33)
        self.check_result(33, npu_tensor)
        npu_tensor = npu_tensor.npu_format_cast(30)
        self.check_result(30, npu_tensor)
        npu_tensor = npu_tensor.npu_format_cast(2)
        self.check_result(2, npu_tensor)
        npu_tensor = npu_tensor.npu_format_cast(33)
        self.check_result(33, npu_tensor)
        npu_tensor = npu_tensor.npu_format_cast(2)
        self.check_result(2, npu_tensor)
        npu_tensor = npu_tensor.npu_format_cast(30)
        self.check_result(30, npu_tensor)
        npu_tensor = npu_tensor.npu_format_cast(32)
        self.check_result(32, npu_tensor)
        npu_tensor = npu_tensor.npu_format_cast(30)
        self.check_result(30, npu_tensor)
        npu_tensor = npu_tensor.npu_format_cast(2)
        self.check_result(2, npu_tensor)
        npu_tensor = npu_tensor.npu_format_cast(32)
        self.check_result(32, npu_tensor)
        npu_tensor = npu_tensor.npu_format_cast(2)
        self.check_result(2, npu_tensor)

    def test_format_cast_(self, device):
        shape_format = [np.float16, -1, (2, 2, 4, 4)]
        npu_tensor = self.create_single_npu_tensor(shape_format, 1, 5)
        npu_tensor = npu_tensor.npu_format_cast_(2)
        self.check_result(2, npu_tensor)
        npu_tensor = npu_tensor.npu_format_cast_(3)
        self.check_result(3, npu_tensor)
        npu_tensor = npu_tensor.npu_format_cast_(0)
        self.check_result(0, npu_tensor)
        npu_tensor = npu_tensor.npu_format_cast_(3)
        self.check_result(3, npu_tensor)
        npu_tensor = npu_tensor.npu_format_cast_(2)
        self.check_result(0, npu_tensor)
        npu_tensor = npu_tensor.npu_format_cast_(0)
        self.check_result(0, npu_tensor)
        npu_tensor = npu_tensor.npu_format_cast_(2)
        self.check_result(2, npu_tensor)

        npu_tensor = npu_tensor.npu_format_cast_(0)
        self.check_result(0, npu_tensor)
        npu_tensor = npu_tensor.npu_format_cast_(4)
        self.check_result(4, npu_tensor)
        npu_tensor = npu_tensor.npu_format_cast_(0)
        self.check_result(0, npu_tensor)
        npu_tensor = npu_tensor.npu_format_cast_(29)
        self.check_result(29, npu_tensor)
        npu_tensor = npu_tensor.npu_format_cast_(0)
        self.check_result(0, npu_tensor)
        npu_tensor = npu_tensor.npu_format_cast_(2)
        self.check_result(2, npu_tensor)
        npu_tensor = npu_tensor.npu_format_cast_(4)
        self.check_result(4, npu_tensor)
        npu_tensor = npu_tensor.npu_format_cast_(0)
        self.check_result(0, npu_tensor)
        npu_tensor = npu_tensor.npu_format_cast_(2)
        self.check_result(2, npu_tensor)
        npu_tensor = npu_tensor.npu_format_cast_(29)
        self.check_result(29, npu_tensor)
        npu_tensor = npu_tensor.npu_format_cast_(0)
        self.check_result(0, npu_tensor)

        npu_tensor = npu_tensor.view(2,2,2,2,4).clone()
        npu_tensor = npu_tensor.npu_format_cast_(30)
        self.check_result(30, npu_tensor)
        npu_tensor = npu_tensor.npu_format_cast_(33)
        self.check_result(33, npu_tensor)
        npu_tensor = npu_tensor.npu_format_cast_(30)
        self.check_result(30, npu_tensor)
        npu_tensor = npu_tensor.npu_format_cast_(2)
        self.check_result(2, npu_tensor)
        npu_tensor = npu_tensor.npu_format_cast_(33)
        self.check_result(33, npu_tensor)
        npu_tensor = npu_tensor.npu_format_cast_(2)
        self.check_result(2, npu_tensor)
        npu_tensor = npu_tensor.npu_format_cast_(30)
        self.check_result(30, npu_tensor)
        npu_tensor = npu_tensor.npu_format_cast_(32)
        self.check_result(32, npu_tensor)
        npu_tensor = npu_tensor.npu_format_cast_(30)
        self.check_result(30, npu_tensor)
        npu_tensor = npu_tensor.npu_format_cast_(2)
        self.check_result(2, npu_tensor)
        npu_tensor = npu_tensor.npu_format_cast_(32)
        self.check_result(32, npu_tensor)
        npu_tensor = npu_tensor.npu_format_cast_(2)
        self.check_result(2, npu_tensor)

    # UT for view + transdata scene 
    def test_format_cast_val(self, device):
        shape_format = [np.float32, -1, (10, 4)]
        npu_tensor = self.create_single_npu_tensor(shape_format, 1, 5)
        npu_tensor = npu_tensor.npu_format_cast(3)
        a = npu_tensor[1].npu_format_cast(0).contiguous()
        b = npu_tensor.npu_format_cast(0)[1].contiguous()
        a = a.to("cpu")
        b = b.to("cpu")
        self.assertRtolEqual(a, b)

instantiate_device_type_tests(TestFormatCast, globals(), except_for="cpu")
if __name__ == "__main__":
    run_tests()