# Copyright (c) 2020 Huawei Technologies Co., Ltd
# Copyright (c) 2019, Facebook CORPORATION. 
# All rights reserved.
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

# format value description：
# -1 ：FORMAT_UNDEFINED
#  0 ：FORMAT_NCHW
#  1 ：FORMAT_NHWC
#  2 ：FORMAT_ND
#  3 ：FORMAT_NC1HWC0
#  4 ：FORMAT_FRACTAL_Z
# 29 ：FORMAT_FRACTAL_NZ
def create_common_tensor(item, minValue, maxValue):
        dtype = item[0]
        format_tensor = item[1]
        shape = item[2]
        input1 = np.random.uniform(minValue, maxValue, shape).astype(dtype)
        cpu_input = torch.from_numpy(input1)
        npu_input = torch.from_numpy(input1).to("npu")
        if format_tensor != -1:
            npu_input = npu_input.npu_format_cast(format_tensor)
        return cpu_input, npu_input


threshold = 1.e-4
threshold2 = 1.e-3


def compare_res_new(cpu_output, npu_output, testcase_name):
    if cpu_output.shape != npu_output.shape:
        return print("result shape error!", cpu_output.shape, npu_output.shape)
    if cpu_output.dtype != npu_output.dtype:
        return print("result dtype error!", cpu_output.dtype, npu_output.dtype)
    if cpu_output.dtype == np.int32:
        result = np.equal(cpu_output, npu_output)
        if result is False:
            return print('testcase_name={0}, npu datatype={1} shape={2} fails!'.format(
                testcase_name, npu_output.dtype, npu_output.shape))
    elif cpu_output.dtype == np.float16:
        result = np.allclose(npu_output, cpu_output, threshold)
        if result is False:
            return print('testcase_name={0}, npu datatype={1} shape={2} fails!'.format(
                testcase_name, npu_output.dtype, npu_output.shape))
    elif cpu_output.dtype == np.float32:
        result = np.allclose(npu_output, cpu_output, 0, 0.0001)
        print(npu_output, cpu_output)
        print(result)
        if not result:
            return print('testcase_name={0}, npu datatype={1} shape={2} fails!'.format(
                testcase_name, npu_output.dtype, npu_output.shape))
    print('testcase_name={0}, datatype={1} shape={2} pass!'.format(testcase_name,cpu_output.dtype, cpu_output.shape))


def create_common_tensor_for_broadcast(item, minValue, maxValue):
    dtype = item[0]
    npu_format = item[1]
    shape = item[2]
    input1 = np.random.uniform(minValue, maxValue, shape[0]).astype(dtype)
    cpu_input = torch.from_numpy(input1)
    npu_input = torch.from_numpy(input1).to("npu")
    if npu_format != -1:
        npu_input = npu_input.npu_format_cast(npu_format)
    return cpu_input, npu_input
