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
import torch_npu
import numpy as np
import os


UT_FAST_MODE = os.getenv('UT_FAST_MODE') == '1' 


def set_npu_device():
    npu_device = get_npu_device()
    torch.npu.set_device(npu_device)
    print(f"Your device is {npu_device}")
    return npu_device


def get_npu_device():
    npu_device = os.environ.get('SET_NPU_DEVICE')
    if npu_device is None:
        npu_device = "npu:0"
    else:
        npu_device = f"npu:{npu_device}"
    return npu_device


def create_common_tensor(item, minValue, maxValue, device=None):
    if device is None:
        device = get_npu_device()
        
    dtype = item[0]
    npu_format = item[1]
    shape = item[2]
    input1 = np.random.uniform(minValue, maxValue, shape).astype(dtype)
    cpu_input = torch.from_numpy(input1)
    npu_input = torch.from_numpy(input1).to(device)
    if npu_format != -1:
        npu_input = torch_npu.npu_format_cast(npu_input, npu_format)
    return cpu_input, npu_input

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
        result = np.allclose(npu_output, cpu_output, 0.0001, 0)
        if result is False:
            return print('testcase_name={0}, npu datatype={1} shape={2} fails!'.format(
                testcase_name, npu_output.dtype, npu_output.shape))
    elif cpu_output.dtype == np.float32:
        result = np.allclose(npu_output, cpu_output, 0.0001, 0)
        print(npu_output, cpu_output)
        print(result)
        if not result:
            return print('testcase_name={0}, npu datatype={1} shape={2} fails!'.format(
                testcase_name, npu_output.dtype, npu_output.shape))
    print('testcase_name={0}, datatype={1} shape={2} pass!'.format(testcase_name, cpu_output.dtype, cpu_output.shape))


def __generate_2args_broadcast_cases(device=None):
    if device is None:
        device = get_npu_device()
        
    # Set broadcast and no axis, i.e. broadcasting 1.
    X = np.random.rand(2, 3, 4, 5).astype(np.float32)
    Y = np.random.rand(1, 1, 1).astype(np.float32)

    cpu_x = torch.from_numpy(X)
    npu_x = torch.from_numpy(X).to(device)

    cpu_y = torch.from_numpy(Y)
    npu_y = torch.from_numpy(Y).to(device)

    yield cpu_x, cpu_y, npu_x, npu_y

    # broadcasting last two dimensions.
    X = np.random.rand(2, 3, 4, 5).astype(np.float32)
    Y = np.random.rand(4, 5).astype(np.float32)

    cpu_x = torch.from_numpy(X)
    npu_x = torch.from_numpy(X).to(device)

    cpu_y = torch.from_numpy(Y)
    npu_y = torch.from_numpy(Y).to(device)

    yield cpu_x, cpu_y, npu_x, npu_y

def test_2args_broadcast(fn):
    output_list = []
    for cpu_x, cpu_y, npu_x, npu_y in __generate_2args_broadcast_cases():
        cpu_out = fn(cpu_x, cpu_y).numpy()
        npu_out = fn(npu_x, npu_y).to("cpu").numpy()
        output_list.append([cpu_out, npu_out])

    return output_list


def create_dtype_tensor(shape, dtype, npu_format=-1, min_value=-5, max_value=5, no_zero=False, device=None):
    if device is None:
        device = get_npu_device()
        
    if dtype == torch.bool:
        x = np.random.randint(0, 2, size=shape).astype(np.bool)

    elif dtype == torch.half:
        x = np.random.uniform(min_value, max_value, shape).astype(np.float16)
    
    elif dtype == torch.float:
        x = np.random.uniform(min_value, max_value, shape).astype(np.float32)

    else:
        x = np.random.randint(min_value, max_value+1, size = shape).astype(np.int32)

    if no_zero:
        ones = np.ones_like(x)
        x = np.where(x != 0, x, ones)

    cpu_input = torch.from_numpy(x)
    npu_input = torch.from_numpy(x).to(device)
    if npu_format != -1 and (dtype in [torch.float, torch.half]):
        npu_input = torch_npu.npu_format_cast(npu_input, npu_format)
    return cpu_input, npu_input

def check_operators_in_prof(expected_operators, prof, unexpected_operators=None):
    unexpected_operators = unexpected_operators or []
    prof_key_averages = prof.key_averages()
    if not prof_key_averages:
        return print("torch profiling is empty, please check it")
    for prof_item in prof_key_averages:        
        if prof_item.key in unexpected_operators:
            # if unexpected oprators are called, pattern inferring in trans-contiguous is failed
            return False
        elif prof_item.key in expected_operators:
            # if expected oprator is called, empty it in expected_operators list
            expected_operators.remove(prof_item.key)
            
    # if expected_operators list is empty, all oprators have been called
    if not expected_operators:
        return True
    return False