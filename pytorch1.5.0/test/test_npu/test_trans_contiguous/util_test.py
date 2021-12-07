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
import os
import sys
import numpy as np
import torch

common_path = os.path.dirname("../common/")
if common_path not in sys.path:
    sys.path.append(common_path)
from util_test_new import create_common_tensor, test_2args_broadcast, create_dtype_tensor, UT_FAST_MODE

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