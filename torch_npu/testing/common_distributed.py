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

import unittest
import torch
import torch_npu


def skipIfUnsupportMultiNPU(npu_number_needed):
    def skip_dec(func):
        def wrapper(self):
            if not torch.npu.is_available() or torch.npu.device_count() < npu_number_needed:
                return unittest.SkipTest("Multi-NPU condition not satisfied")
            return func(self)
        return wrapper
    return skip_dec
