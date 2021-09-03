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
import sys
import copy
import torch.nn as nn
from common_utils import TestCase, run_tests
from common_device_type import dtypes, instantiate_device_type_tests
from util_test import create_common_tensor

class TestNormal(TestCase):
    @dtypes(torch.float, torch.float16)
    def test_normal(self, device, dtype):
        q = torch.empty(100, 100, dtype=dtype, device="cpu").to("npu")
        q.normal_()
        self.assertEqual(q.mean(), 0, 0.2)
        self.assertEqual(q.to("cpu").to(torch.float).std(), 1, 0.2)

        q.normal_(2, 3)
        self.assertEqual(q.mean(), 2, 0.3)
        self.assertEqual(q.to("cpu").to(torch.float).std(), 3, 0.3)

        mean = torch.empty(100, 100, dtype=dtype, device="cpu").to("npu")
        std = torch.empty(100, 100, dtype=dtype, device="cpu").to("npu")
        mean.fill_(-2)
        std.fill_(3)

        r = torch.normal(mean)
        self.assertEqual(r.mean(), -2, 0.2)
        self.assertEqual(r.to("cpu").to(torch.float).std(), 1, 0.2)

        r = torch.normal(mean, 3)
        self.assertEqual(r.mean(), -2, 0.2)
        self.assertEqual(r.to("cpu").to(torch.float).std(), 3, 0.2)

        r = torch.normal(2, std)
        self.assertEqual(r.mean(), 2, 0.2)
        self.assertEqual(r.to("cpu").to(torch.float).std(), 3, 0.2)

        r = torch.normal(mean, std)
        self.assertEqual(r.mean(), -2, 0.2)
        self.assertEqual(r.to("cpu").to(torch.float).std(), 3, 0.3)

        r = torch.normal(2, 3, (100, 100))
        self.assertEqual(r.mean(), 2, 0.2)
        self.assertEqual(r.to("cpu").to(torch.float).std(), 3, 0.2)
        

instantiate_device_type_tests(TestNormal, globals(), except_for="cpu")
if __name__ == "__main__":
    run_tests()