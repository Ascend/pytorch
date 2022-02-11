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
import torch.nn as nn
import numpy as np

from torch_npu.testing.common_utils import TestCase, run_tests
from torch_npu.testing.common_device_type import Dtypes, instantiate_device_type_tests
from torch_npu.testing.util_test import create_common_tensor

class TestRandom(TestCase):
    @dtypes(torch.int32, torch.int64, torch.float, torch.float16)
    def test_random_from_to(self, device, dtype):
        size = 2000
        alpha = 0.1

        int64_min_val = torch.iinfo(torch.int64).min
        int64_max_val = torch.iinfo(torch.int64).max

        if dtype in [torch.float, torch.float16]:
            min_val = int(max(torch.finfo(dtype).min, int64_min_val))
            max_val = int(min(torch.finfo(dtype).max, int64_max_val))
            froms = [min_val, -42, 0]
            tos = [42, max_val >> 1, max_val]
        elif dtype == torch.int64:
            min_val = int64_min_val
            max_val = int64_max_val
            froms = [min_val, -42, 0]
            tos = [42, max_val]
        else:
            min_val = torch.iinfo(dtype).min
            max_val = torch.iinfo(dtype).max
            froms = [min_val, -42, 0]
            tos = [42, max_val]

        for from_ in froms:
            for to_ in tos:
                t = torch.empty(size, dtype=dtype, device="cpu").to("npu")
                if to_ > from_:
                    if not (min_val <= from_ <= max_val) or not (min_val <= (to_ - 1) <= max_val):
                        if not (min_val <= from_ <= max_val):
                            self.assertWarnsRegex(
                                lambda: t.random_(from_, to_),
                                "from is out of bounds"
                            )
                        if not (min_val <= (to_ - 1) <= max_val):
                            self.assertWarnsRegex(
                                lambda: t.random_(from_, to_),
                                "to - 1 is out of bounds"
                            )
                    else:
                        t.random_(from_, to_)
                        range_ = to_ - from_
                        delta = max(1, alpha * range_)
                        self.assertTrue(from_ <= t.to(torch.double).min() < (from_ + delta))
                        self.assertTrue((to_ - delta) <= t.to(torch.double).max() < to_)
                else:
                    self.assertRaisesRegex(
                        RuntimeError,
                        "random_ expects 'from' to be less than 'to', but got from=" + str(from_) + " >= to=" + str(to_),
                        lambda: t.random_(from_, to_)
                    )
    
    @dtypes(torch.int32, torch.int64, torch.float, torch.float16)
    def test_random_to(self, device, dtype):
        size = 2000
        alpha = 0.1

        int64_min_val = torch.iinfo(torch.int64).min
        int64_max_val = torch.iinfo(torch.int64).max

        if dtype in [torch.float, torch.float16]:
            min_val = int(max(torch.finfo(dtype).min, int64_min_val))
            max_val = int(min(torch.finfo(dtype).max, int64_max_val))
            tos = [42, max_val >> 1]
        elif dtype == torch.int64:
            min_val = int64_min_val
            max_val = int64_max_val
            tos = [42, max_val]
        else:
            min_val = torch.iinfo(dtype).min
            max_val = torch.iinfo(dtype).max
            tos = [42, max_val]

        from_ = 0
        for to_ in tos:
            t = torch.empty(size, dtype=dtype, device="cpu").to("npu")
            if to_ > from_:
                if not (min_val <= (to_ - 1) <= max_val):
                    self.assertWarnsRegex(
                        lambda: t.random_(to_),
                        "to - 1 is out of bounds"
                    )
                else:
                    t.random_(to_)
                    range_ = to_ - from_
                    delta = max(1, alpha * range_)
                    self.assertTrue(from_ <= t.to(torch.double).min() < (from_ + delta))
                    self.assertTrue((to_ - delta) <= t.to(torch.double).max() < to_)
            else:
                self.assertRaisesRegex(
                    RuntimeError,
                    "random_ expects 'from' to be less than 'to', but got from=" + str(from_) + " >= to=" + str(to_),
                    lambda: t.random_(from_, to_)
                )

    @dtypes(torch.int32, torch.int64, torch.float, torch.float16)
    def test_random_default(self, device, dtype):
        size = 2000
        alpha = 0.1

        # the dtype of 'to' is int, so its max value is the max of int64
        if dtype == torch.float:
            to_inc = torch.iinfo(torch.int64).max
        elif dtype == torch.float16:
            to_inc = 65504
        else:
            to_inc = torch.iinfo(dtype).max

        t = torch.empty(size, dtype=dtype, device="cpu").to("npu")
        t.random_()
        self.assertTrue(0 <= t.to(torch.double).min() < alpha * to_inc)
        self.assertTrue((to_inc - alpha * to_inc) < t.to(torch.double).max() <= to_inc)


instantiate_device_type_tests(TestRandom, globals(), except_for='cpu')
if __name__ == "__main__":
    run_tests()