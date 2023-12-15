# Copyright (c) 2022 Huawei Technologies Co., Ltd
# Copyright (c) 2022, Facebook CORPORATION.
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

import itertools
import torch
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.decorator import Dtypes, instantiate_tests


@instantiate_tests
class TestRandom(TestCase):

    @Dtypes(torch.int32, torch.int64, torch.float, torch.float16)
    def test_random_from_to(self, dtype):
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

        iter_list = itertools.product(froms, tos)
        for from_, to_ in iter_list:
            t = torch.empty(size, dtype=dtype, device="cpu").to("npu")
            if to_ <= from_:
                self.assertRaisesRegex(
                    RuntimeError,
                    "random_ expects 'from' to be less than 'to', but got from="
                    + str(from_) + " >= to=" + str(to_),
                    lambda: t.random_(from_, to_)
                )
                continue

            if not (min_val <= from_ <= max_val) or not (min_val <= (to_ - 1) <= max_val):
                if not (min_val <= from_ <= max_val):
                    self.assertRaisesRegex(
                        RuntimeError,
                        "from is out of bounds",
                        lambda: t.random_(from_, to_)
                    )
                if not (min_val <= (to_ - 1) <= max_val):
                    self.assertRaisesRegex(
                        RuntimeError,
                        "to - 1 is out of bounds",
                        lambda: t.random_(from_, to_)
                    )
            else:
                t.random_(from_, to_)
                range_ = to_ - from_
                delta = max(1, alpha * range_)
                self.assertTrue(from_ <= t.to(torch.double).min() < (from_ + delta))
                self.assertTrue((to_ - delta) <= t.to(torch.double).max() < to_)

    @Dtypes(torch.int32, torch.int64, torch.float, torch.float16)
    def test_random_to(self, dtype):
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
                    self.assertRaisesRegex(
                        RuntimeError,
                        "to - 1 is out of bounds",
                        lambda: t.random_(to_)
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

    @Dtypes(torch.int32, torch.int64, torch.float, torch.float16, torch.uint8, torch.int16, torch.int8, torch.double)
    def test_random_default(self, dtype):
        size = 2000
        alpha = 0.1

        if dtype == torch.float:
            to_inc = 1 << 24
        elif dtype == torch.double:
            to_inc = 1 << 53
        elif dtype == torch.half:
            to_inc = 1 << 11
        else:
            to_inc = torch.iinfo(dtype).max

        t = torch.empty(size, dtype=dtype, device="cpu").to("npu")
        t.random_()
        self.assertTrue(0 <= t.to(torch.double).min() < alpha * to_inc)
        self.assertTrue((to_inc - alpha * to_inc) < t.to(torch.double).max() <= to_inc)

    @Dtypes(torch.int32, torch.int64, torch.float, torch.float16)
    def test_random_diffent_size(self, dtype):
        from_ = -800
        to_ = 800

        size = [10]
        t = torch.empty(size, dtype=dtype, device="cpu").to("npu")
        t.random_(from_, to_)
        self.assertTrue(from_ <= t.to(torch.double).min() <= to_)
        self.assertTrue(from_ <= t.to(torch.double).max() <= to_)

        size = [10, 8]
        t = torch.empty(size, dtype=dtype, device="cpu").to("npu")
        t.random_(from_, to_)
        self.assertTrue(from_ <= t.to(torch.double).min() <= to_)
        self.assertTrue(from_ <= t.to(torch.double).max() <= to_)

        size = [10, 8, 7]
        t = torch.empty(size, dtype=dtype, device="cpu").to("npu")
        t.random_(from_, to_)
        self.assertTrue(from_ <= t.to(torch.double).min() <= to_)
        self.assertTrue(from_ <= t.to(torch.double).max() <= to_)

        size = [10, 8, 7, 5]
        t = torch.empty(size, dtype=dtype, device="cpu").to("npu")
        t.random_(from_, to_)
        self.assertTrue(from_ <= t.to(torch.double).min() <= to_)
        self.assertTrue(from_ <= t.to(torch.double).max() <= to_)

        size = [10, 8, 7, 5, 2]
        t = torch.empty(size, dtype=dtype, device="cpu").to("npu")
        t.random_(from_, to_)
        self.assertTrue(from_ <= t.to(torch.double).min() <= to_)
        self.assertTrue(from_ <= t.to(torch.double).max() <= to_)

    def test_random_seed(self):
        torch.manual_seed(123)
        input1 = torch.rand(2, 3, 4).npu()
        input1.random_(2, 10)
        torch.manual_seed(123)
        input2 = torch.rand(2, 3, 4).npu()
        input2.random_(2, 10)
        self.assertRtolEqual(input1.cpu(), input2.cpu())

    def test_random_seed_fp16(self):
        torch.manual_seed(3)
        input1 = torch.rand(4, 5, 3).half().npu()
        input1.random_(0, 100)
        torch.manual_seed(3)
        input2 = torch.rand(4, 5, 3).half().npu()
        input2.random_(0, 100)
        self.assertRtolEqual(input1.cpu(), input2.cpu())


if __name__ == "__main__":
    run_tests()
