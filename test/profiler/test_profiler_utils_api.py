# Copyright (c) 2026 Huawei Technologies Co., Ltd
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

"""
Add validation cases for torch.autograd profiler APIs on NPU:
1. PyTorch community lacks sufficient and direct API validations for some APIs, so this file is added.
2. This file validates torch.autograd.profiler.EnforceUnique,
   torch.autograd.profiler_util.MemRecordsAcc.in_interval (extendable).
"""

from torch.autograd import profiler
from torch.autograd import profiler_util
from torch.testing._internal.common_utils import TestCase, run_tests


class TestProfilerUtilsApi(TestCase):

    def test_enforce_unique_raises_on_duplicate_key(self):
        guard = profiler.EnforceUnique()

        guard.see("scope", 1)

        with self.assertRaisesRegex(RuntimeError, "duplicate key"):
            guard.see("scope", 1)

    def test_mem_records_acc_filters_records_in_interval(self):
        class Event:
            def __init__(self, start_ns):
                self._start_ns = start_ns

            def start_ns(self):
                return self._start_ns

        records = [
            (Event(1000), "before"),
            (Event(2500), "inside"),
            (Event(5000), "boundary"),
            (Event(7000), "after"),
        ]

        actual = [record[1] for record in profiler_util.MemRecordsAcc(records).in_interval(2000, 5000)]

        self.assertEqual(actual, ["inside", "boundary"])


if __name__ == "__main__":
    run_tests()
