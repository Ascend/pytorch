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

"""Test cases for torch.autograd.profiler_util.StringTable.clear.

StringTable.clear is inherited from dict and does not involve Tensor or device
behavior. These tests cover clearing an empty table, clearing populated entries,
and reusing the table after clear.
"""

from torch.autograd.profiler_util import StringTable
from torch.testing._internal.common_utils import run_tests, TestCase


class TestStringTableClear(TestCase):

    def test_clear_empty_table(self):
        table = StringTable()
        self.assertIsNone(table.clear())
        self.assertEqual(len(table), 0)

    def test_clear_after_access(self):
        table = StringTable()
        _ = table["aten::add"]
        _ = table["aten::mul"]
        self.assertGreater(len(table), 0)
        self.assertIsNone(table.clear())
        self.assertEqual(len(table), 0)

    def test_clear_then_reuse(self):
        table = StringTable()
        _ = table["aten::add"]
        self.assertIsNone(table.clear())
        self.assertEqual(len(table), 0)
        val = table["aten::sub"]
        self.assertIsInstance(val, str)
        self.assertEqual(len(table), 1)

if __name__ == "__main__":
    run_tests()
