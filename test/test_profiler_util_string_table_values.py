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
Add validation cases for torch.autograd.profiler_util.StringTable.values on NPU.

This file validates that StringTable.values returns a dynamic ValuesView
containing current StringTable values in both empty and populated cases.
"""

from collections.abc import ValuesView

import torch
import torch_npu
from torch.autograd.profiler_util import StringTable
from torch.testing._internal.common_utils import TestCase, run_tests


class TestStringTableValues(TestCase):
    """Functional tests for torch.autograd.profiler_util.StringTable.values."""

    def test_values_empty_string_table(self):
        string_table = StringTable()

        values = string_table.values()

        self.assertIsInstance(values, ValuesView)
        self.assertEqual(list(values), [])

    def test_values_contains_explicit_items(self):
        string_table = StringTable()
        string_table["aten::add"] = "aten::add"
        string_table["aten::relu"] = "aten::relu"

        values = string_table.values()

        self.assertEqual(set(values), {"aten::add", "aten::relu"})
        self.assertEqual(len(values), 2)

    def test_values_reflects_missing_key_insertion(self):
        string_table = StringTable()

        demangled_name = string_table["std::vector<int>"]
        values = string_table.values()

        self.assertIn(demangled_name, values)
        self.assertEqual(list(values), [demangled_name])

    def test_values_keeps_short_key_unchanged(self):
        string_table = StringTable()

        short_name = string_table["t"]

        self.assertEqual(short_name, "t")
        self.assertEqual(list(string_table.values()), ["t"])

    def test_values_view_updates_after_mutation(self):
        string_table = StringTable()
        string_table["first"] = "first"
        values = string_table.values()

        string_table["second"] = "second"

        self.assertEqual(set(values), {"first", "second"})
        self.assertEqual(len(values), 2)


if __name__ == "__main__":
    print(f"torch version: {torch.__version__}", flush=True)
    print(f"torch_npu version: {torch_npu.__version__}", flush=True)
    run_tests()
