# Copyright (c) 2026 Huawei Technologies Co., Ltd
# All rights reserved.
#
# Licensed under the BSD 3-Clause License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://opensource.org/licenses/BSD-3-Clause
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
# implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Add validation cases for torch.autograd.profiler_util.StringTable on NPU:
1. PyTorch community lacks sufficient and direct API validations for some
   torch.autograd.profiler_util.StringTable APIs.
2. This file validates torch.autograd.profiler_util.StringTable.pop and
   torch.autograd.profiler_util.StringTable.default_factory (extendable).
"""

from torch.autograd.profiler_util import StringTable
from torch.testing._internal.common_utils import TestCase, run_tests


class TestStringTable(TestCase):

    def test_pop_existing_key(self):
        """Validate StringTable.pop returns value and removes key when key exists."""
        st = StringTable()
        st["key1"] = "value1"
        result = st.pop("key1")
        self.assertEqual(result, "value1")
        self.assertNotIn("key1", st)

    def test_pop_with_default(self):
        """Validate StringTable.pop returns default value when key does not exist."""
        st = StringTable()
        st["key1"] = "value1"
        result = st.pop("key2", "default")
        self.assertEqual(result, "default")
        self.assertNotIn("key2", st)

    def test_pop_existing_key_with_default(self):
        """Validate StringTable.pop returns actual value even when default is provided."""
        st = StringTable()
        st["key1"] = "value1"
        result = st.pop("key1", "default")
        self.assertEqual(result, "value1")
        self.assertNotIn("key1", st)

    def test_pop_missing_key_no_default(self):
        """Validate StringTable.pop raises KeyError when key is missing and no default."""
        st = StringTable()
        with self.assertRaises(KeyError):
            st.pop("nonexistent")

    def test_default_factory_is_none_by_default(self):
        self.assertIsNone(StringTable().default_factory)
        self.assertIsNone(StringTable(None).default_factory)

    def test_default_factory_preserves_callable(self):
        def factory():
            return "default"

        table = StringTable(factory)

        self.assertIs(table.default_factory, factory)

    def test_default_factory_is_writable(self):
        def factory():
            return "default"

        table = StringTable()
        table.default_factory = factory
        self.assertIs(table.default_factory, factory)

        table.default_factory = None
        self.assertIsNone(table.default_factory)

    def test_default_factory_rejects_non_callable_at_construction(self):
        with self.assertRaises(TypeError):
            StringTable("invalid")

    def test_default_factory_is_not_used_by_string_table_missing(self):
        calls = []

        def factory():
            calls.append(True)
            return "default"

        table = StringTable(factory)

        self.assertEqual(table["t"], "t")
        self.assertEqual(calls, [])


if __name__ == "__main__":
    run_tests()
