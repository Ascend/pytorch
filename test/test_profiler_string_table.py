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
2. This file validates torch.autograd.profiler_util.StringTable.popitem and
   torch.autograd.profiler_util.StringTable.default_factory (extendable).
"""

from torch.autograd.profiler_util import StringTable
from torch.testing._internal.common_utils import TestCase, run_tests


class TestStringTable(TestCase):
    """Test cases for torch.autograd.profiler_util.StringTable."""

    def _populate_table(self, items):
        """Helper: populate StringTable with given key-value pairs."""
        st = StringTable()
        for k, v in items:
            st[k] = v
        return st

    def test_popitem_returns_key_value(self):
        st = self._populate_table([("a", "alpha"), ("b", "beta")])
        initial_len = len(st)

        k, v = st.popitem()

        self.assertIsInstance(k, str)
        self.assertIsInstance(v, str)
        self.assertEqual(len(st), initial_len - 1)
        self.assertNotIn(k, st)

    def test_popitem_removes_item(self):
        st = self._populate_table([("key1", "val1"), ("key2", "val2")])

        k, _ = st.popitem()

        self.assertNotIn(k, st)

    def test_popitem_empty_raises_keyerror(self):
        st = StringTable()

        with self.assertRaises(KeyError):
            st.popitem()

    def test_popitem_exhausts_table(self):
        items = [("a", "1"), ("b", "2"), ("c", "3")]
        st = self._populate_table(items)

        popped = []
        while st:
            popped.append(st.popitem())

        self.assertEqual(len(st), 0)
        self.assertEqual(len(popped), len(items))
        self.assertRaises(KeyError, st.popitem)

    def test_popitem_with_missing_demangle(self):
        # Use a mangled C++ name that triggers __missing__ -> torch._C._demangle
        st = StringTable()
        mangled_key = "_Z3foov"  # demangles to "foo()" when demangle is available
        _ = st[mangled_key]  # triggers __missing__, stores demangled value

        self.assertGreater(len(st), 0)

        k, v = st.popitem()
        self.assertEqual(k, mangled_key)
        self.assertIsInstance(v, str)
        # __missing__ stores the demangled value, which must differ from the
        # mangled key to prove the demangle path was actually exercised
        self.assertNotEqual(v, mangled_key,
                            "demangle should produce different output from mangled input")

    def test_popitem_does_not_trigger_default_factory(self):
        # defaultdict's popitem should NOT call default_factory
        call_count = [0]

        def counting_factory():
            call_count[0] += 1
            return "default"

        st = StringTable(counting_factory)
        st["x"] = "explicit"
        st.popitem()

        # popitem on a table with only explicit items should not trigger factory
        self.assertEqual(call_count[0], 0)

    def test_multiple_popitems_consistent(self):
        items = [("k1", "v1"), ("k2", "v2"), ("k3", "v3"), ("k4", "v4")]
        st = self._populate_table(items)
        popped_items = []

        for _ in range(len(items)):
            popped_items.append(st.popitem())

        self.assertEqual(len(st), 0)
        popped_keys = {k for k, v in popped_items}
        expected_keys = {k for k, v in items}
        self.assertEqual(popped_keys, expected_keys)

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
