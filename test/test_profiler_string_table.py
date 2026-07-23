"""
Add validation cases for torch.autograd.profiler_util APIs on NPU:
1. PyTorch community lacks direct unit tests for torch.autograd.profiler_util.StringTable.
2. This file validates torch.autograd.profiler_util.StringTable.popitem (extendable).
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


if __name__ == "__main__":
    run_tests()
