"""
Add validation cases for torch.autograd.profiler_util.StringTable on NPU:
1. PyTorch community lacks direct API validations for StringTable.pop.
2. This file validates StringTable.pop (extendable).
"""

import torch
from torch.testing._internal.common_utils import run_tests, TestCase


class TestStringTable(TestCase):

    def test_pop_existing_key(self):
        """Validate StringTable.pop returns value and removes key when key exists."""
        from torch.autograd.profiler_util import StringTable

        st = StringTable()
        st["key1"] = "value1"
        result = st.pop("key1")
        self.assertEqual(result, "value1")
        self.assertNotIn("key1", st)

    def test_pop_with_default(self):
        """Validate StringTable.pop returns default value when key does not exist."""
        from torch.autograd.profiler_util import StringTable

        st = StringTable()
        st["key1"] = "value1"
        result = st.pop("key2", "default")
        self.assertEqual(result, "default")
        self.assertNotIn("key2", st)

    def test_pop_existing_key_with_default(self):
        """Validate StringTable.pop returns actual value even when default is provided."""
        from torch.autograd.profiler_util import StringTable

        st = StringTable()
        st["key1"] = "value1"
        result = st.pop("key1", "default")
        self.assertEqual(result, "value1")
        self.assertNotIn("key1", st)

    def test_pop_missing_key_no_default(self):
        """Validate StringTable.pop raises KeyError when key is missing and no default."""
        from torch.autograd.profiler_util import StringTable

        st = StringTable()
        with self.assertRaises(KeyError):
            st.pop("nonexistent")


if __name__ == "__main__":
    run_tests()
