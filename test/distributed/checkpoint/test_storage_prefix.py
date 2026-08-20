# Copyright (c) 2026 Huawei Technologies Co., Ltd
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its contributors
#    may be used to endorse or promote products derived from this software
#    without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

"""Test cases for torch.distributed.checkpoint.filesystem._StoragePrefix.

This module tests the _StoragePrefix dataclass which is a simple container
class holding a prefix string used in distributed checkpoint filesystem storage.

Test coverage:
- Basic construction with prefix string
- Attribute access and modification
- Equality comparison between instances
- Default and edge case prefix values
- Usage in serialization/deserialization context
- Dataclass features (repr, fields)
"""

import tempfile
from dataclasses import asdict, fields

from torch.distributed.checkpoint.filesystem import (
    FileSystemWriter,
    _StoragePrefix,
)
from torch.distributed.checkpoint.planner import SavePlan
from torch.testing._internal.common_utils import TestCase, run_tests


class TestStoragePrefix(TestCase):
    """Test cases for _StoragePrefix dataclass."""

    def test_construction_basic(self):
        """_StoragePrefix can be constructed with a basic prefix string."""
        sp = _StoragePrefix(prefix="checkpoint_")
        self.assertEqual(sp.prefix, "checkpoint_")

    def test_construction_empty_string(self):
        """_StoragePrefix can be constructed with an empty prefix."""
        sp = _StoragePrefix(prefix="")
        self.assertEqual(sp.prefix, "")

    def test_construction_none(self):
        """_StoragePrefix accepts and preserves None without validation."""
        sp = _StoragePrefix(prefix=None)
        self.assertIsNone(sp.prefix)
        self.assertEqual(asdict(sp), {"prefix": None})

    def test_construction_long_path(self):
        """_StoragePrefix can be constructed with a long path prefix."""
        long_prefix = "models/experiment_2024/run_001/checkpoints/epoch_100/"
        sp = _StoragePrefix(prefix=long_prefix)
        self.assertEqual(sp.prefix, long_prefix)

    def test_construction_with_special_chars(self):
        """_StoragePrefix handles special characters in prefix."""
        special = "dir-with.dots_and/slashes"
        sp = _StoragePrefix(prefix=special)
        self.assertEqual(sp.prefix, special)

    def test_attribute_access(self):
        """_StoragePrefix prefix attribute is accessible and modifiable."""
        sp = _StoragePrefix(prefix="initial")
        self.assertEqual(sp.prefix, "initial")
        sp.prefix = "modified"
        self.assertEqual(sp.prefix, "modified")

    def test_equality_same_prefix(self):
        """Two _StoragePrefix instances with same prefix are equal."""
        sp1 = _StoragePrefix(prefix="same_prefix")
        sp2 = _StoragePrefix(prefix="same_prefix")
        self.assertEqual(sp1, sp2)

    def test_equality_different_prefix(self):
        """Two _StoragePrefix instances with different prefixes are not equal."""
        sp1 = _StoragePrefix(prefix="prefix_a")
        sp2 = _StoragePrefix(prefix="prefix_b")
        self.assertNotEqual(sp1, sp2)

    def test_equality_with_non_storage_prefix(self):
        """_StoragePrefix is not equal to non-_StoragePrefix objects."""
        sp = _StoragePrefix(prefix="test")
        self.assertNotEqual(sp, "test")
        self.assertNotEqual(sp, 42)

    def test_repr(self):
        """_StoragePrefix has a valid repr string."""
        sp = _StoragePrefix(prefix="my_prefix")
        repr_str = repr(sp)
        self.assertIn("_StoragePrefix", repr_str)
        self.assertIn("my_prefix", repr_str)

    def test_dataclass_fields(self):
        """_StoragePrefix has exactly one field named 'prefix'."""
        field_names = [f.name for f in fields(_StoragePrefix)]
        self.assertEqual(field_names, ["prefix"])

    def test_asdict(self):
        """_StoragePrefix can be converted to dict via asdict."""
        sp = _StoragePrefix(prefix="dict_test")
        d = asdict(sp)
        self.assertEqual(d, {"prefix": "dict_test"})

    def test_usage_in_checkpoint_plan(self):
        """FileSystemWriter stores _StoragePrefix in real SavePlan objects."""
        with tempfile.TemporaryDirectory() as checkpoint_dir:
            writer = FileSystemWriter(checkpoint_dir)
            plans = writer.prepare_global_plan(
                [SavePlan(items=[]), SavePlan(items=[])]
            )

        self.assertEqual(len(plans), 2)
        for rank, plan in enumerate(plans):
            self.assertIsInstance(plan, SavePlan)
            self.assertIsInstance(plan.storage_data, _StoragePrefix)
            self.assertEqual(plan.storage_data.prefix, f"__{rank}_")

    def test_unicode_prefix(self):
        """_StoragePrefix handles unicode characters in prefix."""
        sp = _StoragePrefix(prefix="检查点_前缀/")
        self.assertEqual(sp.prefix, "检查点_前缀/")

    def test_numeric_prefix(self):
        """_StoragePrefix handles numeric string prefix."""
        sp = _StoragePrefix(prefix="123456")
        self.assertEqual(sp.prefix, "123456")

    def test_default_factory_not_used(self):
        """_StoragePrefix requires prefix argument (no default)."""
        # prefix is a required field, should raise TypeError if not provided
        with self.assertRaises(TypeError):
            _StoragePrefix()

if __name__ == "__main__":
    run_tests()
