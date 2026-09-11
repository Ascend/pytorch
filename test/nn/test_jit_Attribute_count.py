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
Add validation cases for torch.jit.Attribute.count APIs on NPU:
1. PyTorch community lacks direct validation for torch.jit.Attribute.count,
which counts value occurrences in container attributes (e.g., tuple, list)
of TorchScript modules, so this file is added.
2. This file validates torch.jit.Attribute.count (extendable).
"""

from typing import List

import torch
from torch.testing._internal.common_utils import run_tests, TestCase


class TestJitAttributeCount(TestCase):
    """Test torch.jit.Attribute count functionality."""

    def test_jit_attribute_list_count(self):
        """Verify that .count() works correctly on a list attribute."""
        class AttributeModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.my_attrs: List[int] = torch.jit.Attribute(
                    [1, 2, 3, 2, 4, 2], List[int]
                )

            def forward(self, val: int) -> int:
                return self.my_attrs.count(val)

        scripted_module = torch.jit.script(AttributeModule())
        self.assertEqual(scripted_module(2), 3)
        self.assertEqual(scripted_module(5), 0)

    def test_jit_attribute_empty_list_count(self):
        """Test count on an empty list attribute."""
        class Module(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.attrs: List[int] = torch.jit.Attribute([], List[int])

            def forward(self, val: int) -> int:
                return self.attrs.count(val)

        scripted = torch.jit.script(Module())
        self.assertEqual(scripted(1), 0)
        self.assertEqual(scripted(0), 0)

    def test_jit_attribute_str_list_count(self):
        """Test count on a list of strings."""
        class Module(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.attrs: List[str] = torch.jit.Attribute(
                    ["a", "b", "a", "c"], List[str]
                )

            def forward(self, val: str) -> int:
                return self.attrs.count(val)

        scripted = torch.jit.script(Module())
        self.assertEqual(scripted("a"), 2)
        self.assertEqual(scripted("d"), 0)

    def test_jit_attribute_float_list_count(self):
        """Test count on a list of floats."""
        class Module(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.attrs: List[float] = torch.jit.Attribute(
                    [1.0, 2.5, 1.0], List[float]
                )

            def forward(self, val: float) -> int:
                return self.attrs.count(val)

        scripted = torch.jit.script(Module())
        self.assertEqual(scripted(1.0), 2)
        self.assertEqual(scripted(3.0), 0)

    def test_jit_attribute_incompatible_type(self):
        """
        Test incompatible type for count.
        In TorchScript, the argument type must match the list element type,
        otherwise compilation fails.
        """
        class Module(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.attrs: List[int] = torch.jit.Attribute([1, 2, 3], List[int])

            def forward(self, val: str) -> int:
                # This should raise a type error during scripting
                return self.attrs.count(val)

        with self.assertRaises(Exception):
            torch.jit.script(Module())


if __name__ == '__main__':
    run_tests()
