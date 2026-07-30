#!/usr/bin/env python3
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
# Owner(s): ["module: library"]

"""Add validation cases for torch._meta_registrations.register_meta on NPU:
1. PyTorch community lacks sufficient and direct API validations for
   torch._meta_registrations.register_meta, so this file is added.
2. This file validates the decorator's core registration behavior, including
   single-op registration, multi-op registration, and returning the original
   function (extendable).
"""

import torch
from torch.testing._internal.common_utils import run_tests, TestCase


class TestLibraryMetaRegistrations(TestCase):
    """Test torch._meta_registrations.register_meta."""

    def test_register_meta_adds_to_meta_table(self):
        """register_meta adds the decorated function to the meta table."""
        op = torch.ops.aten.add.Tensor
        meta_table = torch._meta_registrations.meta_table
        original_meta = meta_table.get(op)

        def meta_fn(self, other, alpha=1):
            return torch.empty_like(self)

        try:
            wrapped = torch._meta_registrations.register_meta(op)(meta_fn)
            self.assertIs(wrapped, meta_fn)
            self.assertIn(op, meta_table)
            self.assertIs(meta_table[op], meta_fn)
        finally:
            if original_meta is not None:
                meta_table[op] = original_meta
            elif op in meta_table:
                del meta_table[op]

    def test_register_meta_supports_op_list(self):
        """register_meta can register the same function for multiple ops."""
        ops = [torch.ops.aten.sub.Tensor, torch.ops.aten.mul.Tensor]
        meta_table = torch._meta_registrations.meta_table
        original_metas = [meta_table.get(op) for op in ops]

        def meta_fn(self, other, alpha=1):
            return torch.empty_like(self)

        try:
            wrapped = torch._meta_registrations.register_meta(ops)(meta_fn)
            self.assertIs(wrapped, meta_fn)
            for op in ops:
                self.assertIn(op, meta_table)
                self.assertIs(meta_table[op], meta_fn)
        finally:
            for op, original in zip(ops, original_metas):
                if original is not None:
                    meta_table[op] = original
                elif op in meta_table:
                    del meta_table[op]


if __name__ == "__main__":
    run_tests()
