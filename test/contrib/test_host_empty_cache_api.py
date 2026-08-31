# Copyright (c) Huawei Technologies Co., Ltd. 2020-2024. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
torch._C._host_emptyCache API consistency test.

torch._C._host_emptyCache is an NPU-specific mirror created by
torch_npu.contrib.transfer_to_npu, binding torch._C._host_emptyCache to the
C++ implementation torch_npu._C._npu_hostEmptyCache. This file verifies both
the mapping and its callable behavior so the API stays consistent across
torch-npu releases. Additional torch._C._* consistency checks of the same kind
can be appended here.
"""

import torch
from torch.testing._internal.common_utils import run_tests, TestCase
import torch_npu.contrib.transfer_to_npu  # importing this module injects torch._C._host_emptyCache


class TestHostEmptyCacheApi(TestCase):

    def test_host_empty_cache_is_mapped(self):
        # transfer_to_npu injects torch._C._host_emptyCache as a side effect of
        # its module import; the binding exists as soon as that import completes.
        if not hasattr(torch._C, "_host_emptyCache"):
            self.skipTest("torch._C._host_emptyCache not mapped in this torch-npu build")
        self.assertEqual(torch._C._host_emptyCache, torch_npu._C._npu_hostEmptyCache)

    def test_host_empty_cache_is_callable(self):
        # The API is invocable without arguments and returns None; it mirrors the
        # host-cache path of torch_npu.npu.empty_cache(). Illegal arguments must
        # be rejected with TypeError.
        if not hasattr(torch._C, "_host_emptyCache"):
            self.skipTest("torch._C._host_emptyCache not mapped in this torch-npu build")
        # No arguments: callable and returns None.
        self.assertIsNone(torch._C._host_emptyCache())
        # Illegal positional / keyword arguments must be rejected.
        with self.assertRaises(TypeError):
            torch._C._host_emptyCache(0)
        with self.assertRaises(TypeError):
            torch._C._host_emptyCache(device="npu")


if __name__ == "__main__":
    run_tests()
