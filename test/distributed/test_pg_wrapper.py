# Owner(s): ["oncall: distributed"]


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
Add validation cases for torch.distributed APIs on NPU:
1. The community test file test/distributed/test_pg_wrapper.py relies on nccl/xccl
   specific decorators and is not carried by this repository, so it cannot validate
   the NPU backend directly.
2. This file validates
   torch.distributed.distributed_c10d._create_process_group_wrapper (extendable)
   on NPU at the API level. No collective communication is started, so the cases
   cannot deadlock on HCCL.
"""

import torch.distributed as c10d
from torch.testing._internal.common_utils import (
    TestCase,
    run_tests,
)


class TestCreatePGWrapperOnNPU(TestCase):
    """
    NPU specific tests for _create_process_group_wrapper. The factory is
    checked at the API level so that invalid arguments are rejected before
    any process group is created. These cases do not start collectives and
    can therefore run in the main test process.
    """

    def test_create_process_group_wrapper_callable(self):
        self.assertTrue(callable(c10d._create_process_group_wrapper))

    def test_create_process_group_wrapper_no_args(self):
        with self.assertRaises(TypeError):
            c10d._create_process_group_wrapper()

    def test_create_process_group_wrapper_insufficient_args(self):
        with self.assertRaises(TypeError):
            c10d._create_process_group_wrapper(None)

    def test_create_process_group_wrapper_rank_type_error(self):
        with self.assertRaises((TypeError, ValueError)):
            c10d._create_process_group_wrapper(None, "prefix", None, "zero", 1)

    def test_create_process_group_wrapper_world_size_type_error(self):
        with self.assertRaises((TypeError, ValueError)):
            c10d._create_process_group_wrapper(None, "prefix", None, 0, "one")


if __name__ == "__main__":
    run_tests()
