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

"""Validate the upstream API
``torch.distributed.distributed_c10d._add_ephemeral_timeout_for_all_pgs``
on NPU.

torch_npu's ``ProcessGroupHCCL`` implements ``addEphemeralTimeout`` in C++
(torch_npu/csrc/distributed/ProcessGroupHCCL.cpp), so the upstream PyTorch
implementation of this API must work on NPU directly, without any torch_npu
side patch. See the upstream test
``pytorch/test/distributed/test_c10d_nccl.py::ProcessGroupNCCLGroupTest::test_extend_nccl_pg_timeout``.

The API under test is patched into ``torch.distributed.distributed_c10d`` by
``torch_npu._compat.distributed`` (loaded via
``torch_npu._init.patches.distributed_patches``): it uses the upstream
implementation on torch versions that support NPU (backend-generic, since
the 2026-08-04 nightly) and torch_npu's own implementation on older torch,
so the tests exercise the same callsite as user code.
"""

import os
import tempfile
from datetime import timedelta

import torch
import torch.distributed as dist
import torch.distributed.run as launch
from torch.distributed import distributed_c10d as c10d

from torch_npu.testing.common_distributed import skipIfUnsupportMultiNPU
from torch_npu.testing.testcase import TestCase, run_tests


def path(script):
    return os.path.join(os.path.dirname(__file__), script)


class TestAddEphemeralTimeout(TestCase):
    def setUp(self):
        super().setUp()
        self.device_name = torch._C._get_privateuse1_backend_name()
        self.assertEqual(self.device_name, 'npu',
                         f"Expected device 'npu', got '{self.device_name}'")

    @skipIfUnsupportMultiNPU(2)
    def test_add_ephemeral_timeout_for_all_pgs(self):
        """Ephemeral extension applies to collectives issued after the call."""
        launch.main(
            [
                "--nproc-per-node=2",
                path("ephemeral_timeout/ephemeral_timeout_base.py"),
            ]
        )

    def test_add_ephemeral_timeout_ignored_on_unsupported_backend(self):
        """Backends without ephemeral support ignore the call and return None."""
        store = dist.FileStore(os.path.join(tempfile.mkdtemp(), "store"), 1)
        dist.init_process_group("gloo", rank=0, world_size=1, store=store)
        try:
            result = c10d._add_ephemeral_timeout_for_all_pgs(timedelta(seconds=10))
            self.assertIsNone(result)
        finally:
            dist.destroy_process_group()


if __name__ == "__main__":
    run_tests()
