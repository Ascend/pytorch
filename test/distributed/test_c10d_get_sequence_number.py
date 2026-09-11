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

"""Validate ``ProcessGroup._get_sequence_number_for_group`` on NPU.

On torch < 2.14 the upstream implementation only accepts backends in the
backendSupportsSequenceNumbers whitelist and rejects CUSTOM (HCCL), so
``torch_npu._compat.distributed`` patches the method with an HCCL-aware
shim that dispatches to the NPU backend and falls back to the captured
upstream implementation for other backends. On torch >= 2.14
(pytorch#190138) the upstream implementation covers HCCL natively and
the shim is not installed. See the upstream test
``pytorch/test/distributed/test_c10d_nccl.py::ProcessGroupNCCLTest::test_sequence_number_initialized``.
"""

import os
import sys
import tempfile

import torch
import torch.distributed as dist
import torch.distributed.run as launch
from torch.distributed import distributed_c10d as c10d

from torch_npu.testing.common_distributed import skipIfUnsupportMultiNPU
from torch_npu.testing.testcase import TestCase, run_tests

_WORKER_ARG = "--sequence-number-worker"


def _worker_main():
    """Runs on 2 ranks with the hccl backend.

    Mirrors the semantics of the upstream test
    ``pytorch/test/distributed/test_c10d_nccl.py::ProcessGroupNCCLTest::test_sequence_number_initialized``:
    the sequence number of a process group must be queryable and increase
    as collectives complete. On torch < 2.14 without the torch_npu shim
    the query on an HCCL group raises, so running at all already proves
    the dispatch works.
    """
    dist.init_process_group(backend="hccl")
    rank = dist.get_rank()
    torch.npu.set_device(rank)

    pg = c10d._get_default_group()

    seq0 = pg._get_sequence_number_for_group()

    tensor = torch.ones(1).npu()
    dist.all_reduce(tensor)

    seq1 = pg._get_sequence_number_for_group()

    if not isinstance(seq0, int) or not isinstance(seq1, int):
        raise AssertionError(
            f"expected int sequence numbers, got {seq0!r} and {seq1!r}")
    if seq1 <= seq0:
        raise AssertionError(
            f"sequence number did not increase: {seq0} -> {seq1}")

    dist.destroy_process_group()
    print(f"[rank{rank}] sequence number OK: {seq0} -> {seq1}", flush=True)


class TestGetSequenceNumber(TestCase):
    def setUp(self):
        super().setUp()
        self.device_name = torch._C._get_privateuse1_backend_name()
        self.assertEqual(self.device_name, 'npu',
                         f"Expected device 'npu', got '{self.device_name}'")

    @skipIfUnsupportMultiNPU(2)
    def test_get_sequence_number_for_group_hccl(self):
        """The sequence number is queryable and increases on the hccl backend."""
        launch.main(
            [
                "--nproc-per-node=2",
                os.path.abspath(__file__),
                _WORKER_ARG,
            ]
        )

    def test_get_sequence_number_for_group_gloo_fallback(self):
        """Non-HCCL backends dispatch through the captured origin
        (torch < 2.14) or the native implementation (torch >= 2.14)."""
        store = dist.FileStore(os.path.join(tempfile.mkdtemp(), "store"), 1)
        dist.init_process_group("gloo", rank=0, world_size=1, store=store)
        try:
            seq = c10d._get_default_group()._get_sequence_number_for_group()
            self.assertIsInstance(seq, int)
        finally:
            dist.destroy_process_group()


if __name__ == "__main__":
    if _WORKER_ARG in sys.argv:
        _worker_main()
    else:
        run_tests()
