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

"""Worker for test_c10d_add_ephemeral_timeout.py.

Runs on 2 ranks with the hccl backend. Mirrors the semantics of the
upstream test
``pytorch/test/distributed/test_c10d_nccl.py::ProcessGroupNCCLGroupTest::test_extend_nccl_pg_timeout``:
after the base timeout is set to 3s (backend level) and
``_add_ephemeral_timeout_for_all_pgs(10s)``, a collective issued while
the peer stalls must survive past the base timeout.

The API under test is patched into ``torch.distributed.distributed_c10d`` by
``torch_npu._compat.distributed``, which uses the upstream implementation
when it is backend-generic (NPU included; torch nightly >= 2026-08-04) and
torch_npu's own implementation otherwise. The base timeout is set on the
backend level (``backend._set_default_timeout``), as in the upstream test,
which exists across torch 2.13/2.14.

Note: the extension-reset semantic (the extension expires once the
first collective issued after the API call completes) is implemented in
the torch_npu C++ watchdog thread
(torch_npu/csrc/distributed/ProcessGroupHCCL.cpp ``Watchdog::runLoop``).
It is only exercised in the default non-blocking wait mode; with
``TORCH_HCCL_BLOCKING_WAIT=1`` the watchdog thread is not created. This
test runs in blocking wait mode because that is the only mode where
``work.wait()`` blocks and the op timeout is observable from python
(unlike NCCL, torch_npu's ``WorkHCCL`` does not expose ``work.timeout``).
The reset semantic is therefore covered by the upstream NCCL test only.
"""

import os
import time
from datetime import timedelta

import torch
import torch.distributed as dist
from torch.distributed import distributed_c10d as c10d

os.environ.setdefault("TORCH_HCCL_BLOCKING_WAIT", "1")

BASE_TIMEOUT_S = 3
EPHEMERAL_S = 10
STALL_S = 5


def main():
    dist.init_process_group(backend="hccl", timeout=timedelta(seconds=60))
    rank = dist.get_rank()
    torch.npu.set_device(rank)

    tensor = torch.ones(1).npu()
    dist.broadcast(tensor, 0)
    dist.all_reduce(tensor)  # warm up communicators

    backend = c10d._get_default_group()._get_backend(
        torch.device(torch._C._get_privateuse1_backend_name()))
    backend._set_default_timeout(timedelta(seconds=BASE_TIMEOUT_S))

    if rank == 1:
        _validate_extension(tensor)
    else:
        # Stall so the rank-1 collective blocks past the base timeout.
        time.sleep(STALL_S)
        dist.all_reduce(tensor)
    dist.destroy_process_group()


def _validate_extension(tensor):
    # The API returns None.
    result = c10d._add_ephemeral_timeout_for_all_pgs(timedelta(seconds=EPHEMERAL_S))
    if result is not None:
        raise AssertionError(f"expected None return, got {result!r}")
    # With the ephemeral extension the stalled collective must NOT fail
    # at the base timeout (3s), and must still complete well before the
    # extended timeout (13s).
    t0 = time.monotonic()
    dist.all_reduce(tensor)
    elapsed = time.monotonic() - t0
    print(f"[rank1] stalled allreduce survived {elapsed:.2f}s "
          f"(base {BASE_TIMEOUT_S}s + ephemeral {EPHEMERAL_S}s)", flush=True)
    if elapsed < BASE_TIMEOUT_S:
        raise RuntimeError("ephemeral extension was not applied: "
                           f"stalled collective completed after {elapsed:.2f}s")
    if elapsed >= BASE_TIMEOUT_S + EPHEMERAL_S:
        raise RuntimeError(f"unexpectedly long elapsed time: {elapsed:.2f}s")


if __name__ == "__main__":
    main()
