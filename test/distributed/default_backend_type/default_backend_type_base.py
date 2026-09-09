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

"""Worker for test_c10d_default_backend_type.py.

Runs on 2 ranks. Initializes the default process group WITHOUT specifying a
backend, so it goes through the upstream ``Backend.UNDEFINED`` path: the
default backend type must resolve to ``BackendType.CUSTOM`` for the hccl
backend registered by torch_npu. ``ProcessGroup::barrier()`` derives its
tensor device from the default backend type, so a wrong resolution fails
loudly on the first ``dist.barrier()``.
"""

import torch
import torch.distributed as dist


def main():
    # backend=None -> "undefined" -> BackendConfig("undefined") -> npu:hccl.
    dist.init_process_group()
    rank = dist.get_rank()
    world = dist.get_world_size()
    torch.npu.set_device(rank)

    # barrier() uses the resolved default backend type (CUSTOM -> hccl).
    dist.barrier()

    tensor = torch.ones(1).npu()
    dist.all_reduce(tensor)
    if tensor.item() != world:
        raise RuntimeError(
            f"all_reduce mismatch on rank {rank}: {tensor.item()} != {world}"
        )

    # Subgroups without an explicit backend follow the same resolution path.
    subgroup = dist.new_group(ranks=list(range(world)))
    dist.barrier(group=subgroup)
    dist.all_reduce(tensor, group=subgroup)
    dist.destroy_process_group()
    print(f"[rank{rank}] default-backend hccl group OK (world_size={world})",
          flush=True)


if __name__ == "__main__":
    main()
