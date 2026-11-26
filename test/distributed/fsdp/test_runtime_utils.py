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
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Validate torch.distributed.fsdp._runtime_utils._post_backward_hook.

Upstream FSDP tests cover gradient reduction and accumulation but lack a direct
check of the rejection of gradients requiring gradients. Exercise that boundary
through real backward passes, including no_sync and original parameter views.
"""

from contextlib import ExitStack, nullcontext

import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy
from torch.testing._internal.common_distributed import MultiProcessTestCase
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
)


device_type = acc.type if (acc := torch.accelerator.current_accelerator()) else "cpu"


@instantiate_parametrized_tests
class TestPostBackwardHook(MultiProcessTestCase):
    @property
    def world_size(self):
        return 2

    def setUp(self):
        if device_type != "cpu":
            device_count = torch.accelerator.device_count()
            if device_count < self.world_size:
                self.skipTest(
                    f"FSDP requires {self.world_size} {device_type} devices; found {device_count}"
                )
        super().setUp()
        self._spawn_processes()

    @parametrize("strategy", [
        ShardingStrategy.FULL_SHARD,
        ShardingStrategy.SHARD_GRAD_OP,
    ])
    @parametrize("use_orig_params", [False, True])
    @parametrize("sync_gradients", [False, True])
    def test_reject_gradients_of_gradients(self, strategy, use_orig_params, sync_gradients):
        device = torch.device(device_type, self.rank) if device_type != "cpu" else torch.device("cpu")
        if device_type != "cpu":
            torch.accelerator.set_device_index(self.rank)
        with ExitStack() as cleanup:
            dist.init_process_group(
                backend=dist.get_default_backend_for_device(device),
                init_method=f"file://{self.file_name}",
                rank=self.rank,
                world_size=self.world_size,
            )
            cleanup.callback(dist.destroy_process_group)
            model = FSDP(
                torch.nn.Linear(4, 4, bias=False).to(device),
                device_id=device,
                sharding_strategy=strategy,
                use_orig_params=use_orig_params,
            )
            context = nullcontext() if sync_gradients else model.no_sync()
            with context:
                loss = model(torch.ones(2, 4, device=device)).square().sum()
                with self.assertRaisesRegex(RuntimeError, "FSDP does not support gradients of gradients"):
                    loss.backward(create_graph=True)


if __name__ == "__main__":
    run_tests()
