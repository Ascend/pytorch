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

"""
Add validation cases for torch.distributed.GradBucket APIs on NPU:
1. PyTorch community lacks sufficient direct validation for GradBucket.index.
2. This file validates torch.distributed.GradBucket.index (extendable).
"""

import os
from datetime import timedelta

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.algorithms.ddp_comm_hooks import default_hooks
from torch.testing._internal.common_utils import TestCase, find_free_port, run_tests


device_type = acc.type if (acc := torch.accelerator.current_accelerator()) else "cpu"

PROCESS_GROUP_TIMEOUT = timedelta(minutes=2)


class _MultiBucketModel(nn.Module):
    def __init__(self):
        # Build a model with multiple parameters to generate multiple DDP buckets.
        super().__init__()
        self.layers = nn.ModuleList(
            [nn.Linear(32, 32, bias=False) for _ in range(4)]
        )

    def forward(self, input_tensor):
        # Run all linear layers and sum their outputs.
        return sum(layer(input_tensor) for layer in self.layers)


def _record_bucket_index_hook(
    state, bucket: dist.GradBucket
) -> torch.futures.Future[torch.Tensor]:
    # Record the index of each real DDP gradient bucket.
    state["bucket_indices"].append(bucket.index())
    return default_hooks.allreduce_hook(state["process_group"], bucket)


class TestGradBucket(TestCase):
    @staticmethod
    def _init_process_group(port):
        # Initialize a single-rank HCCL process group for DDP testing.
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = str(port)

        torch.accelerator.set_device_index(0)

        dist.init_process_group(
            "hccl",
            rank=0,
            world_size=1,
            timeout=PROCESS_GROUP_TIMEOUT,
        )

        return dist.group.WORLD

    def setUp(self):
        # Initialize HCCL and register automatic cleanup after each test.
        super().setUp()
        self.process_group = self._init_process_group(find_free_port())
        self.addCleanup(dist.destroy_process_group)

    def test_grad_bucket_index(self):
        # Validate GradBucket.index with multiple real DDP gradient buckets.
        process_group = self.process_group

        model = _MultiBucketModel().to(device_type)

        model = nn.parallel.DistributedDataParallel(
            model,
            device_ids=[0],
            process_group=process_group,
            bucket_cap_mb=0.001,
        )

        state = {
            "process_group": process_group,
            "bucket_indices": [],
        }

        model.register_comm_hook(state, _record_bucket_index_hook)

        input_tensor = torch.full(
            (8, 32),
            1.0,
            device=device_type,
        )

        model(input_tensor).sum().backward()

        model.zero_grad(set_to_none=True)
        state["bucket_indices"].clear()

        model(input_tensor).sum().backward()

        bucket_indices = state["bucket_indices"]

        self.assertGreater(len(bucket_indices), 1)

        self.assertTrue(
            all(isinstance(index, int) for index in bucket_indices)
        )

        self.assertEqual(
            len(bucket_indices),
            len(set(bucket_indices)),
        )

        self.assertEqual(
            sorted(bucket_indices),
            list(range(len(bucket_indices))),
        )


if __name__ == "__main__":
    run_tests()
