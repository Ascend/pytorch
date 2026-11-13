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
Add direct validation for torch.distributed.GradBucket.parameters.

PyTorch community communication-hook tests use GradBucket extensively, but do
not directly validate that parameters() exposes every model parameter in a
real DDP bucket on the active accelerator.
"""

import os
import unittest
from datetime import timedelta

import torch
import torch.distributed as dist
from torch import nn
from torch.testing._internal.common_utils import (
    IS_WINDOWS,
    TestCase,
    find_free_port,
    run_tests,
)


device_type = acc.type if (acc := torch.accelerator.current_accelerator()) else "cpu"
backend = "hccl" if device_type == "npu" else "gloo"
PROCESS_GROUP_TIMEOUT = timedelta(minutes=2)


class _TwoLayerModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.first = nn.Linear(4, 3)
        self.second = nn.Linear(3, 2)

    def forward(self, input_tensor):
        return self.second(self.first(input_tensor))


def _capture_parameters_hook(state, bucket):
    state["parameters"] = bucket.parameters()
    state["gradients"] = bucket.gradients()
    future = torch.futures.Future()
    future.set_result(bucket.buffer())
    return future


class TestGradBucketParameters(TestCase):
    @unittest.skipIf(
        IS_WINDOWS and backend == "gloo",
        "The Windows Gloo backend cannot create the local test device",
    )
    def test_parameters_match_model_parameters(self):
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = str(find_free_port())
        if device_type != "cpu":
            torch.accelerator.set_device_index(0)

        dist.init_process_group(
            backend,
            rank=0,
            world_size=1,
            timeout=PROCESS_GROUP_TIMEOUT,
        )
        try:
            model = _TwoLayerModel().to(device_type)
            ddp_model = nn.parallel.DistributedDataParallel(
                model,
                device_ids=[0] if device_type != "cpu" else None,
                bucket_cap_mb=25,
            )
            state = {}
            ddp_model.register_comm_hook(state, _capture_parameters_hook)

            input_tensor = torch.randn(2, 4, device=device_type)
            ddp_model(input_tensor).sum().backward()

            bucket_parameters = state["parameters"]
            bucket_gradients = state["gradients"]
            model_parameters = list(model.parameters())
            self.assertIsInstance(bucket_parameters, list)
            self.assertIsInstance(bucket_gradients, list)
            self.assertEqual(len(bucket_parameters), len(model_parameters))
            self.assertEqual(len(bucket_parameters), len(bucket_gradients))
            self.assertEqual(
                [parameter.data_ptr() for parameter in bucket_parameters],
                [parameter.data_ptr() for parameter in model_parameters],
            )
            for parameter, gradient in zip(bucket_parameters, bucket_gradients):
                self.assertEqual(parameter.shape, gradient.shape)
                self.assertEqual(parameter.dtype, gradient.dtype)
                self.assertEqual(parameter.device, gradient.device)
                self.assertEqual(parameter.device.type, device_type)
                self.assertIsNotNone(parameter.grad)
                self.assertEqual(parameter.grad, gradient)
        finally:
            dist.destroy_process_group()


if __name__ == "__main__":
    run_tests()
