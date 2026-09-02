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
Add validation cases for
torch.distributed.algorithms.ddp_comm_hooks.default_hooks APIs on NPU:
1. PyTorch community lacks sufficient direct validations for some default DDP
   communication hooks.
2. This file validates
   torch.distributed.algorithms.ddp_comm_hooks.default_hooks.allreduce_hook
   (extendable).
"""

import copy
import os
from datetime import timedelta

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn
from torch.distributed.algorithms.ddp_comm_hooks import (
    DDPCommHookType,
    default_hooks,
    register_ddp_comm_hook,
)
from torch.testing._internal.common_utils import TestCase, find_free_port, run_tests
from torch_npu.testing.common_distributed import skipIfUnsupportMultiNPU


device_type = acc.type if (acc := torch.accelerator.current_accelerator()) else "cpu"

WORLD_SIZE = 2
PROCESS_GROUP_TIMEOUT = timedelta(minutes=2)


class _AllreduceModel(nn.Module):
    def __init__(self, shape=(40, 20), dtype=torch.float32):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(shape, dtype=dtype).to(device_type))

    def forward(self, input_tensor):
        return self.weight * input_tensor


class _MultiBucketModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(32, 32, bias=False) for _ in range(4)])

    def forward(self, input_tensor):
        return sum(layer(input_tensor) for layer in self.layers)


class _Bucket:
    def __init__(self, tensor):
        self.tensor = tensor

    def index(self):
        return 0

    def buffer(self):
        return self.tensor

    def gradients(self):
        return [self.tensor]

    def parameters(self):
        return []

    def is_last(self):
        return True

    def set_buffer(self, tensor):
        self.tensor = tensor


def _future_contract_hook(
    state, bucket: dist.GradBucket
) -> torch.futures.Future[torch.Tensor]:
    buffer = bucket.buffer()
    future = default_hooks.allreduce_hook(state["process_group"], bucket)
    state["is_future"] = isinstance(future, torch._C.Future)
    state["buffer_shape"] = buffer.shape

    def validate(fut):
        result = fut.value()
        state["result_shape"] = result.shape
        state["result_dtype"] = result.dtype
        state["result_device_type"] = result.device.type
        return result

    return future.then(validate)


def _counting_hook(
    state, bucket: dist.GradBucket
) -> torch.futures.Future[torch.Tensor]:
    state["calls"] += 1
    return default_hooks.allreduce_hook(state["process_group"], bucket)


class TestAllreduceHook(TestCase):
    @staticmethod
    def _init_process_group(rank, world_size, port):
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = str(port)
        torch.accelerator.set_device_index(rank)
        dist.init_process_group(
            "hccl",
            rank=rank,
            world_size=world_size,
            timeout=PROCESS_GROUP_TIMEOUT,
        )
        return dist.group.WORLD

    @staticmethod
    def _ddp(
        model,
        rank,
        process_group,
        gradient_as_bucket_view=False,
        static_graph=False,
        bucket_cap_mb=None,
    ):
        kwargs = {
            "device_ids": [rank],
            "process_group": process_group,
            "gradient_as_bucket_view": gradient_as_bucket_view,
            "static_graph": static_graph,
        }
        if bucket_cap_mb is not None:
            kwargs["bucket_cap_mb"] = bucket_cap_mb
        return nn.parallel.DistributedDataParallel(model, **kwargs)

    @staticmethod
    def _gradient(model, input_tensor, use_mean=True):
        output = model(input_tensor)
        loss = output.mean() if use_mean else output.sum()
        loss.backward()
        return [parameter.grad.detach().clone() for parameter in model.parameters()]

    @classmethod
    def _run_ddp_parity(
        cls,
        rank,
        world_size,
        port,
        gradient_as_bucket_view=False,
        static_graph=False,
        use_none_process_group=False,
        use_registration_helper=False,
    ):
        self = cls()
        process_group = self._init_process_group(rank, world_size, port)
        input_tensor = torch.full((40, 20), rank + 1.0).to(device_type)
        base_model = _AllreduceModel()

        reference_model = self._ddp(
            copy.deepcopy(base_model),
            rank,
            process_group,
            gradient_as_bucket_view,
            static_graph,
        )
        reference_grads = self._gradient(reference_model, input_tensor)

        hook_model = self._ddp(
            copy.deepcopy(base_model),
            rank,
            process_group,
            gradient_as_bucket_view,
            static_graph,
        )
        hook_state = None if use_none_process_group else process_group
        if use_registration_helper:
            register_ddp_comm_hook(DDPCommHookType.ALLREDUCE, hook_model, hook_state)
        else:
            hook_model.register_comm_hook(hook_state, default_hooks.allreduce_hook)
        hook_grads = self._gradient(hook_model, input_tensor)

        self.assertEqual(hook_grads, reference_grads)
        dist.destroy_process_group()

    @classmethod
    def _run_future_and_dtype_contract(cls, rank, world_size, port):
        self = cls()
        process_group = self._init_process_group(rank, world_size, port)

        # HCCL-supported floating-point gradient dtypes.
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            input_tensor = torch.full((4,), rank + 1.0, dtype=dtype).to(device_type)
            model = self._ddp(_AllreduceModel((4,), dtype), rank, process_group)
            state = {"process_group": process_group}
            model.register_comm_hook(state, _future_contract_hook)
            gradients = self._gradient(model, input_tensor, use_mean=False)
            expected = torch.full((4,), 1.5, dtype=dtype).to(device_type)

            self.assertEqual(gradients[0], expected)
            self.assertTrue(state["is_future"])
            self.assertEqual(state["result_shape"], state["buffer_shape"])
            self.assertEqual(state["result_dtype"], dtype)
            self.assertEqual(state["result_device_type"], device_type)

        dist.destroy_process_group()

    @classmethod
    def _run_overflow_boundary(cls, rank, world_size, port):
        self = cls()
        process_group = self._init_process_group(rank, world_size, port)
        model = self._ddp(_AllreduceModel((4,), torch.float16), rank, process_group)
        model.register_comm_hook(process_group, default_hooks.allreduce_hook)
        input_tensor = torch.full((4,), 60000.0, dtype=torch.float16).to(device_type)
        gradient = self._gradient(model, input_tensor, use_mean=False)[0]

        self.assertTrue(torch.isfinite(gradient).all().item())
        self.assertEqual(gradient, torch.full_like(gradient, 60000.0))
        dist.destroy_process_group()

    @classmethod
    def _run_custom_subgroup(cls, rank, world_size, port):
        self = cls()
        self._init_process_group(rank, world_size, port)
        subgroups = []
        try:
            for group_rank in range(world_size):
                subgroups.append(
                    dist.new_group(
                        [group_rank], backend="hccl", timeout=PROCESS_GROUP_TIMEOUT
                    )
                )
            subgroup = subgroups[rank]
            model = self._ddp(_AllreduceModel((4,)), rank, subgroup)
            model.register_comm_hook(subgroup, default_hooks.allreduce_hook)
            input_tensor = torch.full((4,), rank + 1.0).to(device_type)
            gradient = self._gradient(model, input_tensor, use_mean=False)[0]

            self.assertEqual(gradient, torch.full_like(gradient, rank + 1.0))
        finally:
            try:
                for subgroup in reversed(subgroups):
                    dist.destroy_process_group(subgroup)
            finally:
                dist.destroy_process_group()

    @classmethod
    def _run_multiple_buckets(cls, rank, world_size, port):
        self = cls()
        process_group = self._init_process_group(rank, world_size, port)
        base_model = _MultiBucketModel().to(device_type)
        input_tensor = torch.full((8, 32), rank + 1.0).to(device_type)
        reference_model = self._ddp(
            copy.deepcopy(base_model), rank, process_group, bucket_cap_mb=0.001
        )
        hook_model = self._ddp(
            copy.deepcopy(base_model), rank, process_group, bucket_cap_mb=0.001
        )
        state = {"process_group": process_group, "calls": 0}
        hook_model.register_comm_hook(state, _counting_hook)

        self._gradient(reference_model, input_tensor)
        self._gradient(hook_model, input_tensor)
        reference_model.zero_grad(set_to_none=True)
        hook_model.zero_grad(set_to_none=True)
        state["calls"] = 0
        reference_grads = self._gradient(reference_model, input_tensor)
        hook_grads = self._gradient(hook_model, input_tensor)

        self.assertGreater(state["calls"], 1)
        self.assertEqual(hook_grads, reference_grads)
        dist.destroy_process_group()

    @skipIfUnsupportMultiNPU(WORLD_SIZE)
    def _spawn(self, worker, *args):
        mp.spawn(
            worker,
            args=(WORLD_SIZE, find_free_port(), *args),
            nprocs=WORLD_SIZE,
            join=True,
        )

    def test_allreduce_hook(self):
        self._spawn(self._run_ddp_parity)

    def test_allreduce_hook_grad_is_view(self):
        self._spawn(self._run_ddp_parity, True)

    def test_allreduce_hook_static_graph(self):
        self._spawn(self._run_ddp_parity, False, True)

    def test_allreduce_hook_grad_is_view_static_graph(self):
        self._spawn(self._run_ddp_parity, True, True)

    def test_allreduce_hook_none_pg(self):
        self._spawn(self._run_ddp_parity, False, False, True)

    def test_allreduce_hook_registration_helper(self):
        self._spawn(self._run_ddp_parity, False, False, False, True)

    def test_allreduce_hook_future_and_dtypes(self):
        self._spawn(self._run_future_and_dtype_contract)

    def test_allreduce_hook_overflow_boundary(self):
        self._spawn(self._run_overflow_boundary)

    def test_allreduce_hook_custom_subgroup(self):
        self._spawn(self._run_custom_subgroup)

    def test_allreduce_hook_multiple_buckets(self):
        self._spawn(self._run_multiple_buckets)

    @skipIfUnsupportMultiNPU(1)
    def test_allreduce_hook_single_npu_contract(self):
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = str(find_free_port())
        torch.accelerator.set_device_index(0)
        dist.init_process_group(
            "hccl", rank=0, world_size=1, timeout=PROCESS_GROUP_TIMEOUT
        )
        try:
            process_group = dist.group.WORLD
            tensor = torch.tensor([1.003, 2.007]).to(device_type)
            expected = tensor.clone()
            bucket = _Bucket(tensor)

            future = default_hooks.allreduce_hook(process_group, bucket)
            result = future.wait()

            self.assertIsInstance(future, torch._C.Future)
            self.assertEqual(result, expected)
            self.assertEqual(result.shape, tensor.shape)
            self.assertEqual(result.dtype, tensor.dtype)
            self.assertEqual(result.device.type, device_type)
        finally:
            dist.destroy_process_group()

    @skipIfUnsupportMultiNPU(1)
    def test_allreduce_hook_invalid_arguments(self):
        bucket = _Bucket(torch.ones(4).to(device_type))

        with self.assertRaises(TypeError):
            default_hooks.allreduce_hook()
        with self.assertRaisesRegex(AttributeError, "has no attribute 'buffer'"):
            default_hooks.allreduce_hook(None, None)
        with self.assertRaisesRegex(AttributeError, "has no attribute 'size'"):
            default_hooks.allreduce_hook("invalid", bucket)


if __name__ == "__main__":
    run_tests()
