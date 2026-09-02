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
   communication hook wrappers.
2. This file validates
   torch.distributed.algorithms.ddp_comm_hooks.default_hooks.fp16_compress_wrapper
   (extendable).
"""

import copy
import os

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn
from torch.distributed.algorithms.ddp_comm_hooks import default_hooks
from torch.distributed.algorithms.ddp_comm_hooks import powerSGD_hook as powerSGD
from torch.testing._internal.common_utils import TestCase, find_free_port, run_tests

from torch_npu.testing.common_distributed import skipIfUnsupportMultiNPU


device_type = acc.type if (acc := torch.accelerator.current_accelerator()) else "cpu"

WORLD_SIZE = 2


class _Fp16WrapperModel(nn.Module):
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


class _MutableBucket:
    def __init__(self, tensor):
        self.tensor = tensor

    def buffer(self):
        return self.tensor

    def set_buffer(self, tensor):
        self.tensor = tensor


def _recording_allreduce_hook(
    state, bucket: dist.GradBucket
) -> torch.futures.Future[torch.Tensor]:
    state["inner_called"] = True
    state["inner_dtype"] = bucket.buffer().dtype
    future = default_hooks.allreduce_hook(state["process_group"], bucket)
    state["inner_is_future"] = isinstance(future, torch._C.Future)

    def validate(fut):
        result = fut.value()
        state["inner_result_dtype"] = result.dtype
        return result

    return future.then(validate)


def _wrapper_contract_hook(
    state, bucket: dist.GradBucket
) -> torch.futures.Future[torch.Tensor]:
    buffer_shape = bucket.buffer().shape
    future = state["wrapped_hook"](state, bucket)
    state["wrapper_is_future"] = isinstance(future, torch._C.Future)

    def validate(fut):
        result = fut.value()
        state["wrapper_result_shape"] = result.shape
        state["wrapper_result_dtype"] = result.dtype
        state["wrapper_result_device_type"] = result.device.type
        state["original_shape"] = buffer_shape
        return result

    return future.then(validate)


def _counting_allreduce_hook(
    state, bucket: dist.GradBucket
) -> torch.futures.Future[torch.Tensor]:
    state["calls"] += 1
    return default_hooks.allreduce_hook(state["process_group"], bucket)


def _tensor_returning_hook(state, bucket):
    return bucket.buffer()


def _recording_future_hook(state, bucket):
    state["called"] = True
    state["dtype"] = bucket.buffer().dtype
    future = torch.futures.Future()
    future.set_result(bucket.buffer().add(1))
    return future


def _raising_hook(state, bucket):
    raise RuntimeError("inner hook failure")


class TestFp16CompressWrapper(TestCase):
    @staticmethod
    def _init_process_group(rank, world_size, port):
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = str(port)
        torch.accelerator.set_device_index(rank)
        dist.init_process_group("hccl", rank=rank, world_size=world_size)
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
        use_power_sgd=False,
    ):
        self = cls()
        process_group = self._init_process_group(rank, world_size, port)
        input_tensor = torch.full((40, 20), rank + 1.0).to(device_type)
        base_model = _Fp16WrapperModel()
        reference_model = self._ddp(
            copy.deepcopy(base_model),
            rank,
            process_group,
            gradient_as_bucket_view,
            static_graph,
        )
        if use_power_sgd:
            inner_hook = powerSGD.powerSGD_hook
            hook_state = powerSGD.PowerSGDState(
                process_group=process_group,
                start_powerSGD_iter=2,
            )
        else:
            inner_hook = default_hooks.allreduce_hook
            hook_state = None if use_none_process_group else process_group
        wrapped_hook = default_hooks.fp16_compress_wrapper(inner_hook)
        hook_model = self._ddp(
            copy.deepcopy(base_model),
            rank,
            process_group,
            gradient_as_bucket_view,
            static_graph,
        )
        hook_model.register_comm_hook(hook_state, wrapped_hook)

        iterations = 3 if use_power_sgd else 1
        for _ in range(iterations):
            reference_model.zero_grad(set_to_none=True)
            hook_model.zero_grad(set_to_none=True)
            reference_grads = self._gradient(
                reference_model, input_tensor, use_mean=False
            )
            hook_grads = self._gradient(
                hook_model, input_tensor, use_mean=False
            )

        if use_power_sgd:
            self.assertGreater(hook_state.iter, hook_state.start_powerSGD_iter)
            self.assertGreater(hook_state.total_numel_after_compression, 0)
            self.assertEqual(hook_grads, reference_grads, rtol=1e-3, atol=1e-3)
        else:
            self.assertEqual(hook_grads, reference_grads)
        dist.destroy_process_group()

    @staticmethod
    def _compressed_average(dtype):
        values = []
        for value in (1.003, 2.007):
            tensor = torch.full((4,), value, dtype=dtype).to(device_type)
            values.append(tensor.to(torch.float16).div_(WORLD_SIZE))
        return (values[0] + values[1]).to(dtype)

    @classmethod
    def _run_future_dtype_and_state_contract(cls, rank, world_size, port):
        self = cls()
        process_group = self._init_process_group(rank, world_size, port)

        # HCCL-supported floating-point gradient dtypes.
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            value = (1.003, 2.007)[rank]
            input_tensor = torch.full((4,), value, dtype=dtype).to(device_type)
            model = self._ddp(_Fp16WrapperModel((4,), dtype), rank, process_group)
            wrapped_hook = default_hooks.fp16_compress_wrapper(
                _recording_allreduce_hook
            )
            state = {
                "process_group": process_group,
                "wrapped_hook": wrapped_hook,
            }
            model.register_comm_hook(state, _wrapper_contract_hook)
            gradients = self._gradient(model, input_tensor, use_mean=False)

            self.assertEqual(gradients[0], self._compressed_average(dtype))
            self.assertEqual(gradients[0].dtype, dtype)
            self.assertTrue(state["inner_called"])
            self.assertTrue(state["inner_is_future"])
            self.assertTrue(state["wrapper_is_future"])
            self.assertEqual(state["inner_dtype"], torch.float16)
            self.assertEqual(state["inner_result_dtype"], torch.float16)
            self.assertEqual(state["wrapper_result_dtype"], torch.float16)
            self.assertEqual(
                state["wrapper_result_shape"], state["original_shape"]
            )
            self.assertEqual(state["wrapper_result_device_type"], device_type)

        dist.destroy_process_group()

    @classmethod
    def _run_custom_subgroup(cls, rank, world_size, port):
        self = cls()
        self._init_process_group(rank, world_size, port)
        subgroups = []
        try:
            for group_rank in range(world_size):
                subgroups.append(dist.new_group([group_rank], backend="hccl"))
            subgroup = subgroups[rank]
            model = self._ddp(_Fp16WrapperModel((4,)), rank, subgroup)
            wrapped_hook = default_hooks.fp16_compress_wrapper(
                default_hooks.allreduce_hook
            )
            model.register_comm_hook(subgroup, wrapped_hook)
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
    def _run_predivide_overflow_boundary(cls, rank, world_size, port):
        self = cls()
        process_group = self._init_process_group(rank, world_size, port)
        model = self._ddp(_Fp16WrapperModel((4,)), rank, process_group)
        wrapped_hook = default_hooks.fp16_compress_wrapper(
            default_hooks.allreduce_hook
        )
        model.register_comm_hook(process_group, wrapped_hook)
        # Pre-division avoids a 120000 FP16 intermediate during all-reduce.
        input_tensor = torch.full((4,), 60000.0).to(device_type)
        output = model(input_tensor)
        output.backward(torch.ones_like(output))
        gradient = next(model.parameters()).grad.detach().clone()

        self.assertTrue(torch.isfinite(gradient).all().item())
        self.assertEqual(gradient, torch.full_like(gradient, 60000.0))
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
        wrapped_hook = default_hooks.fp16_compress_wrapper(
            _counting_allreduce_hook
        )
        hook_model.register_comm_hook(state, wrapped_hook)

        self._gradient(reference_model, input_tensor, use_mean=False)
        self._gradient(hook_model, input_tensor, use_mean=False)
        reference_model.zero_grad(set_to_none=True)
        hook_model.zero_grad(set_to_none=True)
        state["calls"] = 0
        reference_grads = self._gradient(
            reference_model, input_tensor, use_mean=False
        )
        hook_grads = self._gradient(hook_model, input_tensor, use_mean=False)

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

    def test_fp16_compress_wrapper_allreduce(self):
        self._spawn(self._run_ddp_parity)

    def test_fp16_compress_wrapper_allreduce_grad_is_view(self):
        self._spawn(self._run_ddp_parity, True)

    def test_fp16_compress_wrapper_allreduce_static_graph(self):
        self._spawn(self._run_ddp_parity, False, True)

    def test_fp16_compress_wrapper_allreduce_grad_is_view_static_graph(self):
        self._spawn(self._run_ddp_parity, True, True)

    def test_fp16_compress_wrapper_allreduce_none_pg(self):
        self._spawn(self._run_ddp_parity, False, False, True)

    def test_fp16_compress_wrapper_powersgd(self):
        self._spawn(self._run_ddp_parity, False, False, False, True)

    def test_fp16_compress_wrapper_powersgd_grad_is_view(self):
        self._spawn(self._run_ddp_parity, True, False, False, True)

    def test_fp16_compress_wrapper_powersgd_static_graph(self):
        self._spawn(self._run_ddp_parity, False, True, False, True)

    def test_fp16_compress_wrapper_powersgd_grad_is_view_static_graph(self):
        self._spawn(self._run_ddp_parity, True, True, False, True)

    def test_fp16_compress_wrapper_future_dtype_and_state(self):
        self._spawn(self._run_future_dtype_and_state_contract)

    def test_fp16_compress_wrapper_custom_subgroup(self):
        self._spawn(self._run_custom_subgroup)

    def test_fp16_compress_wrapper_predivide_overflow_boundary(self):
        self._spawn(self._run_predivide_overflow_boundary)

    def test_fp16_compress_wrapper_multiple_buckets(self):
        self._spawn(self._run_multiple_buckets)

    @skipIfUnsupportMultiNPU(1)
    def test_fp16_compress_wrapper_single_npu_contract(self):
        tensor = torch.tensor([1.003, 2.007]).to(device_type)
        bucket = _MutableBucket(tensor)
        state = {"called": False}
        wrapped_hook = default_hooks.fp16_compress_wrapper(
            _recording_future_hook
        )

        future = wrapped_hook(state, bucket)
        result = future.wait()

        self.assertTrue(state["called"])
        self.assertEqual(state["dtype"], torch.float16)
        self.assertIsInstance(future, torch._C.Future)
        self.assertEqual(result, tensor.to(torch.float16).add(1))
        self.assertEqual(result.dtype, torch.float16)
        self.assertEqual(result.device.type, device_type)

    @skipIfUnsupportMultiNPU(1)
    def test_fp16_compress_wrapper_invalid_arguments(self):
        tensor = torch.ones(4).to(device_type)

        with self.assertRaises(TypeError):
            default_hooks.fp16_compress_wrapper()
        with self.assertRaises(AttributeError):
            default_hooks.fp16_compress_wrapper(default_hooks.allreduce_hook)(
                None, None
            )
        with self.assertRaises(TypeError):
            default_hooks.fp16_compress_wrapper(None)(
                None, _MutableBucket(tensor)
            )
        with self.assertRaises(AttributeError):
            default_hooks.fp16_compress_wrapper(_tensor_returning_hook)(
                None, _MutableBucket(tensor)
            )
        with self.assertRaises(RuntimeError):
            default_hooks.fp16_compress_wrapper(_raising_hook)(
                None, _MutableBucket(tensor)
            )
        with self.assertRaises(AttributeError):
            default_hooks.fp16_compress_wrapper(default_hooks.allreduce_hook)(
                None, (tensor,)
            )


if __name__ == "__main__":
    run_tests()
