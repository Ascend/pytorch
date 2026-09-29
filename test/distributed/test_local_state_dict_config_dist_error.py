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
Add validation cases for torch.distributed.fsdp.LocalStateDictConfig
and torch.distributed.DistError in a torch_npu environment.

Plain Python execution launches local FSDP workers with a private file
rendezvous. WORLD_SIZE selects the worker count when set; otherwise all
visible NPUs are used. Existing process groups and torchrun workers are
also supported. At least two ranks are needed for actual FSDP sharding.
"""

import os
import tempfile
from contextlib import ExitStack
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed._shard.sharded_tensor import ShardedTensor
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    LocalStateDictConfig,
    ShardingStrategy,
    StateDictConfig,
    StateDictType,
)
from torch_npu.testing.testcase import TestCase, run_tests


def _check_local_state_dict(case, offload_to_cpu, requires_grad):
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    device = torch.device("npu", torch.npu.current_device())

    # Each rank receives 16 elements, for any supported world size.
    out_features = 4 * world_size
    total_numel = 4 * out_features
    shard_numel = total_numel // world_size
    offset = rank * shard_numel
    expected_weight = (
        torch.arange(total_numel, dtype=torch.float32)
        .reshape(out_features, 4) / 16
    )
    module = torch.nn.Linear(4, out_features, bias=False)
    module.requires_grad_(requires_grad)
    case.assertIs(module.weight.requires_grad, requires_grad)
    with torch.no_grad():
        module.weight.copy_(expected_weight)
    model = FSDP(
        module.to(device),
        device_id=device,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        use_orig_params=False,
    )
    config = LocalStateDictConfig(offload_to_cpu=offload_to_cpu)
    original = FSDP.get_state_dict_type(model)
    with FSDP.state_dict_type(model, StateDictType.LOCAL_STATE_DICT, config):
        active = FSDP.get_state_dict_type(model)
        case.assertEqual(active.state_dict_type, StateDictType.LOCAL_STATE_DICT)
        case.assertIsInstance(active.state_dict_config, LocalStateDictConfig)
        case.assertIs(active.state_dict_config.offload_to_cpu, offload_to_cpu)
        state = model.state_dict()

    restored = FSDP.get_state_dict_type(model)
    case.assertEqual(restored.state_dict_type, original.state_dict_type)
    case.assertEqual(restored.state_dict_config, original.state_dict_config)
    case.assertEqual(set(state), {"_flat_param"})
    flat = state["_flat_param"]
    case.assertIsInstance(flat, ShardedTensor)
    case.assertEqual(tuple(flat.size()), (total_numel,))
    shards = flat.local_shards()
    case.assertEqual(len(shards), 1)
    shard = shards[0]
    case.assertIs(shard.tensor.requires_grad, requires_grad)
    case.assertIs(
        flat.metadata().tensor_properties.requires_grad,
        shard.tensor.requires_grad,
    )
    case.assertEqual(shard.metadata.shard_offsets, [offset])
    case.assertEqual(shard.metadata.shard_sizes, [shard_numel])
    case.assertEqual(shard.tensor.numel(), shard_numel)
    case.assertEqual(shard.tensor.dtype, torch.float32)
    case.assertEqual(shard.tensor.device.type, "cpu" if offload_to_cpu else "npu")
    if not offload_to_cpu:
        case.assertEqual(shard.tensor.device.index, device.index)
    case.assertEqual(
        shard.tensor.cpu(), expected_weight.flatten()[offset:offset + shard_numel]
    )

    # A distinct zero-initialized model verifies that loading actually works.
    target_module = torch.nn.Linear(4, out_features, bias=False)
    target_module.requires_grad_(requires_grad)
    case.assertIs(target_module.weight.requires_grad, requires_grad)
    with torch.no_grad():
        target_module.weight.zero_()
    target = FSDP(
        target_module.to(device),
        device_id=device,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        use_orig_params=False,
    )
    with FSDP.state_dict_type(target, StateDictType.LOCAL_STATE_DICT, config):
        result = target.load_state_dict(state, strict=True)
    case.assertEqual(result.missing_keys, [])
    case.assertEqual(result.unexpected_keys, [])
    with FSDP.summon_full_params(target, writeback=False):
        case.assertEqual(target.module.weight.detach().cpu(), expected_weight)
    inputs = torch.arange(8, dtype=torch.float32).reshape(2, 4) / 8
    expected_output = inputs @ expected_weight.t()
    with torch.no_grad():
        actual_output = target(inputs.to(device)).cpu()
    case.assertRtolEqual(expected_output, actual_output, prec=1e-4)


def _fsdp_worker(rank, world_size, init_method, offload_to_cpu, requires_grad):
    # A top-level worker is required by multiprocessing's spawn start method.
    _run_worker(rank, rank, world_size, init_method, offload_to_cpu, requires_grad)


def _run_worker(rank, local_rank, world_size, init_method,
                offload_to_cpu, requires_grad):
    torch.npu.set_device(local_rank)
    print(
        f"rank={rank}, world_size={world_size}, local_rank={local_rank}, "
        f"current_device={torch.npu.current_device()}",
        flush=True,
    )
    with ExitStack() as cleanup:
        dist.init_process_group(
            backend="hccl",
            init_method=init_method,
            rank=rank,
            world_size=world_size,
            timeout=timedelta(seconds=120),
        )
        cleanup.callback(dist.destroy_process_group)
        _check_local_state_dict(TestCase(), offload_to_cpu, requires_grad)


class TestLocalStateDictConfig(TestCase):
    def test_default_config(self):
        config = LocalStateDictConfig()
        self.assertIsInstance(config, StateDictConfig)
        self.assertIs(config.offload_to_cpu, False)

    def test_explicit_config(self):
        for offload in (False, True):
            with self.subTest(offload_to_cpu=offload):
                config = LocalStateDictConfig(offload_to_cpu=offload)
                self.assertIs(config.offload_to_cpu, offload)

    def _run_fsdp_test(self, offload_to_cpu, requires_grad):
        # Do not reinitialize or destroy a group owned by the caller.
        if dist.is_initialized():
            if dist.get_world_size() < 2:
                self.skipTest("FSDP sharding requires at least two ranks")
            _check_local_state_dict(self, offload_to_cpu, requires_grad)
            return

        # torchrun already created workers. Do not spawn another worker layer.
        launch_keys = ("LOCAL_RANK", "RANK", "MASTER_ADDR", "MASTER_PORT")
        if all(key in os.environ for key in launch_keys):
            world_size = int(os.environ["WORLD_SIZE"])
            if world_size < 2:
                self.skipTest("FSDP sharding requires at least two ranks")
            _run_worker(
                int(os.environ["RANK"]), int(os.environ["LOCAL_RANK"]),
                world_size, "env://", offload_to_cpu, requires_grad,
            )
            return

        # CI invokes this file with plain Python. WORLD_SIZE alone does not
        # mean that another process has launched distributed workers for us.
        device_count = torch.npu.device_count() if torch.npu.is_available() else 0
        world_size = int(os.environ.get("WORLD_SIZE", str(device_count)))
        if world_size < 2 or device_count < world_size:
            self.skipTest(
                f"Local FSDP workers need at least two NPUs and enough visible "
                f"devices: requested={world_size}, visible={device_count}"
            )
        # A fresh private path for each test avoids stale rendezvous files.
        with tempfile.TemporaryDirectory(prefix="fsdp_local_state_") as temp_dir:
            init_method = (Path(temp_dir) / "rendezvous").as_uri()
            mp.spawn(
                _fsdp_worker,
                args=(world_size, init_method, offload_to_cpu, requires_grad),
                nprocs=world_size,
                join=True,
            )

    def test_local_state_dict_on_npu(self):
        self._run_fsdp_test(offload_to_cpu=False, requires_grad=True)

    def test_local_state_dict_offload_frozen_params_to_cpu(self):
        self._run_fsdp_test(offload_to_cpu=True, requires_grad=False)


class TestDistError(TestCase):
    def test_inheritance_and_message(self):
        message = "distributed API test error"
        error = dist.DistError(message)
        self.assertTrue(issubclass(dist.DistError, RuntimeError))
        self.assertIsInstance(error, RuntimeError)
        self.assertEqual(str(error), message)
        self.assertEqual(error.args, (message,))

    def test_raise_and_catch(self):
        message = "distributed API test error"
        for catch_type in (dist.DistError, RuntimeError):
            with self.subTest(catch_type=catch_type.__name__):
                error = dist.DistError(message)
                with self.assertRaises(catch_type) as caught:
                    raise error
                self.assertIs(caught.exception, error)
                self.assertEqual(str(caught.exception), message)

    def test_subclasses_caught_by_base(self):
        # Synthetic raises verify the hierarchy, not backend fault handling.
        for error_type in (dist.DistBackendError, dist.DistNetworkError, dist.DistStoreError):
            with self.subTest(error_type=error_type.__name__):
                self.assertTrue(issubclass(error_type, dist.DistError))
                with self.assertRaises(dist.DistError) as caught:
                    raise error_type("subclass test error")
                self.assertIs(type(caught.exception), error_type)
                self.assertEqual(str(caught.exception), "subclass test error")

    def test_real_store_timeout_caught_by_dist_error(self):
        # OS-selected port avoids hard-coded port conflicts; each rank owns a store.
        store = dist.TCPStore(
            host_name="127.0.0.1",
            port=0,
            world_size=1,
            is_master=True,
            timeout=timedelta(seconds=2),
            use_libuv=False,
        )
        with self.assertRaises(dist.DistError) as caught:
            store.get("key_that_was_never_written")
        self.assertIsInstance(caught.exception, dist.DistStoreError)
        self.assertTrue(str(caught.exception))


if __name__ == "__main__":
    run_tests()
