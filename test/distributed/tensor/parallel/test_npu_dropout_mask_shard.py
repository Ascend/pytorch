# Copyright (c) Meta Platforms, Inc. and affiliates
# Owner(s): ["oncall: distributed"]

from copy import deepcopy

import torch
import torch.nn as nn
from torch.distributed._tensor import (
    distribute_tensor,
    DTensor,
    init_device_mesh,
    Shard,
)
from torch.distributed.tensor.debug import CommDebugMode
from torch.distributed.tensor.parallel import parallelize_module
from torch.distributed.tensor.parallel.style import (
    SequenceParallel,
)
from torch.testing._internal.common_utils import run_tests
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    NUM_DEVICES,
)

import torch_npu
from torch_npu.testing.common_distributed import with_comms, skipIfUnsupportMultiNPU

c10d_functional = torch.ops.c10d_functional


class NPUDropoutMaskShardTest(DTensorTestBase):
    @property
    def device_type(self):
        return "npu"

    @property
    def world_size(self):
        return NUM_DEVICES

    @with_comms
    @skipIfUnsupportMultiNPU(NUM_DEVICES)
    def test_dropout_mask_shard0_and_backward(self):
        mesh = init_device_mesh(self.device_type, (self.world_size,))
        torch.distributed.tensor._random.manual_seed(0, mesh)

        comm_mode = CommDebugMode()
        batch, N, embedding_dim = 20, 8, 12

        global_input = torch.rand(
            batch,
            N * self.world_size,
            embedding_dim,
            device=self.device_type,
            requires_grad=True,
        )
        sharded_input = distribute_tensor(global_input, mesh, [Shard(1)])

        # forward
        with comm_mode:
            sharded_out, mask_dt = torch.ops.aten.native_dropout.default(
                sharded_input, 0.5, train=True
            )

            # 1. mask placement is Shard(0)
            self.assertIsInstance(mask_dt, DTensor)
            self.assertEqual(mask_dt.placements, (Shard(0),))

            # 2. no communication in forward
            self.assertEqual(comm_mode.get_total_counts(), 0)

        # backward
        with comm_mode:
            grad_out = torch.ones_like(sharded_out)
            sharded_out.backward(grad_out)

            # backward must not trigger mask all-to-all
            self.assertEqual(comm_mode.get_total_counts(), 0)
            self.assertEqual(
                comm_mode.get_comm_counts().get(
                    c10d_functional.all_to_all_single, 0
                ),
                0,
            )

        # 3. mask local tensor is valid 1D uint8 (NPU bit-packed format)
        mask_local = mask_dt._local_tensor
        self.assertEqual(mask_local.dim(), 1)
        self.assertEqual(mask_local.dtype, torch.uint8)

        # 4. mask redistribute to Shard(0) (no-op) must not fail
        # This verifies that Shard(0) is a legal placement for the 1D mask,
        # unlike Shard(1) which would be invalid on a 1D tensor.
        mask_redist = mask_dt.redistribute(mesh, [Shard(0)])
        self.assertEqual(mask_redist.placements, (Shard(0),))

    @with_comms
    @skipIfUnsupportMultiNPU(NUM_DEVICES)
    def test_dropout_mask_shard0_consistency(self):
        mesh = init_device_mesh(self.device_type, (self.world_size,))
        torch.distributed.tensor._random.manual_seed(0, mesh)

        batch, N, embedding_dim = 20, 8, 12

        global_input = torch.rand(
            batch,
            N * self.world_size,
            embedding_dim,
            device=self.device_type,
            requires_grad=True,
        )
        sharded_input = distribute_tensor(global_input, mesh, [Shard(1)])

        # global reference
        dropout = nn.Dropout(0.5).to(self.device_type)
        ref_out = dropout(global_input)
        ref_out.sum().backward()

        # sharded run
        sp_dropout = parallelize_module(
            deepcopy(dropout), mesh, SequenceParallel()
        )
        with CommDebugMode() as comm_mode:
            sharded_out = sp_dropout(sharded_input)
            grad_out = torch.ones_like(sharded_out)
            sharded_out.backward(grad_out)

            self.assertIsInstance(sharded_out, DTensor)
            self.assertEqual(sharded_out.placements, (Shard(1),))
            self.assertEqual(comm_mode.get_total_counts(), 0)

        # output consistency
        self.assertEqual(sharded_out.full_tensor().shape, ref_out.shape)


if __name__ == "__main__":
    run_tests()
