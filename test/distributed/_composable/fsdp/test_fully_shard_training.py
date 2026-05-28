# Owner(s): ["oncall: distributed"]

import copy
from collections.abc import Callable
from typing import Optional

import torch
import torch.distributed as dist
import torch.nn as nn

from torch.distributed.fsdp import (
    fully_shard,
    share_comm_ctx,
)
from torch.distributed.fsdp._fully_shard._fsdp_collectives import (
    foreach_all_gather,
    foreach_reduce,
)
from torch.testing._internal.common_fsdp import (
    check_sharded_parity,
    MLP,
    patch_foreach_all_gather,
    patch_foreach_reduce,
)
from torch.testing._internal.common_fsdp import get_devtype
from torch.testing._internal.common_utils import run_tests

from torch_npu.testing._internal.common_fsdp import FSDPNPUTest
from torch_npu.testing.common_distributed import skipIfUnsupportMultiNPU


device_type = torch.device(get_devtype())


class TestFullyShardShareCommContext(FSDPNPUTest):
    @property
    def world_size(self) -> int:
        return min(torch.get_device_module(device_type).device_count(), 2)

    @skipIfUnsupportMultiNPU(2)
    def test_share_comm_context(self):
        torch.manual_seed(42)
        n_layers = 3
        lin_dim = 16
        model = nn.Sequential(
            *[MLP(lin_dim, torch.device("cpu")) for _ in range(n_layers)]
        )
        ref_model = copy.deepcopy(model).to(device_type)
        for layer in model:
            fully_shard(layer)
            layer._get_fsdp_state()._lazy_init()
        share_comm_ctx(list(model))

        torch.manual_seed(42 + self.rank + 1)
        inp = torch.randn(4, 3, lin_dim, device=device_type.type)
        ref_loss = ref_model(inp).sum()

        all_gather_streams = set()
        reduce_scatter_streams = set()

        from torch.distributed.fsdp._fully_shard._fsdp_api import (
            AllGather,
            ReduceScatter,
        )
        from torch.distributed.fsdp._fully_shard._fsdp_param import FSDPParam

        orig_foreach_all_gather = foreach_all_gather

        def foreach_all_gather_with_assert(
            fsdp_params: list[FSDPParam],
            group: dist.ProcessGroup,
            async_op: bool,
            all_gather_copy_in_stream: torch.Stream,
            all_gather_stream: torch.Stream,
            device: torch.device,
            all_gather_comm: AllGather,
        ):
            nonlocal all_gather_streams
            all_gather_streams.add(all_gather_stream)
            return orig_foreach_all_gather(
                fsdp_params,
                group,
                async_op,
                all_gather_copy_in_stream,
                all_gather_stream,
                device,
                all_gather_comm,
            )

        orig_foreach_reduce = foreach_reduce

        @torch.no_grad()
        def foreach_reduce_with_assert(
            fsdp_params: list[FSDPParam],
            unsharded_grads: list[torch.Tensor],
            reduce_scatter_group: dist.ProcessGroup,
            reduce_scatter_stream: torch.Stream,
            reduce_scatter_comm: ReduceScatter,
            orig_dtype: Optional[torch.dtype],
            reduce_dtype: Optional[torch.dtype],
            device: torch.device,
            gradient_divide_factor: Optional[float],
            all_reduce_group: Optional[dist.ProcessGroup],  # not `None` iff HSDP
            all_reduce_stream: torch.Stream,
            all_reduce_grads: bool,
            partial_reduce_output: Optional[torch.Tensor],  # only used for HSDP
            all_reduce_hook: Optional[Callable[[torch.Tensor], None]],
            force_sum_reduction_for_comms: bool = False,
        ):
            nonlocal reduce_scatter_streams
            reduce_scatter_streams.add(reduce_scatter_stream)
            return orig_foreach_reduce(
                fsdp_params,
                unsharded_grads,
                reduce_scatter_group,
                reduce_scatter_stream,
                reduce_scatter_comm,
                orig_dtype,
                reduce_dtype,
                device,
                gradient_divide_factor,
                all_reduce_group,
                all_reduce_stream,
                all_reduce_grads,
                partial_reduce_output,
                all_reduce_hook,
                force_sum_reduction_for_comms,
            )

        with (
            patch_foreach_all_gather(foreach_all_gather_with_assert),
            patch_foreach_reduce(foreach_reduce_with_assert),
        ):
            loss = model(inp).sum()
            self.assertEqual(ref_loss, loss)
            ref_loss.backward()
            loss.backward()
            for param in ref_model.parameters():
                dist.all_reduce(param.grad, op=dist.ReduceOp.AVG)
        self.assertEqual(len(all_gather_streams), 1)
        self.assertEqual(len(reduce_scatter_streams), 1)
        check_sharded_parity(self, ref_model, model)


if __name__ == "__main__":
    run_tests()
