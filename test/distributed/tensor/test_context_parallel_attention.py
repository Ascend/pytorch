# Copyright (c) Meta Platforms, Inc. and affiliates
# Owner(s): ["oncall: distributed"]

"""
NPU context parallel SDPA regression tests.

The main scenario intentionally mirrors PyTorch's
RingAttentionTest.test_ring_attention_sdpa so torch_npu validates the same
context-parallel user behavior as native DTensor. NPU-specific assertions also
cover the fused npu_fusion_attention_v3 dispatcher path, ring softmax merge,
BNSD layout handling, communication counts, bf16 tolerance, and gradient
unsharding behavior.
"""

from collections.abc import Callable

import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch_npu
import torch_npu.distributed.tensor.experimental._context_parallel._attention
from torch.distributed.tensor import DeviceMesh
from torch.distributed.tensor.debug import CommDebugMode
from torch.distributed.tensor.experimental._context_parallel import (
    _CausalBehavior,
    _context_parallel_shard,
    _ContextParallel,
    _cp_options,
    _disable_context_parallel_dispatcher,
    _enable_context_parallel_dispatcher,
    _HeadTailLoadBalancer,
    _is_causal_behavior,
    _PerDocumentHeadTailLoadBalancer,
    _RotateMethod,
    context_parallel,
    context_parallel_unshard,
    set_rotate_method,
)
from torch.distributed.tensor.parallel import parallelize_module
from torch.nn.attention import sdpa_kernel, SDPBackend

from torch_npu.testing._internal.common_dtensor import NPUDTensorTestBase
from torch_npu.testing.common_distributed import skipIfUnsupportMultiNPU, with_comms
from torch_npu.testing.common_utils import SupportedDevices
from torch_npu.testing.testcase import run_tests


c10d_functional = torch.ops.c10d_functional

ROTATER_ENUM_TO_STR = {
    _RotateMethod.ALL_GATHER: "allgather",
    _RotateMethod.ALL_TO_ALL: "alltoall",
}

ATTENTION_TOLERANCES = {
    torch.bfloat16: (1e-2, 8e-3),
    torch.float32: (2e-6, 1e-5),
}

def _ordered_bf16_bits(x: torch.Tensor) -> torch.Tensor:
    bits = x.detach().to(torch.bfloat16).cpu().contiguous().view(torch.int16).to(torch.int32)
    bits = torch.where(bits < 0, bits + 65536, bits)
    sign = (bits & 0x8000) != 0
    return torch.where(sign, 0xFFFF - bits, bits + 0x8000)


def _bf16_ulp_diff(actual: torch.Tensor, expected: torch.Tensor) -> torch.Tensor:
    return (_ordered_bf16_bits(actual) - _ordered_bf16_bits(expected)).abs()


class SDPAWrapper(torch.nn.Module):
    def __init__(self, compiled: bool, backend: SDPBackend) -> None:
        super().__init__()
        self.compiled = compiled
        self.backend = backend
        if compiled:
            self._compiled_sdpa = torch.compile(
                F.scaled_dot_product_attention,
                fullgraph=True,
                backend="aot_eager",
            )

    def forward(self, *args: object, **kwargs: object) -> torch.Tensor:
        with sdpa_kernel(self.backend):
            if self.compiled:
                return self._compiled_sdpa(*args, **kwargs)
            return F.scaled_dot_product_attention(*args, **kwargs)


class TestContextParallelAttention(NPUDTensorTestBase):
    @property
    def world_size(self) -> int:
        device_count = torch.npu.device_count() if torch.npu.is_available() else 0
        return min(8, device_count) if device_count >= 2 else 2

    def _make_load_balancer(
        self,
        load_balance: bool,
        seq_length: int,
    ) -> _HeadTailLoadBalancer | None:
        if not load_balance:
            return None
        return _HeadTailLoadBalancer(seq_length, self.world_size, torch.device(self.device_type))

    def _ring_attention_sdpa(
        self,
        cp_q: torch.Tensor,
        cp_k: torch.Tensor,
        cp_v: torch.Tensor,
        *,
        fn_eval: Callable,
        mesh: DeviceMesh,
        seq_dim: int,
        is_causal: bool,
        compiled: bool,
        backend: SDPBackend,
        rotater: _RotateMethod,
        test_forward_only: bool,
        load_balance: bool,
        use_context: bool,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None, int]:
        ring_update_call_count = 0
        original_ring_update = getattr(torch_npu, "npu_ring_attention_update", None)

        if callable(original_ring_update):
            def counted_ring_update(*args, **kwargs):
                nonlocal ring_update_call_count
                ring_update_call_count += 1
                return original_ring_update(*args, **kwargs)

            torch_npu.npu_ring_attention_update = counted_ring_update

        cp_context = None
        try:
            if not use_context:
                cp_plan = _ContextParallel(
                    seq_dim=seq_dim,
                    attention_type=_ContextParallel.AttentionType.SDPA,
                )
                attention = parallelize_module(SDPAWrapper(compiled=compiled, backend=backend), mesh, cp_plan)
                load_balancer = self._make_load_balancer(load_balance, cp_q.size(seq_dim))
                cp_q, cp_k, cp_v = _context_parallel_shard(
                    mesh,
                    (cp_q, cp_k, cp_v),
                    (seq_dim,) * 3,
                    load_balancer=load_balancer,
                )
                _enable_context_parallel_dispatcher()
            else:
                _cp_options.enable_load_balance = load_balance
                cp_context = context_parallel(
                    mesh,
                    buffers=(cp_q, cp_k, cp_v),
                    buffer_seq_dims=(seq_dim,) * 3,
                )
                cp_context.__enter__()
                attention = F.scaled_dot_product_attention
                if compiled:
                    attention = torch.compile(attention, fullgraph=True, backend="aot_eager")

            for target in (cp_q, cp_k, cp_v):
                target.requires_grad = True

            with CommDebugMode() as comm_mode:
                with sdpa_kernel(backend):
                    cp_out = fn_eval(
                        attention,
                        cp_q,
                        cp_k,
                        cp_v,
                        is_causal=is_causal,
                    )

                if not compiled and rotater == _RotateMethod.ALL_TO_ALL:
                    expected_all2all = (
                        self.world_size - 1
                        if test_forward_only
                        else self.world_size * 3 - 2
                    )
                    self.assertDictEqual(
                        comm_mode.get_comm_counts(),
                        {c10d_functional.all_to_all_single: expected_all2all},
                    )

            cp_dq, cp_dk, cp_dv = cp_q.grad, cp_k.grad, cp_v.grad
            for target in (cp_q, cp_k, cp_v):
                target.requires_grad = False
            return cp_out, cp_dq, cp_dk, cp_dv, ring_update_call_count
        finally:
            if not use_context:
                _disable_context_parallel_dispatcher()
            elif cp_context is not None:
                cp_context.__exit__(None, None, None)
            if callable(original_ring_update):
                torch_npu.npu_ring_attention_update = original_ring_update

    @SupportedDevices(["Ascend910B"])
    @skipIfUnsupportMultiNPU(2)
    @with_comms
    def test_ring_attention_sdpa(self) -> None:
        self.run_subtests(
            {
                "is_causal": [True, False],
                "compiled": [False, True],
                "backend": [SDPBackend.OVERRIDEABLE],
                "load_balance": [False, True],
                "rotater": [_RotateMethod.ALL_TO_ALL, _RotateMethod.ALL_GATHER],
                "test_forward_only": [False, True],
                "use_context": [False, True],
                "dtype": [torch.bfloat16, torch.float32],
            },
            self._test_ring_attention_sdpa,
        )

    def _test_ring_attention_sdpa(
        self,
        is_causal: bool,
        compiled: bool,
        backend: SDPBackend,
        load_balance: bool,
        rotater: _RotateMethod,
        test_forward_only: bool,
        use_context: bool,
        dtype: torch.dtype,
    ) -> None:
        if load_balance and not is_causal:
            return

        # Align with native PyTorch 2.12: CP compiled paths are not supported yet
        # because DTensor dispatch interferes with SDPA tracing.
        if compiled:
            return

        set_rotate_method(ROTATER_ENUM_TO_STR[rotater])
        self.assertEqual(_cp_options.rotate_method, rotater)

        device_mesh = DeviceMesh(self.device_type, torch.arange(0, self.world_size))
        bs = 8
        seq_length = 1024
        seq_dim = 2
        dim = 32
        nheads = 8

        torch.manual_seed(10)
        q, k, v = [
            torch.rand(
                (bs, nheads, seq_length * self.world_size, dim),
                device=self.device_type,
                dtype=dtype,
                requires_grad=True,
            )
            for _ in range(3)
        ]

        with torch.no_grad():
            dist.broadcast(q, src=0)
            dist.broadcast(k, src=0)
            dist.broadcast(v, src=0)

        def fn_eval(fn, *args, **kwargs):
            if test_forward_only:
                with torch.no_grad():
                    return fn(*args, **kwargs)
            out = fn(*args, **kwargs)
            out.sum().backward()
            return out

        with sdpa_kernel(backend):
            out = fn_eval(F.scaled_dot_product_attention, q, k, v, is_causal=is_causal)

        cp_q, cp_k, cp_v = [target.detach().clone() for target in (q, k, v)]
        cp_out, cp_dq, cp_dk, cp_dv, ring_update_call_count = self._ring_attention_sdpa(
            cp_q,
            cp_k,
            cp_v,
            fn_eval=fn_eval,
            mesh=device_mesh,
            seq_dim=seq_dim,
            is_causal=is_causal,
            compiled=compiled,
            backend=backend,
            rotater=rotater,
            test_forward_only=test_forward_only,
            load_balance=load_balance,
            use_context=use_context,
        )

        call_count = torch.tensor([ring_update_call_count], device=self.device_type)
        dist.all_reduce(call_count)
        if callable(getattr(torch_npu, "npu_ring_attention_update", None)):
            self.assertGreater(call_count.item(), 0)

        load_balancer = self._make_load_balancer(load_balance, q.size(seq_dim))
        (cp_out,) = context_parallel_unshard(
            device_mesh,
            [cp_out],
            [seq_dim],
            load_balancer=load_balancer,
        )

        atol, rtol = ATTENTION_TOLERANCES[dtype]
        torch.testing.assert_close(out, cp_out, atol=atol, rtol=rtol)
        if dtype == torch.bfloat16:
            self.assertLessEqual(int(_bf16_ulp_diff(cp_out, out).max().item()), 1)

        if test_forward_only:
            return

        cp_dq, cp_dk, cp_dv = context_parallel_unshard(
            device_mesh,
            [cp_dq, cp_dk, cp_dv],
            [seq_dim] * 3,
            load_balancer=load_balancer,
        )

        torch.testing.assert_close(q.grad, cp_dq, atol=atol, rtol=rtol)
        torch.testing.assert_close(k.grad, cp_dk, atol=atol, rtol=rtol)
        torch.testing.assert_close(v.grad, cp_dv, atol=atol, rtol=rtol)
        if dtype == torch.bfloat16:
            self.assertLessEqual(int(_bf16_ulp_diff(cp_dv, v.grad).max().item()), 1)


class TestContextParallelUtilities(NPUDTensorTestBase):
    @property
    def world_size(self) -> int:
        return 2

    def test_is_causal_behavior(self) -> None:
        saved_enable_load_balance = _cp_options.enable_load_balance
        try:
            _cp_options.enable_load_balance = False
            self.assertEqual(
                _is_causal_behavior(rank=0, world_size=4, i=0, is_causal=False),
                _CausalBehavior.NOT_IS_CAUSAL,
            )

            ranks = [
                [_CausalBehavior.IS_CAUSAL, _CausalBehavior.SKIP],
                [_CausalBehavior.IS_CAUSAL, _CausalBehavior.NOT_IS_CAUSAL],
            ]
            for rank, iters in enumerate(ranks):
                for i, behavior in enumerate(iters):
                    self.assertEqual(
                        _is_causal_behavior(
                            rank=rank,
                            world_size=self.world_size,
                            i=i,
                            is_causal=True,
                        ),
                        behavior,
                    )

            _cp_options.enable_load_balance = True
            ranks = [
                [_CausalBehavior.IS_CAUSAL, _CausalBehavior.NOT_IS_CAUSAL],
                [_CausalBehavior.IS_CAUSAL, _CausalBehavior.NOT_IS_CAUSAL],
            ]
            for rank, iters in enumerate(ranks):
                for i, behavior in enumerate(iters):
                    self.assertEqual(
                        _is_causal_behavior(
                            rank=rank,
                            world_size=self.world_size,
                            i=i,
                            is_causal=True,
                        ),
                        behavior,
                    )
        finally:
            _cp_options.enable_load_balance = saved_enable_load_balance

    @SupportedDevices(["Ascend910B"])
    @skipIfUnsupportMultiNPU(2)
    @with_comms
    def test_context_parallel_shard_head_tail_load_balance(self) -> None:
        seq_len = 32
        device_mesh = DeviceMesh(self.device_type, torch.arange(0, self.world_size))
        positions = torch.arange(seq_len, device=self.device_type, dtype=torch.float32)
        load_balancer = _HeadTailLoadBalancer(
            seq_len,
            self.world_size,
            torch.device(self.device_type),
        )

        (positions_shard,) = _context_parallel_shard(
            device_mesh,
            [positions],
            [0],
            load_balancer=load_balancer,
        )

        indices = load_balancer._generate_indices(restore=False)[0].long()
        expected = torch.index_select(positions, dim=0, index=indices)
        expected_shard = expected.chunk(self.world_size, dim=0)[self.rank]
        torch.testing.assert_close(positions_shard, expected_shard, atol=0, rtol=0)

    @SupportedDevices(["Ascend910B"])
    @skipIfUnsupportMultiNPU(2)
    @with_comms
    def test_context_parallel_shard_per_document_head_tail_load_balance(self) -> None:
        batch_size = 2
        seq_len = 32
        seq_lengths_per_doc = [[8, 8, 16], [16, 8, 8]]
        device_mesh = DeviceMesh(self.device_type, torch.arange(0, self.world_size))
        positions = torch.arange(
            batch_size * seq_len,
            device=self.device_type,
            dtype=torch.float32,
        ).reshape(batch_size, seq_len)
        load_balancer = _PerDocumentHeadTailLoadBalancer(
            seq_lengths_per_doc,
            self.world_size,
            torch.device(self.device_type),
        )

        (positions_shard,) = _context_parallel_shard(
            device_mesh,
            [positions],
            [1],
            load_balancer=load_balancer,
        )

        indices = load_balancer._generate_indices(restore=False).long()
        expected = torch.gather(positions, dim=1, index=indices)
        expected_shard = expected.chunk(self.world_size, dim=1)[self.rank]
        torch.testing.assert_close(positions_shard, expected_shard, atol=0, rtol=0)

        (positions_unshard,) = context_parallel_unshard(
            device_mesh,
            [positions_shard],
            [1],
            load_balancer=load_balancer,
        )
        torch.testing.assert_close(positions_unshard, positions, atol=0, rtol=0)


if __name__ == "__main__":
    run_tests()
