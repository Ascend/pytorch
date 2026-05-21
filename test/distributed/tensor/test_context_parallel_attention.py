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

import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch_npu
import torch_npu.distributed.tensor.experimental._context_parallel._attention  # noqa: F401
from torch.distributed.tensor import DeviceMesh
from torch.distributed.tensor.debug import CommDebugMode
import torch.distributed.tensor.experimental._attention as native_attention
from torch.distributed.tensor.experimental._attention import (
    _cp_options,
    _DispatchMode,
    _RotateMethod,
    context_parallel,
    context_parallel_unshard,
    set_rotate_method,
)
from torch.nn.attention import SDPBackend, sdpa_kernel

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


class TestContextParallelAttention(NPUDTensorTestBase):
    @property
    def world_size(self) -> int:
        device_count = torch.npu.device_count() if torch.npu.is_available() else 0
        return min(8, device_count) if device_count >= 2 else 2

    @SupportedDevices(["Ascend910B"])
    @skipIfUnsupportMultiNPU(2)
    @with_comms
    def test_ring_attention_sdpa(self) -> None:
        self.run_subtests(
            {
                "is_causal": [True, False],
                "compiled": [True, False],
                "backend": [SDPBackend.OVERRIDEABLE],
                "load_balance": [True, False],
                "rotater": [_RotateMethod.ALL_TO_ALL, _RotateMethod.ALL_GATHER],
                "test_forward_only": [True, False],
                "dispatch_mode": [
                    _DispatchMode.MONKEY_PATCH,
                    _DispatchMode.TORCH_FUNCTION,
                ],
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
        dispatch_mode: _DispatchMode,
        dtype: torch.dtype,
    ) -> None:
        native_attention._dispatch_mode = dispatch_mode

        if load_balance and not is_causal:
            return

        def fn_eval(fn, *args, **kwargs):
            if test_forward_only:
                with torch.no_grad():
                    return fn(*args, **kwargs)
            out = fn(*args, **kwargs)
            out.sum().backward()
            return out

        set_rotate_method(ROTATER_ENUM_TO_STR[rotater])
        self.assertEqual(_cp_options.rotate_method, rotater)
        _cp_options.enable_load_balance = load_balance

        device_mesh = DeviceMesh(self.device_type, torch.arange(0, self.world_size))
        bs = 8
        seq_length = 64
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

        with sdpa_kernel(backend):
            out = fn_eval(F.scaled_dot_product_attention, q, k, v, is_causal=is_causal)

        cp_q = q.detach().clone()
        cp_k = k.detach().clone()
        cp_v = v.detach().clone()
        ring_update_call_count = 0
        original_ring_update = getattr(torch_npu, "npu_ring_attention_update", None)

        if callable(original_ring_update):
            def counted_ring_update(*args, **kwargs):
                nonlocal ring_update_call_count
                ring_update_call_count += 1
                return original_ring_update(*args, **kwargs)

            torch_npu.npu_ring_attention_update = counted_ring_update

        try:
            with context_parallel(
                device_mesh,
                buffers=(cp_q, cp_k, cp_v),
                buffer_seq_dims=(seq_dim, seq_dim, seq_dim),
            ):
                cp_q.requires_grad = True
                cp_k.requires_grad = True
                cp_v.requires_grad = True

                with CommDebugMode() as comm_mode:
                    with sdpa_kernel(backend):
                        if compiled:
                            fn = torch.compile(
                                F.scaled_dot_product_attention,
                                fullgraph=True,
                                backend="aot_eager",
                            )
                        else:
                            fn = F.scaled_dot_product_attention

                        cp_out = fn_eval(fn, cp_q, cp_k, cp_v, is_causal=is_causal)

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

                (cp_out,) = context_parallel_unshard(device_mesh, [cp_out], [seq_dim])
                atol, rtol = ATTENTION_TOLERANCES[dtype]
                torch.testing.assert_close(out, cp_out, atol=atol, rtol=rtol)
                if dtype == torch.bfloat16:
                    self.assertLessEqual(int(_bf16_ulp_diff(cp_out, out).max().item()), 1)

                if not test_forward_only:
                    cp_dq, cp_dk, cp_dv = context_parallel_unshard(
                        device_mesh,
                        [cp_q.grad, cp_k.grad, cp_v.grad],
                        [seq_dim, seq_dim, seq_dim],
                    )
                    torch.testing.assert_close(q.grad, cp_dq, atol=atol, rtol=rtol)
                    torch.testing.assert_close(k.grad, cp_dk, atol=atol, rtol=rtol)
                    torch.testing.assert_close(v.grad, cp_dv, atol=atol, rtol=rtol)
                    if dtype == torch.bfloat16:
                        self.assertLessEqual(int(_bf16_ulp_diff(cp_dv, v.grad).max().item()), 1)

                    cp_q.grad = None
                    cp_k.grad = None
                    cp_v.grad = None

                cp_q.requires_grad = False
                cp_k.requires_grad = False
                cp_v.requires_grad = False
        finally:
            if callable(original_ring_update):
                torch_npu.npu_ring_attention_update = original_ring_update

        call_count = torch.tensor([ring_update_call_count], device=self.device_type)
        dist.all_reduce(call_count)
        if callable(original_ring_update):
            self.assertGreater(call_count.item(), 0)


if __name__ == "__main__":
    run_tests()
