import os
import logging

import torch
import torch_npu
from torch.distributed.tensor import DTensor, Shard

from torch.distributed.tensor.experimental._context_parallel._attention import (
    _cp_options,
    _templated_ring_attention,
    _templated_ring_attention_backward,
)

logger = logging.getLogger("torch.distributed._context_parallel")

# ============================================================================
# npu_fusion_attention_v3 param index
#
# Forward:
#   0:query 1:key 2:value 3:head_num 4:input_layout 5:pse 6:padding_mask
#   7:atten_mask 8:scale 9:keep_prob 10:pre_tockens 11:next_tockens
#   12:inner_precise 13:prefix 14:actual_seq_qlen 15:actual_seq_kvlen
#   16:sparse_mode 17:gen_mask_parallel 18:sync 19:softmax_layout 20:sink
#
# Backward:
#   0:query 1:key 2:value 3:dy 4:head_num 5:input_layout 6:pse 7:padding_mask
#   8:atten_mask 9:softmax_max 10:softmax_sum 11:softmax_in 12:attention_in
#   13:scale_value 14:keep_prob 15:pre_tockens 16:next_tockens 17:inner_precise
#   18:seed 19:offset 20:prefix 21:actual_seq_qlen 22:actual_seq_kvlen
#   23:sparse_mode 24:gen_mask_parallel 25:sync 26:softmax_layout 27:sink
# ============================================================================

_FWD_IX = dict(
    head_num=3, input_layout=4, pse=5, padding_mask=6, atten_mask=7,
    scale=8, keep_prob=9, pre_tockens=10, next_tockens=11, inner_precise=12,
    prefix=13, actual_seq_qlen=14, actual_seq_kvlen=15, sparse_mode=16,
    gen_mask_parallel=17, sync=18, softmax_layout=19, sink=20,
)

_BWD_IX = dict(
    head_num=4, input_layout=5, pse=6, padding_mask=7, atten_mask=8,
    softmax_max=9, softmax_sum=10, softmax_in=11, attention_in=12,
    scale_value=13, keep_prob=14, pre_tockens=15, next_tockens=16,
    inner_precise=17, seed=18, offset=19, prefix=20, actual_seq_qlen=21,
    actual_seq_kvlen=22, sparse_mode=23, gen_mask_parallel=24, sync=25,
    softmax_layout=26, sink=27,
)

def _get(args, ix: dict, name: str, default=None):
    """Get value from args by name via index table; returns default if out of bounds."""
    i = ix[name]
    return args[i] if len(args) > i else default

def _validate_bnsd_layout(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    input_layout,
    *,
    op_name: str,
) -> None:
    """Fail fast for layouts unsupported by the current CP ring path."""
    layout = input_layout.upper() if isinstance(input_layout, str) else input_layout
    if layout != "BNSD":
        raise NotImplementedError(
            f"{op_name} currently supports BNSD q/k/v only in NPU context parallel, "
            f"got input_layout={input_layout!r} with "
            f"q={tuple(query.shape)} k={tuple(key.shape)} v={tuple(value.shape)}"
        )
    if query.dim() != 4 or key.dim() != 4 or value.dim() != 4:
        raise NotImplementedError(
            f"{op_name} currently expects 4D BNSD q/k/v in NPU context parallel, "
            f"got q.dim={query.dim()} k.dim={key.dim()} v.dim={value.dim()} with "
            f"q={tuple(query.shape)} k={tuple(key.shape)} v={tuple(value.shape)}"
        )

# Passthrough param names: not controlled by ring attention, forwarded to every step
_PASSTHROUGH_NAMES = [
    "head_num", "input_layout", "pre_tockens", "next_tockens",
    "inner_precise", "gen_mask_parallel", "sync", "softmax_layout",
]

def _extract_passthrough(args, ix: dict) -> dict:
    """Extract passthrough params from args by index table; skips None values."""
    pt = {}
    for name in _PASSTHROUGH_NAMES:
        v = _get(args, ix, name)
        if v is not None:
            pt[name] = v
    return pt

_UNSUPPORTED_CP_PASSTHROUGH_NAMES = (
    "pse",
    "padding_mask",
    "prefix",
    "actual_seq_qlen",
    "actual_seq_kvlen",
    "sink",
)

def _is_present(value) -> bool:
    if value is None:
        return False
    if isinstance(value, (list, tuple)) and len(value) == 0:
        return False
    return True

def _validate_cp_passthrough_args(args, ix: dict, *, op_name: str) -> None:
    """Reject per-sequence/per-logit inputs that are not ring-step transformed yet."""
    unsupported = [
        name
        for name in _UNSUPPORTED_CP_PASSTHROUGH_NAMES
        if _is_present(_get(args, ix, name))
    ]
    if unsupported:
        raise NotImplementedError(
            f"{op_name} in NPU context parallel does not support "
            f"{', '.join(unsupported)} yet. These inputs are tied to global "
            "sequence positions or attention logits, so they must be sliced and/or "
            "rotated together with q/k/v for each ring step."
        )

    softmax_layout = _get(args, ix, "softmax_layout", "")
    if softmax_layout not in (None, ""):
        raise NotImplementedError(
            f"{op_name} in NPU context parallel currently supports the default "
            f"BNSD softmax layout only, got softmax_layout={softmax_layout!r}."
        )

def _validate_cp_sparse_args(args, ix: dict, *, op_name: str) -> None:
    """Keep mask semantics limited to the ring path we actually transform."""
    sparse_mode = _get(args, ix, "sparse_mode", 0)
    atten_mask = _get(args, ix, "atten_mask")
    if sparse_mode not in (0, 1, 2, 3):
        raise NotImplementedError(
            f"{op_name} in NPU context parallel currently supports only full "
            f"attention and causal sparse modes 1/2/3, got sparse_mode={sparse_mode!r}."
        )
    if _is_present(atten_mask) and sparse_mode not in (1, 2, 3):
        raise NotImplementedError(
            f"{op_name} in NPU context parallel does not support arbitrary "
            f"atten_mask yet. Pass causal sparse_mode 1/2/3 for causal attention; "
            f"got sparse_mode={sparse_mode!r}."
        )

# ============================================================================
# Global stack: forward pushes step_caches, backward pops them.
#
# C++ autograd engine runs backward on different threads (verified via tid mismatch),
# so threading.local() cannot be used. A module-level list works because:
# - Python GIL guarantees thread safety
# - LIFO order matches autograd reverse order (last forward → first backward)
# - Each rank is an independent process
# ============================================================================

_step_cache_stack: list = []

# ============================================================================
# Format Conversion: softmax_max/sum → logsumexp
# ============================================================================

def _convert_softmax_to_logsumexp(
    softmax_max: torch.Tensor,
    softmax_sum: torch.Tensor,
) -> torch.Tensor:
    """npu_fusion_attention softmax_max/sum [B,N,S,8] → logsumexp [B,N,S]。

    slot 0: lse = max + log(sum(exp(x - max)))。
    """
    sm_max = softmax_max[:, :, :, 0].float()
    sm_sum = softmax_sum[:, :, :, 0].float()
    return sm_max + torch.log(sm_sum + 1e-10)

def _get_ring_attention_update():
    ring_update = getattr(torch_npu, "npu_ring_attention_update", None)
    return ring_update if callable(ring_update) else None

def _get_softmax_merge_impl() -> str:
    if _get_ring_attention_update() is not None:
        return "op"
    return "python"

def _bnsd_to_sbh(attn_out: torch.Tensor) -> torch.Tensor:
    """Map BNSD attention output to the SBH layout expected by ring_update."""
    B, N, S, D = attn_out.shape
    return attn_out.permute(2, 0, 1, 3).contiguous().view(S, B, N * D)

def _sbh_to_bnsd(attn_out: torch.Tensor, *, head_num: int) -> torch.Tensor:
    """Map SBH attention output back to the CP main-path BNSD layout."""
    S, B, H = attn_out.shape
    if H % head_num != 0:
        raise RuntimeError(
            f"Cannot convert SBH attention back to BNSD: hidden={H} is not divisible "
            f"by head_num={head_num}"
        )
    D = H // head_num
    return attn_out.view(S, B, head_num, D).permute(1, 2, 0, 3).contiguous()

def _merge_softmax_stats_python(
    prev_softmax_max: torch.Tensor,
    prev_softmax_sum: torch.Tensor,
    cur_softmax_max: torch.Tensor,
    cur_softmax_sum: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference merge in Python, keeping the current BNSD/BNS8 main path."""
    prev_max = prev_softmax_max.float() if _cp_options.convert_to_f32 else prev_softmax_max
    prev_sum = prev_softmax_sum.float() if _cp_options.convert_to_f32 else prev_softmax_sum
    cur_max = cur_softmax_max.float() if _cp_options.convert_to_f32 else cur_softmax_max
    cur_sum = cur_softmax_sum.float() if _cp_options.convert_to_f32 else cur_softmax_sum

    new_max = torch.maximum(prev_max, cur_max)
    prev_scale = torch.exp(prev_max.float() - new_max.float())
    cur_scale = torch.exp(cur_max.float() - new_max.float())
    new_sum = prev_sum.float() * prev_scale + cur_sum.float() * cur_scale
    return new_max.to(prev_max.dtype), new_sum.to(prev_sum.dtype)

def _merge_softmax_stats_with_ring_update(
    prev_attn_out: torch.Tensor,
    prev_softmax_max: torch.Tensor,
    prev_softmax_sum: torch.Tensor,
    cur_attn_out: torch.Tensor,
    cur_softmax_max: torch.Tensor,
    cur_softmax_sum: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Use torch_npu.npu_ring_attention_update with a thin BNSD <-> SBH adapter."""
    ring_update = _get_ring_attention_update()
    if ring_update is None:
        raise RuntimeError(
            "The op softmax merge path requires torch_npu.npu_ring_attention_update, "
            "but the API is not available in the current torch_npu build."
        )

    merged_attn_sbh, merged_max, merged_sum = ring_update(
        _bnsd_to_sbh(prev_attn_out),
        prev_softmax_max.float(),
        prev_softmax_sum.float(),
        _bnsd_to_sbh(cur_attn_out),
        cur_softmax_max.float(),
        cur_softmax_sum.float(),
        input_layout="SBH",
    )
    merged_attn = _sbh_to_bnsd(merged_attn_sbh, head_num=prev_attn_out.size(1))
    return merged_attn, merged_max, merged_sum

def _make_forward_op(step_caches: list, *, pt: dict):
    """Create a per-step forward op closure for ring attention.

    step_caches (out param): list mutated in-place — each call appends
        (merged_max, merged_sum, seed, offset) for the current step.
    pt: passthrough dict of params that do not vary across ring steps
        (head_num, input_layout, pre_tockens, next_tockens, etc.).
    """

    merged_max = None
    merged_sum = None
    merged_attn = None
    merge_impl = _get_softmax_merge_impl()

    def _op(query, key, value, *, is_causal=False, dropout_p=0.0, scale=None):
        nonlocal merged_max, merged_sum, merged_attn

        B, N, S, D = query.shape
        softmax_scale = scale if scale is not None else (1.0 / D**0.5)

        atten_mask = None
        sparse_mode = 0
        if is_causal:
            atten_mask = torch.triu(
                torch.ones(S, S, dtype=torch.uint8, device=query.device), diagonal=1
            ).unsqueeze(0).unsqueeze(0)
            sparse_mode = 1

        (attention_score, softmax_max, softmax_sum, _softmax_out,
         seed, offset, _numels) = torch_npu.npu_fusion_attention(
            query, key, value,
            head_num=pt.get("head_num", N),
            input_layout=pt.get("input_layout", "BNSD"),
            scale=softmax_scale,
            atten_mask=atten_mask,
            sparse_mode=sparse_mode,
            keep_prob=1.0 - dropout_p,
            pse=None,
            padding_mask=None,
            pre_tockens=pt.get("pre_tockens", 2147483647),
            next_tockens=pt.get("next_tockens", 2147483647),
            inner_precise=pt.get("inner_precise", 0),
            prefix=None,
            actual_seq_qlen=None,
            actual_seq_kvlen=None,
            gen_mask_parallel=pt.get("gen_mask_parallel", True),
            sync=pt.get("sync", False),
            softmax_layout=pt.get("softmax_layout", ""),
            sink=None,
        )

        # Online softmax merge: fold current-step stats into the running accumulator.
        # When _cp_options.convert_to_f32 is set, merge runs entirely in fp32.
        if merged_max is None:
            merged_attn = attention_score.detach()
            merged_max = softmax_max.float() if _cp_options.convert_to_f32 else softmax_max
            merged_sum = softmax_sum.float() if _cp_options.convert_to_f32 else softmax_sum
        elif softmax_max.shape[2] != merged_max.shape[2]:
            # IS_CAUSAL partial: _templated_ring_attention uses query.chunk(2, dim=2)[1],
            # so the current step only handles the trailing half of Q positions.
            S_half = softmax_max.shape[2]
            prev_half_max = merged_max[:, :, S_half:, :]
            prev_half_sum = merged_sum[:, :, S_half:, :]
            if merge_impl == "op":
                if merged_attn is None:
                    raise RuntimeError(
                        "merged_attn is None in IS_CAUSAL partial merge with "
                        "merge_impl='op'; this indicates a logic error in forward step merging."
                    )
                prev_half_attn = merged_attn[:, :, S_half:, :]
                merged_half_attn, new_half_max, new_half_sum = _merge_softmax_stats_with_ring_update(
                    prev_half_attn,
                    prev_half_max,
                    prev_half_sum,
                    attention_score.detach(),
                    softmax_max,
                    softmax_sum,
                )
                merged_attn = torch.cat([
                    merged_attn[:, :, :S_half, :],
                    merged_half_attn.detach(),
                ], dim=2)
            else:
                new_half_max, new_half_sum = _merge_softmax_stats_python(
                    prev_half_max,
                    prev_half_sum,
                    softmax_max,
                    softmax_sum,
                )

            merged_max = torch.cat([
                merged_max[:, :, :S_half, :],
                new_half_max,
            ], dim=2)
            merged_sum = torch.cat([
                merged_sum[:, :, :S_half, :],
                new_half_sum,
            ], dim=2)
        else:
            if merge_impl == "op":
                # merged_attn not used in ring attention
                if merged_attn is None:
                    raise RuntimeError(
                        "merged_attn is None with merge_impl='op'; "
                        "this indicates a logic error in forward step merging."
                    )
                merged_attn, merged_max, merged_sum = _merge_softmax_stats_with_ring_update(
                    merged_attn,
                    merged_max,
                    merged_sum,
                    attention_score.detach(),
                    softmax_max,
                    softmax_sum,
                )
                merged_attn = merged_attn.detach()
            else:
                merged_max, merged_sum = _merge_softmax_stats_python(
                    merged_max,
                    merged_sum,
                    softmax_max,
                    softmax_sum,
                )

        # Push merged softmax stats (cloned to guard against later steps), seed, offset.
        # Per-step attention_score is not saved; backward uses the merged output via _bop(out).
        step_caches.append((merged_max.clone(), merged_sum.clone(), seed, offset))

        # Return per-step logsumexp (fed to the native merger for attention output), not the merged version.
        logsumexp = _convert_softmax_to_logsumexp(softmax_max, softmax_sum)

        return (
            attention_score,
            logsumexp,
            None,  # cum_seq_q
            None,  # cum_seq_k
            S,     # max_q
            S,     # max_k
            torch.tensor(seed, dtype=torch.int64, device=query.device),
            None,  # unused
            torch.empty(0, device=query.device),  # debug_attn_mask
        )

    return _op


def _make_backward_op(step_caches: list, *, pt: dict):
    """Create a per-step backward op closure for ring attention.

    step_caches: the same list that _make_forward_op populated — consumed in LIFO order.
        step_caches[i] = (merged_max, merged_sum, seed, offset) for step i;
        step_caches[-1][:2] holds the final globally-normalized stats for the backward kernel.
    pt: passthrough dict (same semantics as _make_forward_op).
    """

    idx_box = [0]

    def _bop(grad_out, query, key, value, out, logsumexp,
             cum_seq_q, cum_seq_k, max_q, max_k,
             dropout_p, is_causal,
             philox_seed, philox_offset,
             *, scale=None):
        N = query.size(1)
        D = query.size(-1)
        S = query.size(2)
        softmax_scale = scale if scale is not None else (1.0 / D**0.5)

        i = idx_box[0]
        # step_caches[i] holds the running merged state at step i (grows over ring rounds).
        # Backward needs the final globally-normalized stats, so take step_caches[-1][:2].
        _, _, seed, offset = step_caches[i]
        merged_max, merged_sum = step_caches[-1][:2]
        idx_box[0] += 1

        # IS_CAUSAL partial: native backward already sliced query/out/grad_out to chunk(2)[1].
        # Slice merged_max/sum to the corresponding trailing half; out is already sliced.
        if query.size(2) != merged_max.size(2):
            S_half = query.size(2)
            merged_max = merged_max[:, :, S_half:, :]
            merged_sum = merged_sum[:, :, S_half:, :]

        atten_mask = None
        sparse_mode = 0
        if is_causal:
            atten_mask = torch.triu(
                torch.ones(S, S, dtype=torch.uint8, device=query.device), diagonal=1
            ).unsqueeze(0).unsqueeze(0)
            sparse_mode = 1

        grads = torch_npu.npu_fusion_attention_grad(
            query, key, value,
            dy=grad_out,
            head_num=pt.get("head_num", N),
            input_layout=pt.get("input_layout", "BNSD"),
            softmax_max=merged_max,
            softmax_sum=merged_sum,
            attention_in=out,
            scale_value=softmax_scale,
            keep_prob=1.0 - dropout_p,
            atten_mask=atten_mask,
            sparse_mode=sparse_mode,
            seed=seed,
            offset=offset,
            pse=None,
            padding_mask=None,
            pre_tockens=pt.get("pre_tockens", 2147483647),
            next_tockens=pt.get("next_tockens", 2147483647),
            inner_precise=pt.get("inner_precise", 0),
            numels=0,
        )
        return grads[0], grads[1], grads[2]

    return _bop


# ============================================================================
# Common: DTensor unwrap helper
# ============================================================================

def _unwrap_args(args):
    """Unwrap all DTensors in args to _local_tensor; return (local_args, mesh)."""
    local_args = []
    mesh = None
    for i, a in enumerate(args):
        if isinstance(a, DTensor):
            local_args.append(a._local_tensor)
            if mesh is None:
                mesh = a.device_mesh
        elif isinstance(a, torch.Tensor):
            local_args.append(a)
        else:
            local_args.append(a)
    return local_args, mesh


# ============================================================================
# Forward DTensor Handler — intercepts npu_fusion_attention_v3
# ============================================================================
def _npu_fa_v3_handler(op_call, args, kwargs):
    """Intercept npu_fusion_attention_v3, run ring attention, return a v3 6-tuple.

    Unwraps DTensor args manually (standard unwrap_to_op_info rejects mixed
    DTensor/plain-tensor args). Delegates directly to _templated_ring_attention.
    Pushes per-step caches onto the global stack for backward consumption.
    """
    local_args, mesh = _unwrap_args(args)
    if mesh is None:
        raise RuntimeError("No DTensor found in _npu_fa_v3_handler args")

    query, key, value = local_args[0], local_args[1], local_args[2]

    # Ring-attention-controlled params: extract causal semantics and scale from args
    sparse_mode = _get(local_args, _FWD_IX, "sparse_mode", 0)
    scale = _get(local_args, _FWD_IX, "scale")
    keep_prob = _get(local_args, _FWD_IX, "keep_prob", 1.0)
    input_layout = _get(local_args, _FWD_IX, "input_layout", "BNSD")

    logger.debug(
        "CP forward handler: q=%s k=%s v=%s mesh=%s sparse_mode=%s scale=%s keep_prob=%s",
        tuple(query.shape), tuple(key.shape), tuple(value.shape),
        mesh, sparse_mode, scale, keep_prob,
    )

    _validate_bnsd_layout(
        query,
        key,
        value,
        input_layout,
        op_name="npu_fusion_attention_v3",
    )
    _validate_cp_passthrough_args(
        local_args,
        _FWD_IX,
        op_name="npu_fusion_attention_v3",
    )
    _validate_cp_sparse_args(
        local_args,
        _FWD_IX,
        op_name="npu_fusion_attention_v3",
    )

    is_causal = sparse_mode in (1, 2, 3)
    dropout_p = (1.0 - keep_prob) if isinstance(keep_prob, (int, float)) else 0.0
    softmax_scale = scale if scale is not None else (1.0 / query.shape[-1] ** 0.5)
    pt = _extract_passthrough(local_args, _FWD_IX)

    step_caches: list = []
    op = _make_forward_op(step_caches, pt=pt)
    group = mesh.get_group()
    result = _templated_ring_attention(
        group,
        seq_dim=2,
        op=op,
        query=query,
        key=key,
        value=value,
        is_causal=is_causal,
        dropout_p=dropout_p,
        scale=softmax_scale,
    )
    attn_output = result[0]
    merged_lse = result[1]

    # Push (step_caches, is_causal, merged_out, merged_lse) for backward handler.
    # Backward needs merged_out to compute D = sum(dout * O_merged); per-step raw output is not enough.
    _step_cache_stack.append(
        (step_caches, is_causal, attn_output.detach(), merged_lse.detach())
    )

    # Build v3 6-tuple. softmax_max/sum come from the final step; op_plugin typically only uses attn_output.
    B, N, S, D = attn_output.shape
    dev = attn_output.device
    if step_caches:
        sm_max, sm_sum, _, _ = step_caches[-1]
        softmax_max = sm_max
        softmax_sum = sm_sum
    else:
        softmax_max = torch.zeros(B, N, S, 8, dtype=torch.float32, device=dev)
        softmax_sum = torch.zeros(B, N, S, 8, dtype=torch.float32, device=dev)

    return (
        attn_output,
        softmax_max,
        softmax_sum,
        torch.zeros(0, device=dev),                                  # softmax_out
        torch.tensor([0], dtype=torch.int64, device="cpu"),          # seed
        torch.tensor([0], dtype=torch.int64, device="cpu"),          # offset
    )


# ============================================================================
# Backward DTensor Handler — intercepts npu_fusion_attention_grad_v3
# ============================================================================

def _npu_fa_grad_v3_handler(op_call, args, kwargs):
    """Intercept npu_fusion_attention_grad_v3, run ring attention backward, return DTensor grads.

    Pops per-step caches from the global stack (pushed by the forward handler).
    Delegates to _templated_ring_attention_backward, then wraps grad_q/k/v as
    DTensors with Shard(2). The softmax stats saved by AutogradNPU's backward
    node are placeholders; the real per-step caches come from _step_cache_stack.
    """
    local_args, mesh = _unwrap_args(args)
    if mesh is None:
        raise RuntimeError("No DTensor found in npu_fusion_attention_grad_v3 args")

    query, key, value, dy = local_args[0], local_args[1], local_args[2], local_args[3]
    input_layout = _get(local_args, _BWD_IX, "input_layout", "BNSD")
    _validate_bnsd_layout(
        query,
        key,
        value,
        input_layout,
        op_name="npu_fusion_attention_grad_v3",
    )
    _validate_cp_passthrough_args(
        local_args,
        _BWD_IX,
        op_name="npu_fusion_attention_grad_v3",
    )
    _validate_cp_sparse_args(
        local_args,
        _BWD_IX,
        op_name="npu_fusion_attention_grad_v3",
    )
    scale_value = _get(local_args, _BWD_IX, "scale_value")
    keep_prob = _get(local_args, _BWD_IX, "keep_prob", 1.0)

    dropout_p = (1.0 - keep_prob) if isinstance(keep_prob, (int, float)) else 0.0
    if scale_value is not None:
        softmax_scale = float(scale_value) if isinstance(scale_value, (int, float)) else scale_value
    else:
        softmax_scale = 1.0 / query.shape[-1] ** 0.5

    stack = _step_cache_stack
    if not stack:
        raise RuntimeError(
            "step_cache_stack is empty in backward handler; "
            "forward caches were never pushed or already consumed."
        )
    step_caches, is_causal, merged_out, merged_lse = stack.pop()

    pt = _extract_passthrough(local_args, _BWD_IX)

    bop = _make_backward_op(step_caches, pt=pt)
    group = mesh.get_group()
    zero = torch.zeros(0, device=query.device)
    result = _templated_ring_attention_backward(
        group,
        seq_dim=2,
        op=bop,
        grad_out=dy,
        grad_out_name="grad_out",
        query=query,
        key=key,
        value=value,
        out=merged_out,
        logsumexp=merged_lse,
        is_causal=is_causal,
        cum_seq_q=zero,
        cum_seq_k=zero,
        max_q=query.size(2),
        max_k=key.size(2),
        dropout_p=dropout_p,
        philox_seed=zero,
        philox_offset=zero,
        scale=softmax_scale,
    )
    grad_q, grad_k, grad_v = result[0], result[1], result[2]

    grad_q_dt = DTensor.from_local(grad_q, mesh, [Shard(2)], run_check=False)
    grad_k_dt = DTensor.from_local(grad_k, mesh, [Shard(2)], run_check=False)
    grad_v_dt = DTensor.from_local(grad_v, mesh, [Shard(2)], run_check=False)

    dev = query.device
    grad_pse = torch.zeros(0, device=dev)
    grad_sink = torch.zeros(0, device=dev)
    return (grad_q_dt, grad_k_dt, grad_v_dt, grad_pse, grad_sink)


# ============================================================================
# CP Sharding Rule
# ============================================================================

def _scaled_dot_product_attention_cp_strategy(op_schema):
    """CP strategy: Shard(2) on q/k/v and output. Hardcoded dim=2 assumes BNSD layout."""
    from torch.distributed.tensor._ops.utils import (
        expand_to_full_mesh_op_strategy as _expand,
    )

    mesh = op_schema.get_mesh_from_args()
    cp_strategy = [
        Shard(2),  # output
        Shard(2),  # query
        Shard(2),  # key
        Shard(2),  # value
    ]
    return _expand(mesh, op_schema, [cp_strategy], input_index=1)


_npu_fa = torch.ops.npu.npu_fusion_attention_v3.default
_npu_fa_grad = torch.ops.npu.npu_fusion_attention_grad_v3.default

_npu_custom_ops = {
    _npu_fa: _npu_fa_v3_handler,
    _npu_fa_grad: _npu_fa_grad_v3_handler,
}

# ============================================================================
# CP Dispatcher Enable/Disable
# ============================================================================
def npu_enable_cp_dtensor_dispatcher() -> None:
    """Register NPU SDPA forward/backward handlers and CP sharding rules."""
    logger.info(f"registering handler keys={[str(k) for k in _npu_custom_ops.keys()]}")

    existing = DTensor._op_dispatcher._custom_op_handlers.copy()
    DTensor._op_dispatcher._custom_op_handlers = {**existing, **_npu_custom_ops}

    from torch.distributed.tensor.experimental._context_parallel._sharding_rules import (
        register_cp_sharding_rules,
    )
    register_cp_sharding_rules()

    from torch.distributed.tensor._ops.utils import register_op_strategy
    from torch.distributed.tensor._op_schema import RuntimeSchemaInfo
    register_op_strategy(
        _npu_fa,
        schema_info=RuntimeSchemaInfo(1),
    )(_scaled_dot_product_attention_cp_strategy)


def npu_disable_cp_dtensor_dispatcher() -> None:
    """Remove NPU handlers and unregister CP sharding rules."""
    logger.info(f"removing handler keys={[str(k) for k in _npu_custom_ops.keys()]}")

    DTensor._op_dispatcher._custom_op_handlers = {
        k: v
        for k, v in DTensor._op_dispatcher._custom_op_handlers.items()
        if k not in _npu_custom_ops
    }

    from torch.distributed.tensor.experimental._context_parallel._sharding_rules import (
        unregister_cp_sharding_rules,
    )
    unregister_cp_sharding_rules(clear_the_cache=False)

    if _step_cache_stack:
        logger.warning(
            "CP dispatcher disabled with step_cache_stack depth=%d; "
            "some forward caches were not consumed by backward.",
            len(_step_cache_stack),
        )
    _step_cache_stack.clear()
