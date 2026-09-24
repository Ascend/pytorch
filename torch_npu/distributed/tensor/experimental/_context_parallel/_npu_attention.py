import logging

import torch
import torch_npu
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor, Shard, Replicate
from torch_npu.distributed.tensor._attention import (
    npu_fusion_attention_v3_strategy as _layer1_fwd_strategy,
    npu_fusion_attention_grad_v3_strategy as _layer1_bwd_strategy,
)

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

def _get_kw(args, kwargs, ix: dict, name: str, default=None):
    """Get value from strategy args (by index) or kwargs (by name).

    Strategy functions receive *args, **kwargs from register_op_strategy.
    Positional params are in args (indexed by _FWD_IX/_BWD_IX), keyword-only
    params are in kwargs. This helper checks both.
    """
    i = ix[name]
    if len(args) > i:
        return args[i]
    return kwargs.get(name, default)

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

def _extract_passthrough_kwargs(kwargs: dict) -> dict:
    """Extract passthrough params from kwargs (backward op has keyword-only args)."""
    pt = {}
    for name in _PASSTHROUGH_NAMES:
        v = kwargs.get(name)
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

def _validate_cp_passthrough_args_kwargs(kwargs: dict, *, op_name: str) -> None:
    """Reject per-sequence/per-logit inputs (kwargs variant for backward op)."""
    unsupported = [name for name in _UNSUPPORTED_CP_PASSTHROUGH_NAMES if _is_present(kwargs.get(name))]
    if unsupported:
        raise NotImplementedError(
            f"{op_name} in NPU context parallel does not support "
            f"{', '.join(unsupported)} yet. These inputs are tied to global "
            "sequence positions or attention logits, so they must be sliced and/or "
            "rotated together with q/k/v for each ring step."
        )
    softmax_layout = kwargs.get("softmax_layout", "")
    if softmax_layout not in (None, ""):
        raise NotImplementedError(
            f"{op_name} in NPU context parallel currently supports the default "
            f"BNSD softmax layout only, got softmax_layout={softmax_layout!r}."
        )

def _validate_cp_sparse_args_kwargs(kwargs: dict, *, op_name: str) -> None:
    """Keep mask semantics limited (kwargs variant for backward op)."""
    sparse_mode = kwargs.get("sparse_mode", 0)
    atten_mask = kwargs.get("atten_mask")
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

def _get_cp_group(mesh: DeviceMesh, args_schema: tuple, tensor_indices: list[int], seq_dim: int):
    """Get the ProcessGroup for the CP ring, compatible with multi-dim DeviceMesh.

    Scans the given arg spec positions for a Shard(seq_dim) placement to
    identify the CP mesh dimension, then returns the corresponding
    ProcessGroup via mesh.get_group(mesh_dim).

    For 1-D mesh (the common CP case), this is equivalent to mesh.get_group().
    For multi-dim mesh (e.g. TP x CP), it finds the mesh dim that carries the
    sequence shard and returns that dim's group.

    Falls back to mesh.get_group() (no args) for 1-D mesh, matching the
    native PyTorch _sdpa_handler behavior.
    """
    if mesh.ndim == 1:
        return mesh.get_group()

    # For multi-dim mesh, find which mesh dim carries Shard(seq_dim).
    # args_schema comes from op_info.schema.args_schema, which contains
    # DTensorSpec objects (with .placements) for each DTensor arg.
    from torch.distributed.tensor._dtensor_spec import DTensorSpec

    for idx in tensor_indices:
        if idx < len(args_schema):
            spec = args_schema[idx]
            if isinstance(spec, DTensorSpec):
                for mesh_dim, placement in enumerate(spec.placements):
                    if placement.is_shard() and placement.dim == seq_dim:
                        return mesh.get_group(mesh_dim)

    # No Shard(seq_dim) found -- fall back to default (will raise if ndim > 1,
    # matching native behavior).
    return mesh.get_group()


# NOTE: The former global stack _step_cache_stack is removed. Softmax stats now
# flow through op autograd (forward returns merged softmax_max/sum in the v3
# 6-tuple; native autograd saves them and passes to the backward handler via
# kwargs). This mirrors native torch CP (merge + chunk + single philox), is
# compile/PP friendly, and needs no module-level state.

# ============================================================================
# Format Conversion: softmax_max/sum -> logsumexp
# ============================================================================

def _convert_softmax_to_logsumexp(
    softmax_max: torch.Tensor,
    softmax_sum: torch.Tensor,
) -> torch.Tensor:
    """npu_fusion_attention softmax_max/sum [B,N,S,8] -> logsumexp [B,N,S].

    slot 0: lse = max + log(sum(exp(x - max))).
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

    step_caches (out param): list mutated in-place -- each call appends
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

        # npu_fusion_attention_v3: graph-mode-friendly (Tensor seed/offset, no numels).
        # v1 returns int seed/offset/numels which graph mode can optimize away under dropout.
        (attention_score, softmax_max, softmax_sum, _softmax_out,
         seed, offset) = torch_npu.npu_fusion_attention_v3(
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
                # merged_attn is only used for op-based merge, not returned by ring attention
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

        # Save merged softmax stats (cloned to guard against later steps), seed, offset.
        # The forward handler extracts the final merged stats from step_caches[-1] for the
        # output 6-tuple. Backward receives stats via autograd kwargs, not from step_caches.
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
            seed.to(torch.int64) if isinstance(seed, torch.Tensor) else torch.tensor(seed, dtype=torch.int64, device=query.device),
            None,  # unused
            torch.empty(0, device=query.device),  # debug_attn_mask
        )

    return _op


def _make_backward_op(merged_max, merged_sum, seed, offset, *, pt: dict):
    """Per-step backward op closure for ring attention (global-stack-free).

    Reads final merged softmax_max/sum (from autograd kwargs) and a single
    seed/offset (like native CP's single philox). The closure receives
    already-chunked query/out/grad_out from _templated_ring_attention_backward;
    for IS_CAUSAL partial steps it slices merged_max/sum to match the chunk.

    Args:
        merged_max, merged_sum: final merged softmax stats (full seq), autograd-flowed.
        seed, offset: single rng tensors (from forward's last step), autograd-flowed.
        pt: passthrough dict (head_num, input_layout, pre/next_tockens, ...).
    """

    def _bop(grad_out, query, key, value, out, logsumexp,
             cum_seq_q, cum_seq_k, max_q, max_k,
             dropout_p, is_causal,
             philox_seed, philox_offset,
             *, scale=None):
        N = query.size(1)
        D = query.size(-1)
        S = query.size(2)
        softmax_scale = scale if scale is not None else (1.0 / D**0.5)

        # IS_CAUSAL partial: _templated_ring_attention_backward already chunked
        # query/out/grad_out to the trailing half of Q. Slice merged_max/sum to match.
        cur_max = merged_max
        cur_sum = merged_sum
        if query.size(2) != cur_max.size(2):
            S_half = query.size(2)
            cur_max = cur_max[:, :, S_half:, :]
            cur_sum = cur_sum[:, :, S_half:, :]

        atten_mask = None
        sparse_mode = 0
        if is_causal:
            atten_mask = torch.triu(
                torch.ones(S, S, dtype=torch.uint8, device=query.device), diagonal=1
            ).unsqueeze(0).unsqueeze(0)
            sparse_mode = 1

        # v3 grad: accepts Tensor seed/offset (no numels). seed/offset are single
        # values flowed through autograd (same for all ring steps, like native CP).
        grads = torch_npu.npu_fusion_attention_grad_v3(
            query, key, value,
            dy=grad_out,
            head_num=pt.get("head_num", N),
            input_layout=pt.get("input_layout", "BNSD"),
            softmax_max=cur_max,
            softmax_sum=cur_sum,
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
        )
        return grads[0], grads[1], grads[2]

    return _bop


# ============================================================================
# Forward DTensor Handler -- intercepts npu_fusion_attention_v3
# ============================================================================
def _npu_fa_v3_handler(op_call, args, kwargs):
    """Intercept npu_fusion_attention_v3, run ring attention, wrap via propagated spec.

    Mirrors native _sdpa_handler: unwrap_to_op_info -> propagate -> ring attention ->
    wrap(local_results, output_spec). No from_local is used inside the handler:
    from_local goes through the _FromTorchTensor autograd.Function, which re-dispatches
    internal aten ops (e.g. detach_) during dynamo/compiled_autograd tracing -- those ops
    have no sharding strategy and break compile. Output DTensors are built via the direct
    constructor OpDispatcher.wrap, exactly like native _sdpa_handler and layer1's
    _npu_fusion_attention_handler.

    Plain non-scalar tensor args (e.g. the atten_mask generated inside the C++ SDPA
    kernel, which never passes through attention_input_fn) are handled by the global
    _allow_implicit_replication flag toggled in npu_enable_cp_dtensor_dispatcher, so
    unwrap_to_op_info auto-replicates them instead of raising 'mixed Tensor/DTensor'.
    """
    op_info = DTensor._op_dispatcher.unwrap_to_op_info(op_call, args, kwargs)

    # sharding propagation -> output_spec (matches native _sdpa_handler)
    DTensor._op_dispatcher.sharding_propagator.propagate(op_info)
    output_sharding = op_info.output_sharding
    if output_sharding is None:
        raise RuntimeError("output sharding should not be None")
    if output_sharding.needs_redistribute:
        raise RuntimeError("inputs need to be redistributed")

    mesh = op_info.compute_mesh
    local_args = op_info.local_args

    query, key, value = local_args[0], local_args[1], local_args[2]

    # Ring-attention-controlled params: extract causal semantics and scale from args
    sparse_mode = _get(local_args, _FWD_IX, "sparse_mode", 0)
    scale = _get(local_args, _FWD_IX, "scale")
    keep_prob = _get(local_args, _FWD_IX, "keep_prob", 1.0)
    input_layout = _get(local_args, _FWD_IX, "input_layout", "BNSD")

    # Fix head_num for TP: original head_num is the global value, but after TP
    # sharding the local query has fewer heads. Recalculate from local query shape.
    # This mirrors layer-1 _npu_fusion_attention_handler's head_num correction.
    if input_layout and 'N' in input_layout:
        head_dim_idx = input_layout.index('N')
        local_args = list(local_args)
        local_args[_FWD_IX["head_num"]] = query.size(head_dim_idx)
        local_args = tuple(local_args)

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
    group = _get_cp_group(mesh, op_info.schema.args_schema, [0, 1, 2], seq_dim=2)
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

    B, N, S, D = attn_output.shape
    dev = attn_output.device
    if step_caches:
        sm_max, sm_sum, seed_step, offset_step = step_caches[-1]
        softmax_max = sm_max
        softmax_sum = sm_sum
    else:
        softmax_max = torch.zeros(B, N, S, 8, dtype=torch.float32, device=dev)
        softmax_sum = torch.zeros(B, N, S, 8, dtype=torch.float32, device=dev)
        seed_step = torch.zeros(1, dtype=torch.int64, device=dev)
        offset_step = torch.zeros(1, dtype=torch.int64, device=dev)

    # local_results matches the v3 forward 6-tuple; wrap() builds the output DTensors via
    # the direct constructor using the propagated output_spec (compile-safe, no from_local).
    local_results = (
        attn_output,
        softmax_max,
        softmax_sum,
        torch.zeros(0, device=dev),    # softmax_out (reserve, unused)
        seed_step.to(torch.int64),     # seed
        offset_step.to(torch.int64),   # offset
    )
    return DTensor._op_dispatcher.wrap(local_results, output_sharding.output_spec)


# ============================================================================
# Backward DTensor Handler -- intercepts npu_fusion_attention_grad_v3
# ============================================================================

def _npu_fa_grad_v3_handler(op_call, args, kwargs):
    """Intercept npu_fusion_attention_grad_v3, run ring attention backward, wrap via spec.

    Same native-mirroring structure as _npu_fa_v3_handler: unwrap_to_op_info ->
    propagate -> ring attention backward -> wrap(local_results, output_spec). No from_local.
    """
    op_info = DTensor._op_dispatcher.unwrap_to_op_info(op_call, args, kwargs)

    DTensor._op_dispatcher.sharding_propagator.propagate(op_info)
    output_sharding = op_info.output_sharding
    if output_sharding is None:
        raise RuntimeError("output sharding should not be None")
    if output_sharding.needs_redistribute:
        raise RuntimeError("inputs need to be redistributed")

    mesh = op_info.compute_mesh
    local_args = op_info.local_args
    local_kwargs = op_info.local_kwargs

    query, key, value, dy = local_args[0], local_args[1], local_args[2], local_args[3]
    input_layout = _get(local_args, _BWD_IX, "input_layout", "BNSD") or local_kwargs.get("input_layout", "BNSD")
    _validate_bnsd_layout(
        query,
        key,
        value,
        input_layout,
        op_name="npu_fusion_attention_grad_v3",
    )
    _validate_cp_passthrough_args_kwargs(local_kwargs, op_name="npu_fusion_attention_grad_v3")
    _validate_cp_sparse_args_kwargs(local_kwargs, op_name="npu_fusion_attention_grad_v3")

    # Stats flow through autograd kwargs (now in local_kwargs after unwrap)
    merged_max = local_kwargs.get("softmax_max", kwargs.get("softmax_max"))
    merged_sum = local_kwargs.get("softmax_sum", kwargs.get("softmax_sum"))
    merged_out = local_kwargs.get("attention_in", kwargs.get("attention_in"))
    seed = local_kwargs.get("seed", kwargs.get("seed"))
    offset = local_kwargs.get("offset", kwargs.get("offset"))
    scale_value = local_kwargs.get("scale_value", kwargs.get("scale_value"))
    keep_prob = local_kwargs.get("keep_prob", kwargs.get("keep_prob", 1.0))
    is_causal = local_kwargs.get("sparse_mode", kwargs.get("sparse_mode", 0)) in (1, 2, 3)

    # Unwrap DTensor stats to local (stats may flow as DTensor in multi-dim mesh).
    merged_max = merged_max._local_tensor if isinstance(merged_max, DTensor) else merged_max
    merged_sum = merged_sum._local_tensor if isinstance(merged_sum, DTensor) else merged_sum
    merged_out = merged_out._local_tensor if isinstance(merged_out, DTensor) else merged_out

    dropout_p = (1.0 - keep_prob) if isinstance(keep_prob, (int, float)) else 0.0
    if scale_value is not None:
        softmax_scale = float(scale_value) if isinstance(scale_value, (int, float)) else scale_value
    else:
        softmax_scale = 1.0 / query.shape[-1] ** 0.5

    # lse is not an op output (NPU fa uses softmax_max/sum); reconstruct it.
    merged_lse = _convert_softmax_to_logsumexp(merged_max, merged_sum)

    pt = _extract_passthrough_kwargs(local_kwargs)

    bop = _make_backward_op(merged_max, merged_sum, seed, offset, pt=pt)
    group = _get_cp_group(mesh, op_info.schema.args_schema, [0, 1, 2, 3], seq_dim=2)
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

    dev = query.device
    local_results = (
        grad_q,
        grad_k,
        grad_v,
        torch.zeros(0, device=dev),  # grad_pse (unused)
        torch.zeros(0, device=dev),  # grad_sink (unused)
    )
    return DTensor._op_dispatcher.wrap(local_results, output_sharding.output_spec)


# ============================================================================
# CP Sharding Strategies -- extend layer-1 strategies with CP (seq shard)
#
# Registered via _op_strategy_context in npu_enable_cp_dtensor_dispatcher (save
# layer-1's original strategy + schema, register CP-extended strategy) and
# restored in npu_disable_cp_dtensor_dispatcher (restore layer-1's original
# strategy + schema). This mirrors native register_cp_sharding_rules/
# unregister_cp_sharding_rules and ensures CP strategies are only active while
# the CP dispatcher is enabled.
#
# _op_strategy_context registers the function via register_op_strategy, which
# calls it with a single OpSchema argument (not unpacked args/kwargs like
# @register_sharding does). So these functions receive op_schema and must:
#   1. Extract args/kwargs from op_schema.args_schema / op_schema.kwargs_schema
#   2. Call layer-1's strategy (which expects unpacked specs)
#   3. Append CP strategy
#   4. Return via expand_to_full_mesh_op_strategy (like @register_sharding does)
# ============================================================================

def _extract_strategy_args(op_schema):
    """Extract args and kwargs from op_schema, converting OpStrategy to DTensorSpec.

    Mirrors @register_sharding's custom_strategy: strategy_to_spec extracts the
    spec from each arg so layer-1 strategy functions receive what they expect.
    """
    from torch.distributed.tensor._op_schema import OpStrategy, TupleStrategy

    def strategy_to_spec(item):
        if isinstance(item, OpStrategy):
            return item.strategies[0].output_spec
        elif isinstance(item, TupleStrategy):
            return tuple(strategy_to_spec(child) for child in item.children)
        else:
            return item

    args = tuple(strategy_to_spec(i) for i in op_schema.args_schema)
    kwargs = {k: strategy_to_spec(v) for k, v in op_schema.kwargs_schema.items()}
    return args, kwargs


def _npu_fa_v3_cp_strategy(op_schema):
    """CP-extended sharding strategy for npu_fusion_attention_v3.

    Calls layer-1's strategy to get Replicate/DP/TP, then appends CP (seq shard).
    """
    from torch.distributed.tensor._ops.utils import (
        expand_to_full_mesh_op_strategy as _expand,
    )

    args, kwargs = _extract_strategy_args(op_schema)

    # Get layer-1 strategies (Replicate/DP/TP)
    strategies = _layer1_fwd_strategy(*args, **kwargs)

    # Extract params from args (positional) or kwargs (keyword-only)
    input_layout = _get_kw(args, kwargs, _FWD_IX, "input_layout", "BNSD")
    pse = _get_kw(args, kwargs, _FWD_IX, "pse")
    padding_mask = _get_kw(args, kwargs, _FWD_IX, "padding_mask")
    prefix = _get_kw(args, kwargs, _FWD_IX, "prefix")
    actual_seq_qlen = _get_kw(args, kwargs, _FWD_IX, "actual_seq_qlen")
    actual_seq_kvlen = _get_kw(args, kwargs, _FWD_IX, "actual_seq_kvlen")
    sink = _get_kw(args, kwargs, _FWD_IX, "sink")
    keep_prob = _get_kw(args, kwargs, _FWD_IX, "keep_prob", 1.0)
    atten_mask = _get_kw(args, kwargs, _FWD_IX, "atten_mask")

    mesh = op_schema.get_mesh_from_args()
    input_index = len(op_schema.op._schema.returns)

    # Determine seq_dim from layout
    if 'S' in input_layout:
        seq_dim = input_layout.index('S')
    elif 'T' in input_layout:
        seq_dim = input_layout.index('T')
    else:
        flat = [out + inp for out, inp in strategies]
        return _expand(mesh, op_schema, flat, input_index=input_index)

    # Same guard as layer-1: only add sharding if no dropout/per-seq args
    unused_args = [pse, padding_mask, prefix, actual_seq_qlen, actual_seq_kvlen, sink]
    if not all(arg is None for arg in unused_args) or keep_prob < 1.0:
        flat = [out + inp for out, inp in strategies]
        return _expand(mesh, op_schema, flat, input_index=input_index)

    # Build CP strategy: q/k/v/softmax_max/sum/attention_out all Shard(seq_dim)
    cp_output = [
        Shard(seq_dim),  # attention_out
        Shard(seq_dim),  # softmax_max
        Shard(seq_dim),  # softmax_sum
        Replicate(),     # softmax_out (unused)
        Replicate(),     # seed
        Replicate(),     # offset
    ]
    # input placements: same length as layer-1's (21 entries matching schema)
    cp_input = list(strategies[0][1])  # copy replicate strategy's input list as template
    # Override tensor inputs to Shard(seq_dim)
    for i in range(3):  # query, key, value (indices 0, 1, 2)
        if cp_input[i] is not None:
            cp_input[i] = Shard(seq_dim)
    # atten_mask (index 7) stays as-is from the replicate template (Replicate if
    # provided, None otherwise) -- it is global, not seq-sharded
    # All other tensor inputs (pse=5, padding_mask=6, actual_seq_qlen=14, actual_seq_kvlen=15, sink=20)
    # are None in CP path, so their placement stays None

    strategies.append((cp_output, cp_input))
    flat = [out + inp for out, inp in strategies]
    return _expand(mesh, op_schema, flat, input_index=input_index)


def _npu_fa_grad_v3_cp_strategy(op_schema):
    """CP-extended sharding strategy for npu_fusion_attention_grad_v3 (backward).

    Calls layer-1's strategy to get Replicate/DP/TP, then appends CP.
    """
    from torch.distributed.tensor._ops.utils import (
        expand_to_full_mesh_op_strategy as _expand,
    )

    args, kwargs = _extract_strategy_args(op_schema)

    # Get layer-1 strategies
    strategies = _layer1_bwd_strategy(*args, **kwargs)

    input_layout = _get_kw(args, kwargs, _BWD_IX, "input_layout", "BNSD")
    pse = _get_kw(args, kwargs, _BWD_IX, "pse")
    padding_mask = _get_kw(args, kwargs, _BWD_IX, "padding_mask")
    prefix = _get_kw(args, kwargs, _BWD_IX, "prefix")
    actual_seq_qlen = _get_kw(args, kwargs, _BWD_IX, "actual_seq_qlen")
    actual_seq_kvlen = _get_kw(args, kwargs, _BWD_IX, "actual_seq_kvlen")
    sink = _get_kw(args, kwargs, _BWD_IX, "sink")
    keep_prob = _get_kw(args, kwargs, _BWD_IX, "keep_prob", 1.0)

    mesh = op_schema.get_mesh_from_args()
    input_index = len(op_schema.op._schema.returns)

    if 'S' in input_layout:
        seq_dim = input_layout.index('S')
    elif 'T' in input_layout:
        seq_dim = input_layout.index('T')
    else:
        flat = [out + inp for out, inp in strategies]
        return _expand(mesh, op_schema, flat, input_index=input_index)

    unused_args = [pse, padding_mask, prefix, actual_seq_qlen, actual_seq_kvlen, sink]
    if not all(arg is None for arg in unused_args) or keep_prob < 1.0:
        flat = [out + inp for out, inp in strategies]
        return _expand(mesh, op_schema, flat, input_index=input_index)

    cp_output = [
        Shard(seq_dim),  # grad_query
        Shard(seq_dim),  # grad_key
        Shard(seq_dim),  # grad_value
        Replicate(),     # grad_pse (unused)
        Replicate(),     # grad_sink
    ]
    # Copy layer-1's replicate input list as template, override tensor inputs
    cp_input = list(strategies[0][1])
    # query(0), key(1), value(2), dy(3) -> Shard(seq_dim)
    for i in range(4):
        if cp_input[i] is not None:
            cp_input[i] = Shard(seq_dim)
    # softmax_max(9), softmax_sum(10), attention_in(12) -> Shard(seq_dim) if present,
    # matching the v3 grad schema positions (they flow in as Shard(seq) from the
    # forward output). These indices correspond to _BWD_IX values.
    for idx in [9, 10, 12]:  # softmax_max, softmax_sum, attention_in
        if idx < len(cp_input) and cp_input[idx] is not None:
            cp_input[idx] = Shard(seq_dim)

    strategies.append((cp_output, cp_input))
    flat = [out + inp for out, inp in strategies]
    return _expand(mesh, op_schema, flat, input_index=input_index)


_npu_fa = torch.ops.npu.npu_fusion_attention_v3.default
_npu_fa_grad = torch.ops.npu.npu_fusion_attention_grad_v3.default

_npu_custom_ops = {
    _npu_fa: _npu_fa_v3_handler,
    _npu_fa_grad: _npu_fa_grad_v3_handler,
}

# ============================================================================
# CP Dispatcher Enable/Disable
# ============================================================================
# Saved values so disable restores the pre-CP state exactly.
_cp_implicit_replication_prev = None
_cp_prev_handlers = {}

# Saved _op_strategy_context handles for NPU CP strategies, restored on disable.
_npu_cp_strategy_contexts = {}


_strategy_context_cache = None


def _get_strategy_context_compat():
    """Return the strategy context manager, compatible across torch versions.

    torch <= 2.13: _op_strategy_context
    torch >= 2.14: _single_dim_strategy_context (renamed)

    Cached after first call to avoid repeated import attempts.
    """
    global _strategy_context_cache
    if _strategy_context_cache is not None:
        return _strategy_context_cache

    try:
        from torch.distributed.tensor.experimental._context_parallel._sharding_rules import (
            _single_dim_strategy_context as _ctx,
        )
    except ImportError:
        from torch.distributed.tensor.experimental._context_parallel._sharding_rules import (
            _op_strategy_context as _ctx,
        )
    _strategy_context_cache = _ctx
    return _ctx


def _register_npu_cp_sharding_rules() -> None:
    """Register NPU CP sharding rules, saving layer-1 originals for restore."""
    if _npu_cp_strategy_contexts:
        return

    from torch.distributed.tensor._op_schema import RuntimeSchemaInfo

    _strategy_context = _get_strategy_context_compat()
    npu_cp_strategies = [
        (_npu_fa, _npu_fa_v3_cp_strategy, RuntimeSchemaInfo(1)),
        (_npu_fa_grad, _npu_fa_grad_v3_cp_strategy, RuntimeSchemaInfo(1)),
    ]
    for op_overload, strategy_func, schema_info in npu_cp_strategies:
        ctx = _strategy_context(op_overload, strategy_func, schema_info)
        ctx.__enter__()
        _npu_cp_strategy_contexts[op_overload] = ctx


def _unregister_npu_cp_sharding_rules() -> None:
    """Restore NPU sharding rules that were active before CP was enabled."""
    for ctx in _npu_cp_strategy_contexts.values():
        ctx.__exit__(None, None, None)
    _npu_cp_strategy_contexts.clear()


def npu_enable_cp_dtensor_dispatcher() -> None:
    """Register NPU SDPA forward/backward handlers and CP sharding rules."""
    # Idempotent guard: if already enabled, do nothing (prevents overwriting
    # saved layer-1 handlers on repeated enable calls).
    if _cp_prev_handlers:
        return

    logger.info("registering handler keys=%s", [str(k) for k in _npu_custom_ops.keys()])

    # Save layer-1 handlers so disable can restore them (CP handlers overwrite
    # the same op_overload keys; simply deleting would leave no handler at all).
    _cp_prev_handlers.clear()
    _cp_prev_handlers.update(
        {k: DTensor._op_dispatcher._custom_op_handlers.get(k) for k in _npu_custom_ops}
    )

    existing = DTensor._op_dispatcher._custom_op_handlers.copy()
    DTensor._op_dispatcher._custom_op_handlers = {**existing, **_npu_custom_ops}

    # Let plain non-scalar tensor args (e.g. atten_mask passed as a plain Tensor
    # alongside DTensor q/k/v) auto-receive a Replicate spec in unwrap_to_op_info.
    # Native SDPA has no such args so it doesn't need this; v3 does. Without it,
    # unwrap_to_op_info raises 'mixed Tensor/DTensor'.
    global _cp_implicit_replication_prev
    _cp_implicit_replication_prev = DTensor._op_dispatcher._allow_implicit_replication
    DTensor._op_dispatcher._allow_implicit_replication = True

    # Register NPU CP sharding rules (save layer-1 originals, install CP-extended).
    _register_npu_cp_sharding_rules()

    # Register native CP sharding rules (flash/efficient/cudnn attention) for
    # completeness, matching the native _enable_cp_dtensor_dispatcher behavior.
    from torch.distributed.tensor.experimental._context_parallel._sharding_rules import (
        register_cp_sharding_rules,
    )
    register_cp_sharding_rules()


def npu_disable_cp_dtensor_dispatcher() -> None:
    """Remove NPU handlers and unregister CP sharding rules."""
    logger.info("removing handler keys=%s", [str(k) for k in _npu_custom_ops.keys()])

    # Restore layer-1 handlers (or remove if none existed before CP).
    handlers = DTensor._op_dispatcher._custom_op_handlers
    for k, prev_handler in _cp_prev_handlers.items():
        if prev_handler is not None:
            handlers[k] = prev_handler
        elif k in handlers:
            del handlers[k]
    _cp_prev_handlers.clear()

    global _cp_implicit_replication_prev
    if _cp_implicit_replication_prev is not None:
        DTensor._op_dispatcher._allow_implicit_replication = _cp_implicit_replication_prev
        _cp_implicit_replication_prev = None

    # Restore NPU sharding rules to layer-1 originals.
    _unregister_npu_cp_sharding_rules()

    from torch.distributed.tensor.experimental._context_parallel._sharding_rules import (
        unregister_cp_sharding_rules,
    )
    unregister_cp_sharding_rules(clear_the_cache=False)
