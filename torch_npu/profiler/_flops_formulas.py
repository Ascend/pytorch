from itertools import pairwise
from math import prod

from ._flops_registry import register_npu_flop


@register_npu_flop(target="torch_npu:npu_fusion_attention", is_default=True)
def npu_fusion_attention_flops(
    query,
    key,
    value,
    head_num,
    input_layout,
    pse=None,
    padding_mask=None,
    atten_mask=None,
    scale=1.0,
    keep_prob=1.0,
    pre_tockens=2147483647,
    next_tockens=2147483647,
    inner_precise=0,
    prefix=None,
    actual_seq_qlen=None,
    actual_seq_kvlen=None,
    sparse_mode=0,
    *args,
    **kwargs,
):
    q_shape = query.shape
    k_shape = key.shape
    v_shape = value.shape
    if input_layout == "TND":
        return _calculate_tnd_layout_flops(
            q_shape, k_shape, v_shape, actual_seq_qlen, actual_seq_kvlen
        )
    kv_heads = _infer_kv_heads(q_shape, k_shape, input_layout, head_num)
    return _calculate_common_layout_flops(
        q_shape, k_shape, v_shape, input_layout, sparse_mode, head_num, kv_heads
    )


@register_npu_flop(target="torch_npu:npu_fused_infer_attention_score", is_default=True)
def npu_fused_infer_attention_score_flops(
    query,
    key,
    value,
    *,
    input_layout,
    num_heads,
    num_key_value_heads=0,
    actual_seq_lengths=None,
    actual_seq_lengths_kv=None,
    sparse_mode=0,
    **kwargs,
):
    num_key_value_heads = num_key_value_heads or num_heads
    q_shape = query.shape
    k_shape = key.shape
    v_shape = value.shape
    if input_layout == "TND":
        return _calculate_tnd_layout_flops(
            q_shape,
            k_shape,
            v_shape,
            actual_seq_lengths,
            actual_seq_lengths_kv,
            num_heads,
        )
    return _calculate_common_layout_flops(
        q_shape,
        k_shape,
        v_shape,
        input_layout,
        sparse_mode,
        num_heads,
        num_key_value_heads,
    )


@register_npu_flop(target="torch_npu:npu_all_gather_base_mm", is_default=True)
def npu_all_gather_base_mm_flops(
    x1,
    x2,
    hcom,
    world_size,
    bias=None,
    x1_scale=None,
    x2_scale=None,
    gather_index=0,
    gather_output=True,
    comm_turn=0,
    output_dtype=None,
    comm_mode=None,
    **kwargs,
):
    x1_shape = _shape(x1)
    x2_shape = _shape(x2)
    m_local, k = x1_shape[-2:]
    n = x2_shape[-1]
    return 2 * m_local * int(world_size) * k * n


@register_npu_flop(target="torch_npu:npu_transpose_batchmatmul", is_default=True)
def npu_transpose_batchmatmul_flops(
    input,
    weight,
    *,
    bias=None,
    scale=None,
    perm_x1=(0, 1, 2),
    perm_x2=(0, 1, 2),
    perm_y=(1, 0, 2),
    batch_split_factor=1,
    **kwargs,
):
    input_shape = _permute_shape(_shape(input), perm_x1)
    weight_shape = _permute_shape(_shape(weight), perm_x2)
    return _matmul_shape_flops(input_shape, weight_shape)


@register_npu_flop(target="torch_npu:npu_grouped_matmul", is_default=True)
def npu_grouped_matmul_flops(
    x,
    weight,
    *,
    bias=None,
    scale=None,
    offset=None,
    antiquant_scale=None,
    antiquant_offset=None,
    per_token_scale=None,
    group_list=None,
    activation_input=None,
    activation_quant_scale=None,
    activation_quant_offset=None,
    split_item=0,
    group_type=None,
    group_list_type=0,
    act_type=0,
    output_dtype=None,
    tuning_config=None,
    **kwargs,
):
    return _grouped_matmul_flops(x, weight, group_list)


@register_npu_flop(target="torch_npu:npu_quant_matmul_gelu", is_default=True)
def npu_quant_matmul_gelu_flops(
    x1,
    x2,
    x1_scale,
    x2_scale,
    *,
    bias=None,
    approximate="gelu_erf",
    **kwargs,
):
    return _matrix_tensor_flops(x1, x2)


@register_npu_flop(
    target="torch_npu:npu_grouped_matmul_swiglu_quant_v2", is_default=True
)
def npu_grouped_matmul_swiglu_quant_v2_flops(
    x,
    weight,
    weight_scale,
    x_scale,
    group_list,
    *,
    smooth_scale=None,
    weight_assist_matrix=None,
    bias=None,
    dequant_mode=0,
    dequant_dtype=0,
    quant_mode=0,
    quant_dtype=0,
    group_list_type=0,
    tuning_config=None,
    **kwargs,
):
    return _matrix_tensor_flops(x, weight)


@register_npu_flop(target="torch_npu:npu_alltoallv_gmm", is_default=True)
def npu_alltoallv_gmm_flops(
    gmm_x,
    gmm_weight,
    hcom,
    ep_world_size,
    send_counts,
    recv_counts,
    *,
    send_counts_tensor=None,
    recv_counts_tensor=None,
    mm_x=None,
    mm_weight=None,
    trans_gmm_weight=False,
    trans_mm_weight=False,
    permute_out_flag=False,
    **kwargs,
):
    return _gmm_with_optional_mm_flops(
        gmm_x, gmm_weight, mm_x, mm_weight, trans_gmm_weight, trans_mm_weight
    )


@register_npu_flop(target="torch_npu:npu_gmm_alltoallv", is_default=True)
def npu_gmm_alltoallv_flops(
    gmm_x,
    gmm_weight,
    hcom,
    ep_world_size,
    send_counts,
    recv_counts,
    *,
    send_counts_tensor=None,
    recv_counts_tensor=None,
    mm_x=None,
    mm_weight=None,
    trans_gmm_weight=False,
    trans_mm_weight=False,
    **kwargs,
):
    return _gmm_with_optional_mm_flops(
        gmm_x, gmm_weight, mm_x, mm_weight, trans_gmm_weight, trans_mm_weight
    )


@register_npu_flop(target="torch_npu:npu_block_sparse_attention", is_default=True)
def npu_block_sparse_attention_flops(
    query,
    key,
    value,
    block_sparse_mask,
    block_shape,
    *,
    q_input_layout="TND",
    kv_input_layout="TND",
    num_key_value_heads=1,
    scale_value=0.0,
    inner_precise=1,
    actual_seq_lengths=None,
    actual_seq_lengths_kv=None,
    softmax_lse_flag=0,
    **kwargs,
):
    q_shape = _shape(query)
    v_shape = _shape(value)
    mask = _to_nested_list(block_sparse_mask)
    q_heads = len(mask[0]) if mask else None
    _, _, q_s, q_d = _parse_attention_dims(q_shape, q_input_layout, q_heads)
    _, _, kv_s, v_d = _parse_attention_dims(
        v_shape, kv_input_layout, num_key_value_heads
    )
    batch = len(mask)
    q_lens = _parse_actual_lengths(
        actual_seq_lengths, batch, q_s, q_input_layout == "TND"
    )
    kv_lens = _parse_actual_lengths(
        actual_seq_lengths_kv, batch, kv_s, kv_input_layout == "TND"
    )
    block_x, block_y = [int(dim) for dim in block_shape]
    score_elems = _count_block_sparse_score_elems(
        mask, q_lens, kv_lens, block_x, block_y
    )
    return int(2 * score_elems * (q_d + v_d))


@register_npu_flop(target="torch:mm", is_default=True)
def mm_flops(input, other, **kwargs):
    m, k = input.shape
    _, n = other.shape
    return 2 * m * n * k


@register_npu_flop(target="torch:bmm", is_default=True)
def bmm_flops(input, other, **kwargs):
    b, m, k = input.shape
    _, _, n = other.shape
    return 2 * b * m * n * k


@register_npu_flop(target="torch:matmul", is_default=True)
def matmul_flops(input, other, **kwargs):
    input_shape = tuple(input.shape)
    other_shape = tuple(other.shape)
    if len(input_shape) == 1 and len(other_shape) == 1:
        return 2 * input_shape[0]
    if len(input_shape) == 1:
        batch_shape = other_shape[:-2]
        m, k, n = 1, input_shape[0], other_shape[-1]
    elif len(other_shape) == 1:
        batch_shape = input_shape[:-2]
        m, k, n = input_shape[-2], input_shape[-1], 1
    else:
        batch_shape = _broadcast_shapes(input_shape[:-2], other_shape[:-2])
        m, k, n = input_shape[-2], input_shape[-1], other_shape[-1]
    return 2 * prod(batch_shape) * m * n * k


@register_npu_flop(target="torch.nn.functional:linear", is_default=True)
def linear_flops(input, weight, bias=None, **kwargs):
    n, k = weight.shape
    return 2 * prod(input.shape[:-1]) * n * k


@register_npu_flop(target="torch:addmm", is_default=True)
def addmm_flops(self, mat1, mat2, beta=1, alpha=1, **kwargs):
    m, k = mat1.shape
    _, n = mat2.shape
    return 2 * m * n * k


def _shape(tensor):
    return tuple(int(dim) for dim in tensor.shape)


def _matmul_shape_flops(left_shape, right_shape, trans_right=False):
    if len(left_shape) < 2 or len(right_shape) < 2:
        raise ValueError(f"Matmul FLOPs requires rank >= 2: {left_shape}, {right_shape}")
    m = prod(left_shape[:-1])
    k = left_shape[-1]
    n = right_shape[-2] if trans_right else right_shape[-1]
    return int(2 * m * k * n)


def _matrix_tensor_flops(left, right, trans_right=False):
    return _matmul_shape_flops(_shape(left), _shape(right), trans_right)


def _as_tensor_list(tensors):
    return list(tensors) if isinstance(tensors, (list, tuple)) else [tensors]


def _grouped_matmul_flops(x, weight, group_list=None):
    x_list = _as_tensor_list(x)
    weight_list = _as_tensor_list(weight)
    if len(x_list) == len(weight_list):
        return sum(
            _matrix_tensor_flops(left, right)
            for left, right in zip(x_list, weight_list)
        )
    if len(x_list) == 1:
        left_shape = _shape(x_list[0])
        group_lengths = _parse_group_lengths(
            group_list, len(weight_list), prod(left_shape[:-1])
        )
        return sum(
            _matmul_shape_flops((group_m, left_shape[-1]), _shape(right))
            for group_m, right in zip(group_lengths, weight_list)
        )
    raise ValueError(
        f"Grouped matmul FLOPs requires matching groups: {len(x_list)}, {len(weight_list)}"
    )


def _parse_group_lengths(group_list, group_count, total_m):
    if group_count == 1:
        return [total_m]
    if group_list is None:
        raise ValueError("Grouped matmul FLOPs requires group_list for split groups")
    groups = [int(group) for group in _to_sequence(group_list)]
    if len(groups) != group_count:
        raise ValueError(f"Expected {group_count} groups, got {len(groups)}")
    if groups[-1] == total_m and sum(groups) > total_m:
        groups = [groups[0]] + [curr - prev for prev, curr in pairwise(groups)]
    if sum(groups) != total_m:
        raise ValueError("group_list does not match grouped matmul token count")
    return groups


def _gmm_with_optional_mm_flops(
    gmm_x, gmm_weight, mm_x, mm_weight, trans_gmm_weight, trans_mm_weight
):
    flops = _matrix_tensor_flops(gmm_x, gmm_weight, trans_gmm_weight)
    if mm_x is not None and mm_weight is not None:
        flops += _matrix_tensor_flops(mm_x, mm_weight, trans_mm_weight)
    return flops


def _permute_shape(tensor_shape, permutation):
    if len(tensor_shape) != len(permutation):
        raise ValueError(
            f"Permutation {permutation} does not match tensor shape {tensor_shape}"
        )
    return tuple(tensor_shape[int(index)] for index in permutation)


def _broadcast_shapes(left_shape, right_shape):
    result = []
    for left, right in zip(reversed(left_shape), reversed(right_shape)):
        if left != right and left != 1 and right != 1:
            raise ValueError(
                f"Cannot broadcast matmul batch dimensions: {left_shape}, {right_shape}"
            )
        result.append(max(left, right))
    longer = left_shape if len(left_shape) > len(right_shape) else right_shape
    result.extend(reversed(longer[: abs(len(left_shape) - len(right_shape))]))
    return tuple(reversed(result))


def _calculate_common_layout_flops(
    q_shape, k_shape, v_shape, input_layout, sparse_mode, q_heads, kv_heads
):
    q_b, q_n, q_s, q_d = _parse_dims(q_shape, input_layout, q_heads)
    _, _, k_s, k_d = _parse_dims(k_shape, input_layout, kv_heads)
    _, _, _, v_d = _parse_dims(v_shape, input_layout, kv_heads)
    attention_scores = _calculate_attention_scores(q_s, k_s, sparse_mode)
    return int(2 * q_b * q_n * attention_scores * (q_d + v_d))


def _calculate_tnd_layout_flops(
    q_shape, k_shape, v_shape, actual_seq_qlen, actual_seq_kvlen, q_heads=None
):
    if actual_seq_qlen is None or actual_seq_kvlen is None:
        raise ValueError("TND layout requires actual_seq_qlen and actual_seq_kvlen")
    _, shape_q_heads, q_d = q_shape
    _, _, v_d = v_shape
    q_lens = _parse_seq_len(actual_seq_qlen)
    kv_lens = _parse_seq_len(actual_seq_kvlen)
    if len(q_lens) != len(kv_lens) or any(length <= 0 for length in q_lens + kv_lens):
        raise ValueError("actual_seq_qlen and actual_seq_kvlen must contain valid cumulative lengths")
    attention_scores = sum(q_len * kv_len for q_len, kv_len in zip(q_lens, kv_lens))
    return int(2 * (q_heads or shape_q_heads) * (q_d + v_d) * attention_scores)


def _infer_kv_heads(q_shape, k_shape, input_layout, q_heads):
    if input_layout == "BNSD":
        return k_shape[1]
    if input_layout == "BSND":
        return k_shape[2]
    if input_layout == "BSH":
        _, _, q_hidden = q_shape
        _, _, k_hidden = k_shape
    elif input_layout == "SBH":
        _, _, q_hidden = q_shape
        _, _, k_hidden = k_shape
    else:
        return q_heads

    q_head_dim = _head_dim(q_hidden, q_heads)
    return _head_dim(k_hidden, q_head_dim)


def _calculate_attention_scores(q_s, k_s, sparse_mode):
    if sparse_mode == 0:
        return q_s * k_s
    if sparse_mode not in (2, 3):
        raise ValueError(f"Unknown FLOPs formula for sparse_mode={sparse_mode}")
    if sparse_mode == 2:
        return q_s * k_s - k_s * k_s / 2 if q_s >= k_s else q_s * q_s / 2
    return k_s * k_s / 2 if q_s >= k_s else q_s * k_s - q_s * q_s / 2


def _parse_dims(tensor_shape, input_layout, heads):
    if input_layout == "BNSD":
        return tensor_shape
    if input_layout == "BSND":
        b, s, n, d = tensor_shape
        return b, n, s, d
    if input_layout == "BSH":
        b, s, h = tensor_shape
        return b, heads, s, _head_dim(h, heads)
    if input_layout == "SBH":
        s, b, h = tensor_shape
        return b, heads, s, _head_dim(h, heads)
    raise ValueError(f"Invalid layout for FlashAttention input tensor: {input_layout}")


def _parse_attention_dims(tensor_shape, input_layout, heads):
    if input_layout == "TND":
        s, n, d = tensor_shape
        return None, n, s, d
    return _parse_dims(tensor_shape, input_layout, heads)


def _head_dim(hidden_size, heads):
    if heads <= 0 or hidden_size % heads != 0:
        raise ValueError(
            f"Hidden size {hidden_size} must be divisible by the number of heads {heads}"
        )
    return hidden_size // heads


def _parse_seq_len(original_seq_lens):
    seq_lens = [int(length) for length in original_seq_lens]
    while seq_lens and seq_lens[-1] == 0:
        seq_lens.pop()
    if not seq_lens:
        return []
    return [seq_lens[0]] + [
        curr - prev for prev, curr in pairwise(seq_lens)
    ]


def _parse_actual_lengths(seq_lens, batch, default_len, is_cumulative=False):
    if seq_lens is None:
        return [default_len] * batch
    lengths = [int(length) for length in seq_lens]
    while lengths and lengths[-1] == 0:
        lengths.pop()
    if len(lengths) != batch:
        raise ValueError(f"Expected {batch} sequence lengths, got {len(lengths)}")
    if is_cumulative:
        lengths = [lengths[0]] + [
            curr - prev for prev, curr in pairwise(lengths)
        ]
    if any(length < 0 for length in lengths):
        raise ValueError("Sequence lengths must be non-negative")
    return lengths


def _to_nested_list(value):
    return _to_sequence(value)


def _to_sequence(value):
    if isinstance(value, (list, tuple)):
        return value
    if hasattr(value, "tolist"):
        sequence = value.tolist()
        if isinstance(sequence, (list, tuple)):
            return sequence
    raise ValueError("Value must be a sequence or expose tolist() returning a sequence")


def _count_block_sparse_score_elems(mask, q_lens, kv_lens, block_x, block_y):
    score_elems = 0
    for batch_idx, heads in enumerate(mask):
        q_len = q_lens[batch_idx]
        kv_len = kv_lens[batch_idx]
        for q_blocks in heads:
            for q_block_idx, kv_blocks in enumerate(q_blocks):
                q_start = q_block_idx * block_x
                q_tokens = min(block_x, max(q_len - q_start, 0))
                if q_tokens == 0:
                    continue
                for kv_block_idx, is_valid in enumerate(kv_blocks):
                    if not is_valid:
                        continue
                    kv_start = kv_block_idx * block_y
                    kv_tokens = min(block_y, max(kv_len - kv_start, 0))
                    score_elems += q_tokens * kv_tokens
    return score_elems
