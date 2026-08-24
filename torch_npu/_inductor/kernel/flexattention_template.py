"""Triton template definitions for NPU FlexAttention."""

try:
    from torch._inductor.kernel.flex.flex_attention import SymbolicGridFn
except ImportError:
    try:
        from torch._inductor.kernel.flex_attention import SymbolicGridFn
    except ImportError:
        def SymbolicGridFn(fn):
            return fn

from torch._inductor.select_algorithm import TritonTemplate

from torch_npu._inductor.select_algorithm import NPUTritonTemplate


def _with_kernel_signature(
    source: str,
    old_signature: str,
    new_signature: str,
) -> str:
    body = source.lstrip("\n")
    prefix = source[: len(source) - len(body)]
    if not body.startswith(old_signature):
        raise RuntimeError("FlexAttention template source has an unexpected signature")
    return prefix + new_signature + body[len(old_signature) :]


# Inner Triton functions shared by flex_attention & split-k decoding kernels.
compute_next_offset_func = r"""
@triton.jit
def get_offset_for_next_block(
    loop_iter, col_indices, total_blocks,
    SPARSE_BLOCK, SPARSE_BLOCK_MULTIPLE, BLOCK,
    BLOCKS_ARE_CONTIGUOUS: tl.constexpr
):
    if BLOCKS_ARE_CONTIGUOUS:
        return BLOCK
    cur_block_idx = loop_iter // SPARSE_BLOCK_MULTIPLE
    cur_block = tl.load(col_indices + cur_block_idx, eviction_policy="evict_last")
    next_block = tl.load(col_indices + cur_block_idx + 1, eviction_policy="evict_last", mask=cur_block_idx + 1 < total_blocks)
    needs_jump = (loop_iter + 1) % SPARSE_BLOCK_MULTIPLE == 0
    jump_to_block = (next_block - cur_block ) * SPARSE_BLOCK - (SPARSE_BLOCK_MULTIPLE - 1) * BLOCK
    offset = jump_to_block * needs_jump + (1 - needs_jump) * BLOCK
    return offset
"""

get_bounded_indices_func = r"""
@triton.jit
def get_bounded_indices(indices, max_len=None):
    return indices % max_len if max_len is not None else indices
"""


compute_sparse_mask_kernel_compact = r"""
{{def_kernel("SPARSE_MASK", "Q_OFFSETS", "FLAT_TO_ROW", "FLAT_TO_BLK", "KV_NUM_BLKS", "KV_IDX")}}
    stride_kv_idx_z = {{stride("KV_IDX", 0)}}
    stride_kv_idx_h = {{stride("KV_IDX", 1)}}
    stride_kv_idx_m = {{stride("KV_IDX", 2)}}
    stride_kv_idx_blk = {{stride("KV_IDX", 3)}}

    TOTAL_ENTRIES : tl.constexpr = TOTAL_FLAT_ENTRIES * NUM_Q_SUB_BLOCKS * NUM_KV_SUB_BLOCKS

    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)

    for entry_idx in range(pid, TOTAL_ENTRIES, num_programs):
        kv_sub = entry_idx % NUM_KV_SUB_BLOCKS
        tmp = entry_idx // NUM_KV_SUB_BLOCKS
        q_sub = tmp % NUM_Q_SUB_BLOCKS
        flat_blk = tmp // NUM_Q_SUB_BLOCKS

        flat_row = tl.load(FLAT_TO_ROW + flat_blk)
        blk_pos = tl.load(FLAT_TO_BLK + flat_blk)
        sq_idx = flat_row % NUM_SPARSE_Q_BLOCKS
        tmp_row = flat_row // NUM_SPARSE_Q_BLOCKS
        sparse_h = tmp_row % SPARSE_HQ
        sparse_z = tmp_row // SPARSE_HQ

        q_offset_idx = sparse_z * SPARSE_HQ * (NUM_SPARSE_Q_BLOCKS + 1) + sparse_h * (NUM_SPARSE_Q_BLOCKS + 1) + sq_idx
        expected_flat_blk = tl.load(Q_OFFSETS + q_offset_idx) + blk_pos

        idx_offset = (
            sparse_z * stride_kv_idx_z
            + sparse_h * stride_kv_idx_h
            + sq_idx * stride_kv_idx_m
            + blk_pos * stride_kv_idx_blk
        )
        kv_block = tl.load(KV_IDX + idx_offset)

        q_start = sq_idx * SPARSE_Q_BLOCK_SIZE
        kv_start = kv_block * SPARSE_KV_BLOCK_SIZE

        offs_m = q_start + q_sub * MASK_BLOCK_M + tl.arange(0, MASK_BLOCK_M)
        offs_m_local = q_sub * MASK_BLOCK_M + tl.arange(0, MASK_BLOCK_M)
        offs_n = kv_start + kv_sub * MASK_BLOCK_N + tl.arange(0, MASK_BLOCK_N)
        offs_n_local = kv_sub * MASK_BLOCK_N + tl.arange(0, MASK_BLOCK_N)

        m = offs_m[:, None]
        n = offs_n[None, :]
        off_z = sparse_z
        off_h = sparse_h

        {{ modification(
            subgraph_number=0,
            output_name="mask_mod_output",
            score="qk",
            b="off_z",
            h="off_h",
            m="m",
            n="n",
        ) | indent_except_first(2) }}

        store_mask = (offs_m[:, None] < Q_LEN) & (offs_n[None, :] < KV_LEN)
        mask_mod_output = mask_mod_output & store_mask
        mask_base = SPARSE_MASK + expected_flat_blk * SPARSE_MASK_STRIDE_BLK
        mask_offsets = offs_m_local[:, None] * SPARSE_MASK_STRIDE_M + offs_n_local[None, :]
        tl.store(mask_base + mask_offsets, mask_mod_output.to(tl.int8))
"""


compute_bwd_sparse_mask_kernel_compact = r"""
{{def_kernel("Q_OFFSETS", "FLAT_TO_ROW", "FLAT_TO_BLK", "KV_NUM_BLKS", "KV_IDX")}}
    SPARSE_MASK = arg_SPARSE_MASK
    stride_kv_idx_z = {{stride("KV_IDX", 0)}}
    stride_kv_idx_h = {{stride("KV_IDX", 1)}}
    stride_kv_idx_m = {{stride("KV_IDX", 2)}}
    stride_kv_idx_blk = {{stride("KV_IDX", 3)}}

    TOTAL_ENTRIES : tl.constexpr = TOTAL_FLAT_ENTRIES * NUM_Q_SUB_BLOCKS * NUM_KV_SUB_BLOCKS

    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)

    for entry_idx in range(pid, TOTAL_ENTRIES, num_programs):
        kv_sub = entry_idx % NUM_KV_SUB_BLOCKS
        tmp = entry_idx // NUM_KV_SUB_BLOCKS
        q_sub = tmp % NUM_Q_SUB_BLOCKS
        flat_blk = tmp // NUM_Q_SUB_BLOCKS

        flat_row = tl.load(FLAT_TO_ROW + flat_blk)
        blk_pos = tl.load(FLAT_TO_BLK + flat_blk)
        sq_idx = flat_row % NUM_SPARSE_Q_BLOCKS
        tmp_row = flat_row // NUM_SPARSE_Q_BLOCKS
        sparse_h = tmp_row % SPARSE_HQ
        sparse_z = tmp_row // SPARSE_HQ

        q_offset_idx = sparse_z * SPARSE_HQ * (NUM_SPARSE_Q_BLOCKS + 1) + sparse_h * (NUM_SPARSE_Q_BLOCKS + 1) + sq_idx
        expected_flat_blk = tl.load(Q_OFFSETS + q_offset_idx) + blk_pos

        idx_offset = (
            sparse_z * stride_kv_idx_z
            + sparse_h * stride_kv_idx_h
            + sq_idx * stride_kv_idx_m
            + blk_pos * stride_kv_idx_blk
        )
        kv_block = tl.load(KV_IDX + idx_offset)

        q_start = sq_idx * SPARSE_Q_BLOCK_SIZE
        kv_start = kv_block * SPARSE_KV_BLOCK_SIZE

        offs_m = q_start + q_sub * MASK_BLOCK_M + tl.arange(0, MASK_BLOCK_M)
        offs_m_local = q_sub * MASK_BLOCK_M + tl.arange(0, MASK_BLOCK_M)
        offs_n = kv_start + kv_sub * MASK_BLOCK_N + tl.arange(0, MASK_BLOCK_N)
        offs_n_local = kv_sub * MASK_BLOCK_N + tl.arange(0, MASK_BLOCK_N)

        m = offs_m[:, None]
        n = offs_n[None, :]
        off_z = sparse_z
        off_h = sparse_h

        {{ modification(
            subgraph_number=0,
            output_name="mask_mod_output",
            score="qk",
            b="off_z",
            h="off_h",
            m="m",
            n="n",
        ) | indent_except_first(2) }}

        store_mask = (offs_m[:, None] < Q_LEN) & (offs_n[None, :] < KV_LEN)
        mask_mod_output = mask_mod_output & store_mask
        mask_base = SPARSE_MASK + expected_flat_blk * SPARSE_MASK_STRIDE_BLK
        mask_offsets = offs_m_local[:, None] * SPARSE_MASK_STRIDE_M + offs_n_local[None, :]
        tl.store(mask_base + mask_offsets, mask_mod_output & store_mask)
"""

compute_sparse_mask_block_pos_kernel = r"""
{{def_kernel("KV_NUM_BLKS", "KV_IDX", "Q_OFFSETS", "SPARSE_MASK_BLOCK_POS")}}
    SPARSE_MASK_BLOCK_POS = arg_SPARSE_MASK_BLOCK_POS
    stride_kv_num_blks_z = {{stride("KV_NUM_BLKS", 0)}}
    stride_kv_num_blks_h = {{stride("KV_NUM_BLKS", 1)}}
    stride_kv_num_blks_m = {{stride("KV_NUM_BLKS", 2)}}
    stride_kv_idx_z = {{stride("KV_IDX", 0)}}
    stride_kv_idx_h = {{stride("KV_IDX", 1)}}
    stride_kv_idx_m = {{stride("KV_IDX", 2)}}
    stride_kv_idx_blk = {{stride("KV_IDX", 3)}}
    stride_block_pos_z = SPARSE_MASK_BLOCK_POS_STRIDE_Z
    stride_block_pos_h = SPARSE_MASK_BLOCK_POS_STRIDE_H
    stride_block_pos_q = SPARSE_MASK_BLOCK_POS_STRIDE_Q

    TOTAL_ENTRIES : tl.constexpr = SPARSE_Z * SPARSE_HQ * NUM_SPARSE_Q_BLOCKS * MAX_NORMAL_BLOCKS

    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)

    for entry_idx in range(pid, TOTAL_ENTRIES, num_programs):
        blk_pos = entry_idx % MAX_NORMAL_BLOCKS
        tmp = entry_idx // MAX_NORMAL_BLOCKS
        sq_idx = tmp % NUM_SPARSE_Q_BLOCKS
        tmp = tmp // NUM_SPARSE_Q_BLOCKS
        sparse_h = tmp % SPARSE_HQ
        sparse_z = tmp // SPARSE_HQ

        nb_offset = (
            sparse_z * stride_kv_num_blks_z
            + sparse_h * stride_kv_num_blks_h
            + sq_idx * stride_kv_num_blks_m
        )
        num_blks = tl.load(KV_NUM_BLKS + nb_offset)

        if blk_pos < num_blks:
            idx_offset = (
                sparse_z * stride_kv_idx_z
                + sparse_h * stride_kv_idx_h
                + sq_idx * stride_kv_idx_m
                + blk_pos * stride_kv_idx_blk
            )
            kv_block = tl.load(KV_IDX + idx_offset)
            block_pos_offset = (
                sparse_z * stride_block_pos_z
                + sparse_h * stride_block_pos_h
                + sq_idx * stride_block_pos_q
                + kv_block
            )
            q_offset_idx = (
                sparse_z * SPARSE_HQ * (NUM_SPARSE_Q_BLOCKS + 1)
                + sparse_h * (NUM_SPARSE_Q_BLOCKS + 1)
                + sq_idx
            )
            partial_block_idx = tl.load(Q_OFFSETS + q_offset_idx) + blk_pos
            tl.store(
                SPARSE_MASK_BLOCK_POS + block_pos_offset,
                partial_block_idx,
            )
"""


compute_forward_block_mn_sparse_mask = r"""
@triton.jit
def forward_block_mn_sparse_mask(
    {{gen_argdefs()}},
    q, k, v, Q_LEN, KV_LEN,
    # accumulated values
    acc, l_i, m_i,
    # Offsets
    off_z, off_h, offs_m, offs_n,
    MATMUL_PRECISION,
    q_start,
    blk_idx_in_list,
    IS_FULL_BLOCKS, CHECK_BLOCK_BOUNDARY=False,

):
    # Redefines all kernel parameters (BLOCK_M, etc.) so we don't need to plumb them all through
    {{gen_defines() | indent_except_first(1)}}
    # -- compute qk ---
    qk = tl.dot(q, tl.trans(k), input_precision="ieee")
    if not PRESCALE_QK:
        qk *= SM_SCALE
    # ~~~~~~~~~~~~~~~~~~~ Apply score modification  ~~~~~~~~~~~~~~~~~~~
    m = get_bounded_indices(offs_m, Q_LEN if CHECK_BLOCK_BOUNDARY else None)
    n = get_bounded_indices(offs_n, KV_LEN if CHECK_BLOCK_BOUNDARY else None)

    {{ modification(
        subgraph_number=0,
        output_name="post_mod_scores",
        score="qk",
        b="off_z",
        h="off_h",
        m="m",
        n="n",
        out="qk"
    ) | indent_except_first(1) }}

    if not IS_FULL_BLOCKS:
{% if TORCHINDUCTOR_FLEXATTENTION_MASKOUT %}
        SPARSE_Q_MULTIPLE: tl.constexpr = (SPARSE_Q_BLOCK_SIZE // BLOCK_M)
        SPARSE_HQ: tl.constexpr = {{size("KV_NUM_BLKS", 1)}}
        q_sparse_idx = q_start // SPARSE_Q_MULTIPLE
        q_sparse_start = q_sparse_idx * SPARSE_Q_BLOCK_SIZE
        sparse_h = off_h % SPARSE_HQ
        sparse_mask_h = off_h % SPARSE_MASK_HQ
        SPARSE_Z: tl.constexpr = {{size("KV_NUM_BLKS", 0)}}
        sparse_idx_z = off_z % SPARSE_Z

        stride_kv_idx_z = {{stride("KV_IDX", 0)}}
        stride_kv_idx_h = {{stride("KV_IDX", 1)}}
        stride_kv_idx_m = {{stride("KV_IDX", 2)}}
        stride_kv_idx_blk = {{stride("KV_IDX", 3)}}
        kv_block = tl.load(
            arg_KV_IDX
            + sparse_idx_z * stride_kv_idx_z
            + sparse_h * stride_kv_idx_h
            + q_sparse_idx * stride_kv_idx_m
            + blk_idx_in_list * stride_kv_idx_blk
        )

        offs_m_local = offs_m - q_sparse_start
        offs_n_local = offs_n - kv_block * SPARSE_KV_BLOCK_SIZE
        q_offsets_idx = (
            sparse_idx_z * SPARSE_MASK_HQ * (NUM_SPARSE_Q_BLOCKS + 1)
            + sparse_mask_h * (NUM_SPARSE_Q_BLOCKS + 1)
            + q_sparse_idx
        )
        flat_blk = tl.load(arg_Q_OFFSETS + q_offsets_idx) + blk_idx_in_list
        mask_base = arg_SPARSE_MASK + flat_blk * SPARSE_MASK_STRIDE_BLK
        mask_offsets = offs_m_local * SPARSE_MASK_STRIDE_M + offs_n_local
        mask_mod_output = tl.load(mask_base + mask_offsets) != 0
{% else %}
        {{ modification(
            subgraph_number=1,
            output_name="mask_mod_output",
            score="qk",
            b="off_z",
            h="off_h",
            m="m",
            n="n",
        ) | indent_except_first(2) }}
        mask_mod_output = mask_mod_output & (offs_m < Q_LEN) & (offs_n < KV_LEN)
{% endif %}
        # apply mask for partially unmasked blocks
        post_mod_scores = tl.where(mask_mod_output, post_mod_scores, float("-inf"))
    elif CHECK_BLOCK_BOUNDARY:
        post_mod_scores = tl.where(offs_n < KV_LEN, post_mod_scores, float("-inf"))

    # -- compute scaling constant ---
    m_ij = tl.maximum(
        m_i,
        tl.max(post_mod_scores, 1, propagate_nan=True),
        propagate_nan=tl.PropagateNan.ALL,
    )
    if not ROWS_GUARANTEED_SAFE:
        masked_out_rows = (m_ij == float("-inf"))
        m_ij_masked = tl.where(masked_out_rows, 0, m_ij)
    else:
        m_ij_masked = m_ij

    alpha = tl.math.exp(m_i - m_ij_masked)
    p = tl.math.exp(post_mod_scores - m_ij_masked[:, None])

    # NB: l_i update is pulled up here since it's a bit faster
    # NB: For headdim=256, it's faster to move it back down to after m_i =
    # m_ij
    l_i = l_i * alpha + tl.sum(p, 1)
    # # -- scale and update acc --
    acc = acc * alpha[:, None]
    acc = tl.dot(p.to(MATMUL_PRECISION), v, acc, input_precision="ieee")
    # -- update m_i
    m_i = m_ij

    return acc, l_i, m_i

"""

compute_forward_inner_sparse_mask_direct_index = r"""
@triton.jit
def forward_inner_sparse_mask_direct_index(
    {{gen_argdefs()}},
    q, K, V, Q_LEN, KV_LEN,
    stride_kk, stride_kn, stride_vn, stride_vk,
    # accumulated values
    acc, l_i, m_i,
    # Offsets used as inputs to score_mod & mask_mod
    off_z, off_h, offs_m,
    # blocksparse data
    kv_indices, kv_num_blocks,
    # start kv and end kv block
    block_n_start, block_n_end,
    MATMUL_PRECISION,
    q_start,
    IS_FULL_BLOCKS,
):
    {{gen_defines() | indent_except_first(1)}}

    SPARSE_KV_MULTIPLE: tl.constexpr = (SPARSE_KV_BLOCK_SIZE // BLOCK_N)
    offs_k = tl.arange(0, QK_HEAD_DIM)
    offs_v = tl.arange(0, V_HEAD_DIM)

    if PRESCALE_QK:
        q = (q * SM_SCALE).to(MATMUL_PRECISION)

    for start_n in range(block_n_start, block_n_end):
        blk_idx_in_list = start_n // SPARSE_KV_MULTIPLE
        kv_block = tl.load(kv_indices + blk_idx_in_list)
        kv_start = kv_block * SPARSE_KV_BLOCK_SIZE + (start_n % SPARSE_KV_MULTIPLE) * BLOCK_N
        offs_n = kv_start + tl.arange(0, BLOCK_N)
        k = tl.load(
            K + offs_n[:, None] * stride_kn + offs_k[None, :] * stride_kk,
            mask=offs_n[:, None] < KV_LEN,
            other=0.0,
        )
        v = tl.load(
            V + offs_n[:, None] * stride_vn + offs_v[None, :] * stride_vk,
            mask=offs_n[:, None] < KV_LEN,
            other=0.0,
        )

        if IS_DIVISIBLE:
            acc, l_i, m_i = forward_block_mn_sparse_mask(
                {{gen_argdefs()}},
                q, k, v, Q_LEN, KV_LEN,
                acc, l_i, m_i,
                off_z, off_h, offs_m, offs_n[None, :],
                MATMUL_PRECISION,
                q_start,
                blk_idx_in_list,
                IS_FULL_BLOCKS,
            )
        else:
            acc, l_i, m_i = forward_block_mn_sparse_mask(
                {{gen_argdefs()}},
                q, k, v, Q_LEN, KV_LEN,
                acc, l_i, m_i,
                off_z, off_h, offs_m, offs_n[None, :],
                MATMUL_PRECISION,
                q_start,
                blk_idx_in_list,
                IS_FULL_BLOCKS, CHECK_BLOCK_BOUNDARY=True,
            )

    return acc, l_i, m_i

"""


compute_flex_attention_sparse_mask_in_loop_no_load_balance = r"""
{{def_kernel("Q", "K", "V", "SPARSE_MASK", "Q_OFFSETS", "KV_NUM_BLKS", "KV_IDX", "LSE", "FULL_KV_NUM_BLKS", "FULL_KV_IDX")}}
    tl.static_assert(SPARSE_Q_BLOCK_SIZE >= BLOCK_M and SPARSE_Q_BLOCK_SIZE % BLOCK_M == 0)
    tl.static_assert(SPARSE_KV_BLOCK_SIZE >= BLOCK_N and SPARSE_KV_BLOCK_SIZE % BLOCK_N == 0)

    stride_qz, stride_qh, stride_qm, stride_qk = {{stride("Q")}}
    stride_kz, stride_kh, stride_kn, stride_kk = {{stride("K")}}
    stride_vz, stride_vh, stride_vn, stride_vk = {{stride("V")}}

    ZQ = {{size("Q", 0)}}
    HQ = {{size("Q", 1)}}
    Q_LEN = {{size("Q", 2)}}
    ZKV = {{size("K", 0)}}
    KV_LEN = {{size("K", 2)}}

    MATMUL_PRECISION = Q.dtype.element_ty

    SPARSE_Q_MULTIPLE: tl.constexpr = (SPARSE_Q_BLOCK_SIZE // BLOCK_M)
    NUM_Q_TILES: tl.constexpr = NUM_SPARSE_Q_BLOCKS * SPARSE_Q_MULTIPLE

    for tile_id in range(tl.program_id(0), NUM_Q_TILES * ZQ * HQ, tl.num_programs(0)):
        q_start = tile_id % NUM_Q_TILES
        off_zh = tile_id // NUM_Q_TILES
        off_zq = off_zh // HQ
        off_hq = off_zh % HQ
        off_zkv = off_zq % ZKV
        off_hkv = off_hq // GQA_SHARED_HEADS

        Q_tile = Q + off_zq * stride_qz + off_hq * stride_qh
        K_tile = K + off_zkv * stride_kz + off_hkv * stride_kh
        V_tile = V + off_zkv * stride_vz + off_hkv * stride_vh

        SPARSE_Z = {{size("KV_NUM_BLKS", 0)}}
        SPARSE_HQ = {{size("KV_NUM_BLKS", 1)}}
        sparse_idx_z = off_zq % SPARSE_Z
        sparse_idx_hq = off_hq % SPARSE_HQ

        SPARSE_KV_MULTIPLE: tl.constexpr = (SPARSE_KV_BLOCK_SIZE // BLOCK_N)
        FULL128_SUBTILES: tl.constexpr = (SPARSE_KV_BLOCK_SIZE // BLOCK_N)

        stride_kv_num_blks_h = {{stride("KV_NUM_BLKS", 1)}}
        stride_kv_idx_h = {{stride("KV_IDX", 1)}}
        stride_kv_idx_m = {{stride("KV_IDX", 2)}}

        m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
        l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
        acc = tl.zeros([BLOCK_M, V_HEAD_DIM], dtype=tl.float32)

        offs_m = q_start * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_k = tl.arange(0, QK_HEAD_DIM)
        offs_v = tl.arange(0, V_HEAD_DIM)
        q_sparse_idx = q_start // SPARSE_Q_MULTIPLE

        sparse_hz_offset = sparse_idx_z * SPARSE_HQ + sparse_idx_hq
        sparse_kv_num_blks_offset = sparse_hz_offset * stride_kv_num_blks_h + q_sparse_idx
        sparse_kv_idx_offset = sparse_hz_offset * stride_kv_idx_h + q_sparse_idx * stride_kv_idx_m

        q = tl.load(
            Q_tile + offs_m[:, None] * stride_qm + offs_k[None, :] * stride_qk,
            mask=offs_m[:, None] < Q_LEN,
            other=0.0,
        )

        kv_indices = KV_IDX + sparse_kv_idx_offset
        kv_num_blocks = tl.load(KV_NUM_BLKS + sparse_kv_num_blks_offset)
        block_n_end = tl.minimum(
            kv_num_blocks * SPARSE_KV_MULTIPLE,
            tl.maximum(tl.cdiv(KV_LEN, BLOCK_N), 1, propagate_nan=True),
            propagate_nan=tl.PropagateNan.ALL,
        )

        acc, l_i, m_i = forward_inner_sparse_mask_direct_index(
            {{gen_argdefs()}},
            q, K_tile, V_tile, Q_LEN, KV_LEN,
            stride_kk, stride_kn, stride_vn, stride_vk,
            acc, l_i, m_i,
            off_zq, off_hq, offs_m[:, None],
            kv_indices, kv_num_blocks,
            0, block_n_end,
            MATMUL_PRECISION,
            q_start,
            IS_FULL_BLOCKS=False,
        )

        FULL_SPARSE_Z = {{size("FULL_KV_NUM_BLKS", 0)}}
        FULL_SPARSE_HQ = {{size("FULL_KV_NUM_BLKS", 1)}}
        full_sparse_idx_z = off_zq % FULL_SPARSE_Z
        full_sparse_idx_hq = off_hq % FULL_SPARSE_HQ

        stride_full_kv_num_blks_h = {{stride("FULL_KV_NUM_BLKS", 1)}}
        stride_full_kv_idx_h = {{stride("FULL_KV_IDX", 1)}}
        stride_full_kv_idx_m = {{stride("FULL_KV_IDX", 2)}}

        full_hz_offset = full_sparse_idx_z * FULL_SPARSE_HQ + full_sparse_idx_hq
        full_kv_num_blks_offset = full_hz_offset * stride_full_kv_num_blks_h + q_sparse_idx
        full_kv_idx_offset = full_hz_offset * stride_full_kv_idx_h + q_sparse_idx * stride_full_kv_idx_m
        kv_num_blocks = tl.load(FULL_KV_NUM_BLKS + full_kv_num_blks_offset)

        if kv_num_blocks > 0:
            kv_indices = FULL_KV_IDX + full_kv_idx_offset

            for start_n in range(0, kv_num_blocks):
                kv_block_start = tl.load(kv_indices + start_n) * SPARSE_KV_BLOCK_SIZE
                for sub_idx in range(0, FULL128_SUBTILES):
                    kv_start = kv_block_start + sub_idx * BLOCK_N
                    offs_n = kv_start + tl.arange(0, BLOCK_N)
                    k = tl.load(
                        K_tile + offs_n[:, None] * stride_kn + offs_k[None, :] * stride_kk,
                        mask=offs_n[:, None] < KV_LEN,
                        other=0.0,
                    )
                    v = tl.load(
                        V_tile + offs_n[:, None] * stride_vn + offs_v[None, :] * stride_vk,
                        mask=offs_n[:, None] < KV_LEN,
                        other=0.0,
                    )

                    if IS_DIVISIBLE:
                        acc, l_i, m_i = forward_block_mn_full(
                            {{gen_argdefs()}},
                            q, k, v, Q_LEN, KV_LEN,
                            acc, l_i, m_i,
                            off_zq, off_hq, offs_m[:, None], offs_n[None, :],
                            MATMUL_PRECISION,
                        )
                    else:
                        acc, l_i, m_i = forward_block_mn_full(
                            {{gen_argdefs()}},
                            q, k, v, Q_LEN, KV_LEN,
                            acc, l_i, m_i,
                            off_zq, off_hq, offs_m[:, None], offs_n[None, :],
                            MATMUL_PRECISION,
                            CHECK_BLOCK_BOUNDARY=True,
                        )

        l_i = tl.where(l_i == 0.0, 1, l_i)
        acc = acc / l_i[:, None]
        idx_zq = off_zq
        idx_hq = off_hq
        idx_m = offs_m[:, None]
        idx_d = offs_v[None, :]
        mask = (idx_m < Q_LEN) & (idx_d < V_HEAD_DIM)

        {{store_output(("idx_zq", "idx_hq", "idx_m", "idx_d"), "acc", "mask", indent_width=8)}}

        if OUTPUT_LOGSUMEXP:
            off_hz = off_zq * HQ + off_hq
            l_ptrs = LSE + off_hz * Q_LEN + offs_m
            lse = m_i + tl.math.log(l_i)
            if IS_DIVISIBLE:
                tl.store(l_ptrs, lse)
            else:
                tl.store(l_ptrs, lse, mask=offs_m < Q_LEN)
"""

compute_forward_block_mn_full = r"""
@triton.jit
def forward_block_mn_full(
    {{gen_argdefs()}},
    q, k, v, Q_LEN, KV_LEN,
    acc, l_i, m_i,
    off_z, off_h, offs_m, offs_n,
    MATMUL_PRECISION,
    CHECK_BLOCK_BOUNDARY=False,
):
    {{gen_defines() | indent_except_first(1)}}
    qk = tl.dot(q, tl.trans(k), input_precision="ieee")
    if not PRESCALE_QK:
        qk *= SM_SCALE

    m = get_bounded_indices(offs_m, Q_LEN if CHECK_BLOCK_BOUNDARY else None)
    n = get_bounded_indices(offs_n, KV_LEN if CHECK_BLOCK_BOUNDARY else None)

    {{ modification(
        subgraph_number=0,
        output_name="post_mod_scores",
        score="qk",
        b="off_z",
        h="off_h",
        m="m",
        n="n",
        out="qk"
    ) | indent_except_first(1) }}

{% if not TORCHINDUCTOR_FLEXATTENTION_MASKOUT %}
    if True:
        {{ modification(
            subgraph_number=1,
            output_name="mask_mod_output",
            score="qk",
            b="off_z",
            h="off_h",
            m="m",
            n="n",
        ) | indent_except_first(2) }}
        mask_mod_output = mask_mod_output & (offs_m < Q_LEN) & (offs_n < KV_LEN)
        post_mod_scores = tl.where(
            mask_mod_output,
            post_mod_scores,
            float("-inf"),
        )
{% endif %}

    m_ij = tl.maximum(
        m_i,
        tl.max(post_mod_scores, 1, propagate_nan=True),
        propagate_nan=tl.PropagateNan.ALL,
    )
    if not ROWS_GUARANTEED_SAFE:
        masked_out_rows = (m_ij == float("-inf"))
        m_ij_masked = tl.where(masked_out_rows, 0, m_ij)
    else:
        m_ij_masked = m_ij

    alpha = tl.math.exp(m_i - m_ij_masked)
    p = tl.math.exp(post_mod_scores - m_ij_masked[:, None])
    l_i = l_i * alpha + tl.sum(p, 1)
    acc = acc * alpha[:, None]
    acc = tl.dot(p.to(MATMUL_PRECISION), v, acc, input_precision="ieee")
    m_i = m_ij
    return acc, l_i, m_i

"""


@SymbolicGridFn
def flex_attention_in_loop_grid(batch_size, q_heads, num_queries, d_model, meta, *, cdiv):
    num_m_blocks = cdiv(num_queries, meta["BLOCK_M"])
    total_tiles = num_m_blocks * batch_size * q_heads
    return (min(total_tiles, meta["NUM_CUBE_CORE"]), 1, 1)


@SymbolicGridFn
def sparse_mask_grid(*args, **kwargs):
    """Compute grid for sparse mask materialization kernel."""
    meta = kwargs.get("meta")
    if meta is None:
        meta = args[-1]
    if "TOTAL_FLAT_ENTRIES" in meta:
        total_entries = (
            meta["TOTAL_FLAT_ENTRIES"]
            * meta["NUM_Q_SUB_BLOCKS"]
            * meta["NUM_KV_SUB_BLOCKS"]
        )
    else:
        total_entries = (
            meta["SPARSE_Z"]
            * meta["SPARSE_HQ"]
            * meta["NUM_SPARSE_Q_BLOCKS"]
            * meta["NUM_Q_SUB_BLOCKS"]
            * meta["MAX_NORMAL_BLOCKS"]
            * meta["NUM_KV_SUB_BLOCKS"]
        )
    num_vector_cores = 48
    return (min(total_entries, num_vector_cores), 1, 1)

del TritonTemplate.all_templates["flex_attention"]
del TritonTemplate.all_templates["flex_attention_backward"]

_FWD_MASK_OUT_SIGNATURE = (
    '{{def_kernel("Q", "K", "V", "SPARSE_MASK", "Q_OFFSETS", '
    '"KV_NUM_BLKS", "KV_IDX", "LSE", "FULL_KV_NUM_BLKS", "FULL_KV_IDX")}}'
)
_FWD_MASK_IN_SIGNATURE = (
    '{{def_kernel("Q", "K", "V", "KV_NUM_BLKS", "KV_IDX", "LSE", '
    '"FULL_KV_NUM_BLKS", "FULL_KV_IDX")}}'
)
_FWD_MASK_OUT_SOURCE = (
    compute_flex_attention_sparse_mask_in_loop_no_load_balance
    + compute_forward_inner_sparse_mask_direct_index
    + compute_forward_block_mn_sparse_mask
    + compute_forward_block_mn_full
    + get_bounded_indices_func
)
_FWD_MASK_IN_SOURCE = _with_kernel_signature(
    _FWD_MASK_OUT_SOURCE,
    _FWD_MASK_OUT_SIGNATURE,
    _FWD_MASK_IN_SIGNATURE,
)

flex_attention_fwd_mask_out = NPUTritonTemplate(
    name="flex_attention_fwd_mask_out",
    grid=flex_attention_in_loop_grid,
    source=_FWD_MASK_OUT_SOURCE,
)

flex_attention_fwd_mask_in = NPUTritonTemplate(
    name="flex_attention_fwd_mask_in",
    grid=flex_attention_in_loop_grid,
    source=_FWD_MASK_IN_SOURCE,
)

flex_attention_fwd_mask_compact = NPUTritonTemplate(
    name="flex_attention_fwd_mask_compact",
    grid=sparse_mask_grid,
    source=compute_sparse_mask_kernel_compact,
)

flex_attention_bwd_mask_compact = NPUTritonTemplate(
    name="flex_attention_bwd_mask_compact",
    grid=sparse_mask_grid,
    source=compute_bwd_sparse_mask_kernel_compact,
    manual_output_buffer="arg_SPARSE_MASK",
)

flex_attention_bwd_mask_pos = NPUTritonTemplate(
    name="flex_attention_bwd_mask_pos",
    grid=sparse_mask_grid,
    source=compute_sparse_mask_block_pos_kernel,
)


@SymbolicGridFn
def flex_attention_backward_dq_grid(
    batch_size, q_heads, num_queries, qk_head_dim, kv_heads, num_key_value, meta, *, cdiv
):
    return (meta["DQ_LAUNCH_PROGRAMS"], 1, 1)


@SymbolicGridFn
def flex_attention_backward_dkdv_grid(
    batch_size, q_heads, num_queries, qk_head_dim, kv_heads, num_key_value, meta, *, cdiv
):
    return (meta["LAUNCH_PROGRAMS"], 1, 1)


flex_attention_backward_qmajor_dq_source = r"""
{{def_kernel("Q", "K", "V", "LSE", "DELTA", "DO", "DQ", "SPARSE_MASK", "Q_OFFSETS", "SPARSE_MASK_BLOCK_POS", "KV_NUM_BLKS", "KV_IDX", "Q_NUM_BLKS", "Q_IDX", "FULL_KV_NUM_BLKS", "FULL_KV_IDX", "FULL_Q_NUM_BLKS", "FULL_Q_IDX")}}  # noqa: B950
    stride_qz, stride_qh, stride_qm, stride_qd = {{stride("Q")}}
    stride_kz, stride_kh, stride_kn, stride_kd = {{stride("K")}}
    stride_vz, stride_vh, stride_vn, stride_vd = {{stride("V")}}
    stride_doz, stride_doh, stride_dom, stride_dod = {{stride("DO")}}
    stride_dqz, stride_dqh, stride_dqm, stride_dqd = {{stride("DQ")}}

    ZQ = {{size("Q", 0)}}
    HQ = {{size("Q", 1)}}
    HKV = {{size("K", 1)}}
    Q_LEN = {{size("Q", 2)}}
    ZKV = {{size("K", 0)}}
    KV_LEN = {{size("K", 2)}}
    MATMUL_PRECISION = Q.dtype.element_ty

    tl.static_assert(BLOCK_M2 == SPARSE_Q_BLOCK_SIZE)
    tl.static_assert(BLOCK_N2 == SPARSE_KV_BLOCK_SIZE)

    pid = tl.program_id(0).to(tl.int32)
    num_core = tl.num_programs(0).to(tl.int32)

    SPARSE_Z = {{size("KV_NUM_BLKS", 0)}}
    SPARSE_HQ = {{size("KV_NUM_BLKS", 1)}}

    offs_k = tl.arange(0, QK_HEAD_DIM)
    offs_v = tl.arange(0, V_HEAD_DIM)

    stride_kv_num_blks_z = {{stride("KV_NUM_BLKS", 0)}}
    stride_kv_num_blks_h = {{stride("KV_NUM_BLKS", 1)}}
    stride_kv_num_blks_m = {{stride("KV_NUM_BLKS", 2)}}
    stride_kv_idx_z = {{stride("KV_IDX", 0)}}
    stride_kv_idx_h = {{stride("KV_IDX", 1)}}
    stride_kv_idx_m = {{stride("KV_IDX", 2)}}
    stride_kv_idx_blk = {{stride("KV_IDX", 3)}}
    stride_full_kv_num_blks_z = {{stride("FULL_KV_NUM_BLKS", 0)}}
    stride_full_kv_num_blks_h = {{stride("FULL_KV_NUM_BLKS", 1)}}
    stride_full_kv_num_blks_m = {{stride("FULL_KV_NUM_BLKS", 2)}}
    stride_full_kv_idx_z = {{stride("FULL_KV_IDX", 0)}}
    stride_full_kv_idx_h = {{stride("FULL_KV_IDX", 1)}}
    stride_full_kv_idx_m = {{stride("FULL_KV_IDX", 2)}}
    stride_full_kv_idx_blk = {{stride("FULL_KV_IDX", 3)}}

    for task_id in range(pid, DQ_NUM_TASKS, num_core):
        q_block = task_id % DQ_NUM_Q_BLOCKS
        off_zq = (task_id // DQ_NUM_Q_BLOCKS) // HQ
        off_hq = (task_id // DQ_NUM_Q_BLOCKS) % HQ
        off_hkv = off_hq // GQA_SHARED_HEADS
        off_zkv = off_zq % ZKV
        sparse_idx_z = off_zq % SPARSE_Z
        sparse_h = off_hq % SPARSE_HQ
        sparse_mask_h = off_hq % SPARSE_MASK_HQ

        q_start = q_block * BLOCK_M2
        offs_m = q_start + tl.arange(0, BLOCK_M2)

        q_base = Q + stride_qz * off_zq + stride_qh * off_hq
        k_base = K + stride_kz * off_zkv + stride_kh * off_hkv
        v_base = V + stride_vz * off_zkv + stride_vh * off_hkv
        do_base = DO + stride_doz * off_zq + stride_doh * off_hq
        dq_base = DQ + stride_dqz * off_zq + stride_dqh * off_hq
        off_chz = ((off_zq * HQ + off_hq) * Q_LEN).to(tl.int64)
        lse_base = LSE + off_chz
        delta_base = DELTA + off_chz

        q = tl.load(
            Q + stride_qz * off_zq + stride_qh * off_hq
            + offs_m[:, None] * stride_qm + offs_k[None, :] * stride_qd,
            mask=offs_m[:, None] < Q_LEN,
            other=0.0,
        )
        do = tl.load(
            DO + stride_doz * off_zq + stride_doh * off_hq
            + offs_m[:, None] * stride_dom + offs_v[None, :] * stride_dod,
            mask=offs_m[:, None] < Q_LEN,
            other=0.0,
        )
        lse = tl.load(lse_base + offs_m, mask=offs_m < Q_LEN, other=float("-inf"))
        lse = tl.where(lse == -float("inf"), 0.0, lse)
        Di = tl.load(delta_base + offs_m, mask=offs_m < Q_LEN, other=0.0)
        dq = tl.zeros([BLOCK_M2, QK_HEAD_DIM], dtype=tl.float32)

        kv_num_offset = (
            sparse_idx_z * stride_kv_num_blks_z
            + sparse_h * stride_kv_num_blks_h
            + q_block * stride_kv_num_blks_m
        )
        kv_idx_offset = (
            sparse_idx_z * stride_kv_idx_z
            + sparse_h * stride_kv_idx_h
            + q_block * stride_kv_idx_m
        )
{% if TORCHINDUCTOR_FLEXATTENTION_MASKOUT %}
        q_offsets_idx = (
            sparse_idx_z * SPARSE_MASK_HQ * (NUM_SPARSE_Q_BLOCKS + 1)
            + sparse_mask_h * (NUM_SPARSE_Q_BLOCKS + 1)
            + q_block
        )
        q_offset_base = tl.load(arg_Q_OFFSETS + q_offsets_idx)
{% endif %}

        kv_num_blocks = tl.load(arg_KV_NUM_BLKS + kv_num_offset)
        for blk_pos in range(0, kv_num_blocks):
            kv_sparse_idx = tl.load(arg_KV_IDX + kv_idx_offset + blk_pos * stride_kv_idx_blk)
            offs_n = kv_sparse_idx * SPARSE_KV_BLOCK_SIZE + tl.arange(0, BLOCK_N2)
            k = tl.load(
                k_base + offs_n[:, None] * stride_kn + offs_k[None, :] * stride_kd,
                mask=offs_n[:, None] < KV_LEN,
                other=0.0,
            )
            v = tl.load(
                v_base + offs_n[:, None] * stride_vn + offs_v[None, :] * stride_vd,
                mask=offs_n[:, None] < KV_LEN,
                other=0.0,
            )
            qk = tl.dot(q, tl.trans(k), input_precision="ieee")
            if not PRESCALE_QK:
                qk *= SM_SCALE

            m = get_bounded_indices(offs_m[:, None], Q_LEN if (not IS_DIVISIBLE or not SAFE_HEAD_DIM) else None)
            n = get_bounded_indices(offs_n[None, :], KV_LEN if (not IS_DIVISIBLE or not SAFE_HEAD_DIM) else None)
{% if TORCHINDUCTOR_FLEXATTENTION_MASKOUT %}
            flat_blk = q_offset_base + blk_pos
            offs_m_local = offs_m[:, None] - q_block * SPARSE_Q_BLOCK_SIZE
            offs_n_local = offs_n[None, :] - kv_sparse_idx * SPARSE_KV_BLOCK_SIZE
            mask_offsets = offs_m_local * SPARSE_MASK_STRIDE_M + offs_n_local
            mask_mod_output = tl.load(
                arg_SPARSE_MASK + flat_blk * SPARSE_MASK_STRIDE_BLK + mask_offsets
            )
{% else %}
            {{ modification(
                subgraph_number=2,
                output_name="mask_mod_output",
                score="qk",
                b="off_zq",
                h="off_hq",
                m="m",
                n="n",
            ) | indent_except_first(3) }}
            mask_mod_output = mask_mod_output & (offs_m[:, None] < Q_LEN) & (offs_n[None, :] < KV_LEN)
{% endif %}
{% if TORCHINDUCTOR_FLEXATTENTION_MASKOUT %}
            qk = tl.where(mask_mod_output, qk, float("-inf"))
            p = tl.math.exp(qk - lse[:, None])
{% else %}
            pre_mod_scores = qk
            {{ modification(
                subgraph_number=0,
                output_name="post_mod_scores",
                score="qk",
                b="off_zq",
                h="off_hq",
                m="m",
                n="n",
                out="qk",
            ) | indent_except_first(3) }}
            post_mod_scores = tl.where(mask_mod_output & (offs_n[None, :] < KV_LEN), post_mod_scores, float("-inf"))
            p = tl.math.exp(post_mod_scores - lse[:, None])
{% endif %}
            dp = tl.dot(do, tl.trans(v), input_precision="ieee")
            ds = p * (dp - Di[:, None])
{% if TORCHINDUCTOR_FLEXATTENTION_MASKOUT %}
            dq += tl.dot(ds.to(MATMUL_PRECISION), k, input_precision="ieee")
{% else %}
            {{ modification(
                subgraph_number=1,
                output_name="grad_scores",
                score="pre_mod_scores",
                b="off_zq",
                h="off_hq",
                m="m",
                n="n",
                grad_score_mod="ds",
            ) | indent_except_first(3) }}
            grad_scores = tl.where(mask_mod_output, grad_scores, 0.0)
            dq += tl.dot(grad_scores.to(MATMUL_PRECISION), k, input_precision="ieee")
{% endif %}

        if HAS_FULL_BLOCKS:
            full_kv_num_offset = (
                sparse_idx_z * stride_full_kv_num_blks_z
                + sparse_h * stride_full_kv_num_blks_h
                + q_block * stride_full_kv_num_blks_m
            )
            full_kv_idx_offset = (
                sparse_idx_z * stride_full_kv_idx_z
                + sparse_h * stride_full_kv_idx_h
                + q_block * stride_full_kv_idx_m
            )
            full_kv_num_blocks = tl.load(arg_FULL_KV_NUM_BLKS + full_kv_num_offset)
            for blk_pos in range(0, full_kv_num_blocks):
                kv_sparse_idx = tl.load(arg_FULL_KV_IDX + full_kv_idx_offset + blk_pos * stride_full_kv_idx_blk)
                offs_n = kv_sparse_idx * SPARSE_KV_BLOCK_SIZE + tl.arange(0, BLOCK_N2)
                k = tl.load(
                    k_base + offs_n[:, None] * stride_kn + offs_k[None, :] * stride_kd,
                    mask=offs_n[:, None] < KV_LEN,
                    other=0.0,
                )
                v = tl.load(
                    v_base + offs_n[:, None] * stride_vn + offs_v[None, :] * stride_vd,
                    mask=offs_n[:, None] < KV_LEN,
                    other=0.0,
                )
                qk = tl.dot(q, tl.trans(k), input_precision="ieee")
                if not PRESCALE_QK:
                    qk *= SM_SCALE

                m = get_bounded_indices(offs_m[:, None], Q_LEN if (not IS_DIVISIBLE or not SAFE_HEAD_DIM) else None)
                n = get_bounded_indices(offs_n[None, :], KV_LEN if (not IS_DIVISIBLE or not SAFE_HEAD_DIM) else None)
{% if not TORCHINDUCTOR_FLEXATTENTION_MASKOUT %}
                {{ modification(
                    subgraph_number=2,
                    output_name="mask_mod_output",
                    score="qk",
                    b="off_zq",
                    h="off_hq",
                    m="m",
                    n="n",
                ) | indent_except_first(4) }}
                mask_mod_output = mask_mod_output & (offs_m[:, None] < Q_LEN) & (offs_n[None, :] < KV_LEN)
{% endif %}
{% if TORCHINDUCTOR_FLEXATTENTION_MASKOUT %}
                # full block don't need
                # qk = tl.where(offs_n[None, :] < KV_LEN, qk, float("-inf"))
                p = tl.math.exp(qk - lse[:, None])
{% else %}
                pre_mod_scores = qk
                {{ modification(
                    subgraph_number=0,
                    output_name="post_mod_scores",
                    score="qk",
                    b="off_zq",
                    h="off_hq",
                    m="m",
                    n="n",
                    out="qk",
                ) | indent_except_first(4) }}
                post_mod_scores = tl.where(mask_mod_output, post_mod_scores, float("-inf"))
                p = tl.math.exp(post_mod_scores - lse[:, None])
{% endif %}
                dp = tl.dot(do, tl.trans(v), input_precision="ieee")
                ds = p * (dp - Di[:, None])
{% if TORCHINDUCTOR_FLEXATTENTION_MASKOUT %}
                dq += tl.dot(ds.to(MATMUL_PRECISION), k, input_precision="ieee")
{% else %}
                {{ modification(
                    subgraph_number=1,
                    output_name="grad_scores",
                    score="pre_mod_scores",
                    b="off_zq",
                    h="off_hq",
                    m="m",
                    n="n",
                    grad_score_mod="ds",
                ) | indent_except_first(4) }}
                grad_scores = tl.where(mask_mod_output, grad_scores, 0.0)
                dq += tl.dot(grad_scores.to(MATMUL_PRECISION), k, input_precision="ieee")
{% endif %}

        dq *= SM_SCALE
        index_m = offs_m[:, None]
        index_k = offs_k[None, :]
        if SAFE_HEAD_DIM:
            dq_mask = index_m < Q_LEN
        else:
            dq_mask = (index_m < Q_LEN) & (index_k < QK_HEAD_DIM)
        tl.store(dq_base + index_m * stride_dqm + index_k * stride_dqd, dq, mask=dq_mask)

@triton.jit
def get_bounded_indices(indices, max_len=None):
    return indices % max_len if max_len is not None else indices
"""



flex_attention_backward_dkdv_only_source = r"""
{{def_kernel("Q", "K", "V", "LSE", "DELTA", "DO", "DV", "DK", "SPARSE_MASK", "Q_OFFSETS", "SPARSE_MASK_BLOCK_POS", "KV_NUM_BLKS", "KV_IDX", "Q_NUM_BLKS", "Q_IDX", "FULL_KV_NUM_BLKS", "FULL_KV_IDX", "FULL_Q_NUM_BLKS", "FULL_Q_IDX")}}  # noqa: B950
    # Sub notation for this kernel:
    #
    # Q: Query, K: Key, V: Value
    # LSE: logsumexp (logsumexp is always stored in fp32 regardless of the input dtype)
    # DELTA: Precomputed sum(OUT*DO, axis=-1)
    # DO: Derivative of Output, DQ: Derivative of Query, DV: Derivative of Value
    # DK: Derivative of Key
    # M: Number of queries, N: Number of keys/values
    # QK_HEAD_DIM: The dimension of the query and key embeddings
    # V_HEAD_DIM: The dimension of the value embeddings
    # z: Batch size, h: Number of heads, m: Number of queries or keys/values, d: Head dim
    # GQA_SHARED_HEADS: number of query heads sharing one kv head in GQA setups.
    # (Modifiable) Performance tuning options
    # BLOCK_M1: when calculating DK & DV, iterate over BLOCK_M1 across the seqlen dim of Q in each thread block.
    # BLOCK_N1: when calculating DK & DV, the thread block size across the seqlen dim of K/V.
    # BLOCK_M2: when calculating DQ, the thread block size across the seqlen dim of Q.
    # BLOCK_N2: when calculating DQ, iterate over BLOCK_N2 across the seqlen dim of K/V in each thread block.
    #
    # The following FULL_* and PARTIAL_* is defined in the block sparse mask grid, rather than the thread block grid.
    # KV_NUM_BLKS: The number of KV blocks (that may or may not require masking) for each query.
    # KV_IDX: The indices of KV blocks (that may or may not require masking) for each query.
    # Q_NUM_BLKS: The number of Q blocks (that may or may not require masking) for each query.
    # Q_IDX: The indices of Q blocks (that may or may not require masking) for each query.
    # FULL_KV_NUM_BLKS: The number of fully unmasked KV blocks (so we don't need masking) for each query.
    # FULL_KV_IDX: The indices of fully unmasked KV blocks (so we don't need masking) for each query.
    # FULL_Q_NUM_BLKS: The number of fully unmasked Q blocks (so we don't need masking) for each query.
    # FULL_Q_IDX: The indices of fully unmasked Q blocks (so we don't need masking) for each query.

    # The below are kernel options that can be applied for certain score_mods,
    # or involve a numerics vs. perf tradeoff
    # PRESCALE_QK: Whether to pre-scale QK by 1/sqrt(d). Has
    # about 20% more numerical error, but slightly faster.

    # Define strides of inputs
    stride_qz, stride_qh, stride_qm, stride_qd = {{stride("Q")}}
    stride_kz, stride_kh, stride_kn, stride_kd = {{stride("K")}}
    stride_vz, stride_vh, stride_vn, stride_vd = {{stride("V")}}
    stride_doz, stride_doh, stride_dom, stride_dod = {{stride("DO")}}

    stride_dvz, stride_dvh, stride_dvm, stride_dvd = {{stride("DV")}}

    ZQ = {{size("Q", 0)}}
    HQ = {{size("Q", 1)}}
    HKV = {{size("K", 1)}}
    Q_LEN = {{size("Q", 2)}}
    ZKV = {{size("K", 0)}}
    KV_LEN = {{size("K", 2)}}

    MATMUL_PRECISION = Q.dtype.element_ty

    NUM_KV_BLOCKS = tl.cdiv(KV_LEN, BLOCK_N1)
    NUM_TASKS = NUM_KV_BLOCKS * ZKV * HKV

    pid = tl.program_id(0).to(tl.int32)
    num_core = tl.num_programs(0).to(tl.int32)

    SPARSE_Z = {{size("KV_NUM_BLKS", 0)}}
    SPARSE_HQ = {{size("KV_NUM_BLKS", 1)}}

    offs_k = tl.arange(0, QK_HEAD_DIM)
    offs_v = tl.arange(0, V_HEAD_DIM)

    SPARSE_Q_MULTIPLE = (SPARSE_Q_BLOCK_SIZE // BLOCK_M1)
    SPARSE_KV_MULTIPLE = (SPARSE_KV_BLOCK_SIZE // BLOCK_N1)

    stride_q_num_blks_h = {{stride("Q_NUM_BLKS", 1)}}
    stride_q_idx_h = {{stride("Q_IDX", 1)}}
    stride_q_idx_n = {{stride("Q_IDX", 2)}}

    for task_id in range(pid, NUM_TASKS, num_core):
        kv_start_block = task_id % NUM_KV_BLOCKS
        off_zq = (task_id // NUM_KV_BLOCKS) // HKV
        off_hkv = (task_id // NUM_KV_BLOCKS) % HKV
        off_zkv = off_zq % ZKV
        sparse_idx_z = off_zq % SPARSE_Z

        start_n1 = kv_start_block * BLOCK_N1
        offs_n1 = start_n1 + tl.arange(0, BLOCK_N1)
        pid_mask = kv_start_block // SPARSE_KV_MULTIPLE

        k_adj = (stride_kh * off_hkv + stride_kz * off_zkv).to(tl.int64)
        v_adj = (stride_vh * off_hkv + stride_vz * off_zkv).to(tl.int64)
        dv_adj = (stride_dvh * off_hkv + stride_dvz * off_zq).to(tl.int64)

        K1 = K + k_adj
        V1 = V + v_adj
        DV1 = DV + dv_adj

        k = tl.load(
            K1 + offs_n1[:, None] * stride_kn + offs_k[None, :] * stride_kd,
            mask=(offs_n1[:, None] < KV_LEN) & (offs_k[None, :] < QK_HEAD_DIM),
            other=0.0,
        )
        if PRESCALE_QK:
            k = (k * SM_SCALE).to(MATMUL_PRECISION)
        v = tl.load(
            V1 + offs_n1[:, None] * stride_vn + offs_v[None, :] * stride_vd,
            mask=(offs_n1[:, None] < KV_LEN) & (offs_v[None, :] < V_HEAD_DIM),
            other=0.0,
        )

        for off_g in range(0, GQA_SHARED_HEADS):
            off_hq1 = off_hkv * GQA_SHARED_HEADS + off_g

            q_adj1 = (stride_qh * off_hq1 + stride_qz * off_zq).to(tl.int64)
            do_adj1 = (stride_doh * off_hq1 + stride_doz * off_zq).to(tl.int64)
            off_chz1 = ((off_zq * HQ + off_hq1) * Q_LEN).to(tl.int64)

            Q1 = Q + q_adj1
            DO1 = DO + do_adj1
            LSE1 = LSE + off_chz1
            DELTA1 = DELTA + off_chz1

            sparse_idx_hq1 = off_hq1 % SPARSE_HQ
            sparse_hz_offset = sparse_idx_z * SPARSE_HQ + sparse_idx_hq1
            sparse_q_num_blks_offset = sparse_hz_offset * stride_q_num_blks_h + pid_mask
            sparse_q_idx_offset = sparse_hz_offset * stride_q_idx_h + pid_mask * stride_q_idx_n

            q_indices = Q_IDX + sparse_q_idx_offset
            sparse_q_num_blocks = tl.load(Q_NUM_BLKS + sparse_q_num_blks_offset)
            hi = tl.minimum(
                sparse_q_num_blocks * SPARSE_Q_MULTIPLE,
                tl.maximum(tl.cdiv(Q_LEN, BLOCK_M1), 1),
            )
            for start_m in range(0, hi):
                blk_idx_in_list = start_m // SPARSE_Q_MULTIPLE
                q_block = tl.load(q_indices + blk_idx_in_list)
                q_start = q_block * SPARSE_Q_BLOCK_SIZE + (start_m % SPARSE_Q_MULTIPLE) * BLOCK_M1
                offs_m1 = q_start + tl.arange(0, BLOCK_M1)
                bwd_dkdv_block_mn(
                    {{gen_argdefs()}},
                    Q1, DO1, DK, DELTA1, LSE1, DV1,
                    k, v, Q_LEN, KV_LEN,
                    off_zq, off_hq1, off_hkv, offs_n1, offs_m1, q_start, q_block, pid_mask, offs_k, offs_v,
                    stride_qm, stride_qd, stride_dom, stride_dod,
                    stride_dvm, stride_dvd, stride_kz, stride_kh, stride_kn, stride_kd,
                    MATMUL_PRECISION,
                    False, CHECK_BLOCK_BOUNDARY=not IS_DIVISIBLE,
                )

            if HAS_FULL_BLOCKS:
                q_indices = FULL_Q_IDX + sparse_q_idx_offset
                sparse_q_num_blocks = tl.load(FULL_Q_NUM_BLKS + sparse_q_num_blks_offset)
                hi = tl.minimum(
                    sparse_q_num_blocks * SPARSE_Q_MULTIPLE,
                    tl.maximum(tl.cdiv(Q_LEN, BLOCK_M1), 1),
                )
                for start_m in range(0, hi):
                    blk_idx_in_list = start_m // SPARSE_Q_MULTIPLE
                    q_block = tl.load(q_indices + blk_idx_in_list)
                    q_start = q_block * SPARSE_Q_BLOCK_SIZE + (start_m % SPARSE_Q_MULTIPLE) * BLOCK_M1
                    offs_m1 = q_start + tl.arange(0, BLOCK_M1)
{% if not PRESCALE_QK %}
                    bwd_dkdv_full_block_mn(
                        {{gen_argdefs()}},
                        Q1, DO1, DK, DELTA1, LSE1, DV1,
                        k, v, Q_LEN, KV_LEN,
                        off_zq, off_hq1, off_hkv, offs_n1, offs_m1, q_start, offs_k, offs_v,
                        stride_qm, stride_qd, stride_dom, stride_dod,
                        stride_dvm, stride_dvd, stride_kz, stride_kh, stride_kn, stride_kd,
                        MATMUL_PRECISION,
                        CHECK_BLOCK_BOUNDARY=False,
                    )
{% else %}
                    bwd_dkdv_block_mn(
                        {{gen_argdefs()}},
                        Q1, DO1, DK, DELTA1, LSE1, DV1,
                        k, v, Q_LEN, KV_LEN,
                        off_zq, off_hq1, off_hkv, offs_n1, offs_m1, q_start, q_block, pid_mask, offs_k, offs_v,
                        stride_qm, stride_qd, stride_dom, stride_dod,
                        stride_dvm, stride_dvd, stride_kz, stride_kh, stride_kn, stride_kd,
                        MATMUL_PRECISION,
                        True, CHECK_BLOCK_BOUNDARY=not IS_DIVISIBLE,
                    )
{% endif %}

@triton.jit
def bwd_dkdv_block_mn(
    {{gen_argdefs()}},
    Q, DO, DK, DELTA, LSE, DV,
    k, v, Q_LEN, KV_LEN,
    off_z, off_hq, off_hkv, offs_n1, offs_m1, start_m1, q_sparse_idx, kv_sparse_idx, offs_k, offs_v,
    stride_qm, stride_qd, stride_dom, stride_dod,
    stride_dvm, stride_dvd, stride_kz, stride_kh, stride_kn, stride_kd,
    MATMUL_PRECISION,
    IS_FULL_BLOCKS, CHECK_BLOCK_BOUNDARY=False,
):
    {{gen_defines() | indent_except_first(1) }}
    qT = tl.load(
        Q + offs_m1[:, None] * stride_qm + offs_k[None, :] * stride_qd,
        mask=(offs_m1[:, None] < Q_LEN) & (offs_k[None, :] < QK_HEAD_DIM),
        other=0.0,
    )
    if IS_DIVISIBLE:
        lse = tl.load(LSE + offs_m1)
    else:
        lse = tl.load(LSE + offs_m1, mask=offs_m1 < Q_LEN, other=float("-inf"))
    lse = tl.where(lse == -float("inf"), 0.0, lse)
    qkT = tl.dot(qT, tl.trans(k), input_precision="ieee")
    if not PRESCALE_QK:
        qkT *= SM_SCALE
    m = get_bounded_indices(offs_m1[:, None], Q_LEN if CHECK_BLOCK_BOUNDARY else None)
    n = get_bounded_indices(offs_n1[None, :], KV_LEN if (not IS_DIVISIBLE or CHECK_BLOCK_BOUNDARY) else None)

{% if not TORCHINDUCTOR_FLEXATTENTION_MASKOUT %}
    pre_mod_scores = qkT
    {{ modification(
        subgraph_number=0,
        output_name="post_mod_scores",
        score="qkT",
        b="off_z",
        h="off_hq",
        m="m",
        n="n",
        out="qkT"
    ) | indent_except_first(1) }}

    if CHECK_BLOCK_BOUNDARY:
        post_mod_scores = tl.where(offs_n1[None, :] < KV_LEN, post_mod_scores, float("-inf"))
{% endif %}

    if not IS_FULL_BLOCKS:
{% if TORCHINDUCTOR_FLEXATTENTION_MASKOUT %}
        SPARSE_Z: tl.constexpr = {{size("KV_NUM_BLKS", 0)}}
        SPARSE_HQ: tl.constexpr = {{size("KV_NUM_BLKS", 1)}}
        sparse_idx_z = off_z % SPARSE_Z
        sparse_mask_h = off_hq % SPARSE_MASK_HQ
        q_sparse_start = q_sparse_idx * SPARSE_Q_BLOCK_SIZE
        block_pos_offset = (
            sparse_idx_z * {{stride("SPARSE_MASK_BLOCK_POS", 0)}}
            + sparse_mask_h * {{stride("SPARSE_MASK_BLOCK_POS", 1)}}
            + q_sparse_idx * {{stride("SPARSE_MASK_BLOCK_POS", 2)}}
            + kv_sparse_idx
        )
        partial_block_idx = tl.load(
            arg_SPARSE_MASK_BLOCK_POS + block_pos_offset
        )
        safe_partial_block_idx = tl.maximum(partial_block_idx, 0)

        offs_m_local = offs_m1[:, None] - q_sparse_start
        offs_n_local = offs_n1[None, :] - kv_sparse_idx * SPARSE_KV_BLOCK_SIZE
        mask_base = (
            arg_SPARSE_MASK
            + safe_partial_block_idx * SPARSE_MASK_STRIDE_BLK
        )
        mask_offsets = offs_m_local * SPARSE_MASK_STRIDE_M + offs_n_local
        mask_mod_output = tl.load(mask_base + mask_offsets)
        mask_mod_output = mask_mod_output & (partial_block_idx >= 0)
{% else %}
        {{ modification(
            subgraph_number=2,
            output_name="mask_mod_output",
            score="qkT",
            b="off_z",
            h="off_hq",
            m="m",
            n="n",
        ) | indent_except_first(2) }}
        mask_mod_output = mask_mod_output & (offs_m1[:, None] < Q_LEN) & (offs_n1[None, :] < KV_LEN)
{% endif %}
{% if TORCHINDUCTOR_FLEXATTENTION_MASKOUT %}
        qkT = tl.where(mask_mod_output, qkT, float("-inf"))
{% else %}
        post_mod_scores = tl.where(
            mask_mod_output,
            post_mod_scores,
            float("-inf"),
        )
{% endif %}

{% if TORCHINDUCTOR_FLEXATTENTION_MASKOUT %}
    pT = tl.math.exp(qkT - lse[:, None]).to(MATMUL_PRECISION)
{% else %}
    pT = tl.math.exp(post_mod_scores - lse[:, None])
{% endif %}
    do = tl.load(
        DO + offs_m1[:, None] * stride_dom + offs_v[None, :] * stride_dod,
        mask=(offs_m1[:, None] < Q_LEN) & (offs_v[None, :] < V_HEAD_DIM),
        other=0.0,
    )
    dv = tl.dot(tl.trans(pT.to(MATMUL_PRECISION)), do, input_precision="ieee")
    index_n = offs_n1[:, None]
    index_v = offs_v[None, :]
    dv_ptrs = DV + index_n * stride_dvm + index_v * stride_dvd
    tl.atomic_add(
        dv_ptrs,
        dv,
        mask=(index_n < KV_LEN) & (index_v < V_HEAD_DIM),
    )
    if IS_DIVISIBLE:
        Di = tl.load(DELTA + offs_m1)
    else:
        Di = tl.load(DELTA + offs_m1, mask=offs_m1 < Q_LEN, other=0.0)
    dpT = tl.dot(do, tl.trans(v), input_precision="ieee")
    dsT = (pT * (dpT - Di[:, None])).to(MATMUL_PRECISION)
{% if not TORCHINDUCTOR_FLEXATTENTION_MASKOUT %}
    {{ modification(
        subgraph_number=1,
        output_name="grad_scores",
        score="pre_mod_scores",
        b="off_z",
        h="off_hq",
        m="m",
        n="n",
        grad_score_mod="dsT"
    ) | indent_except_first(1) }}
{% endif %}

{% if RUN_CAPTURED_GRADS %}
    idx_b = off_z
    idx_h = off_hq
    idx_m = m
    idx_n = n
    scatter_mask = (offs_m1[:, None] < Q_LEN) & (offs_n1[None, :] < KV_LEN)
    {{ modification(
        subgraph_number=3,
        output_name=None,
        mask="scatter_mask",
        score="pre_mod_scores",
        b="idx_b",
        h="idx_h",
        m="idx_m",
        n="idx_n",
        grad_score_mod="dsT"
    ) | indent_except_first(1) }}
{% endif %}

{% if not TORCHINDUCTOR_FLEXATTENTION_MASKOUT %}
    dsT = grad_scores
    if not IS_FULL_BLOCKS:
        dsT = tl.where(mask_mod_output, dsT, 0.0)
    dsT = tl.where(offs_m1[:, None] < Q_LEN, dsT, 0.0)
{% endif %}

    index_k = offs_k[None, :]

    dk = tl.dot(tl.trans(dsT).to(MATMUL_PRECISION), qT, input_precision="ieee")
{% if PRESCALE_QK %}
    dk *= SM_SCALE
{% endif %}
    if SAFE_HEAD_DIM:
        dk_mask = index_n < KV_LEN
    else:
        dk_mask = (index_n < KV_LEN) & (index_k < QK_HEAD_DIM)
    dk_ptrs = DK + tl.broadcast_to(
        index_n * stride_kn + index_k * stride_kd + stride_kh * off_hkv + stride_kz * off_z,
        dk.shape,
    )
    tl.atomic_add(dk_ptrs, dk, mask=dk_mask)

@triton.jit
def bwd_dkdv_full_block_mn(
    {{gen_argdefs()}},
    Q, DO, DK, DELTA, LSE, DV,
    k, v, Q_LEN, KV_LEN,
    off_z, off_hq, off_hkv, offs_n1, offs_m1, start_m1, offs_k, offs_v,
    stride_qm, stride_qd, stride_dom, stride_dod,
    stride_dvm, stride_dvd, stride_kz, stride_kh, stride_kn, stride_kd,
    MATMUL_PRECISION,
    CHECK_BLOCK_BOUNDARY=False,
):
    {{gen_defines() | indent_except_first(1) }}
    qT = tl.load(
        Q + offs_m1[:, None] * stride_qm + offs_k[None, :] * stride_qd,
        mask=(offs_m1[:, None] < Q_LEN) & (offs_k[None, :] < QK_HEAD_DIM),
        other=0.0,
    )
    if IS_DIVISIBLE:
        lse = tl.load(LSE + offs_m1)
    else:
        lse = tl.load(LSE + offs_m1, mask=offs_m1 < Q_LEN, other=float("-inf"))
    lse = tl.where(lse == -float("inf"), 0.0, lse)
    qkT = tl.dot(qT, tl.trans(k), input_precision="ieee")
    if not PRESCALE_QK:
        qkT *= SM_SCALE
    m = get_bounded_indices(offs_m1[:, None], Q_LEN if CHECK_BLOCK_BOUNDARY else None)
    n = get_bounded_indices(offs_n1[None, :], KV_LEN if (not IS_DIVISIBLE or CHECK_BLOCK_BOUNDARY) else None)

{% if not TORCHINDUCTOR_FLEXATTENTION_MASKOUT %}
    {{ modification(
            subgraph_number=2,
            output_name="mask_mod_output",
            score="qkT",
            b="off_z",
            h="off_hq",
            m="m",
            n="n",
        ) | indent_except_first(1) }}
    mask_mod_output = mask_mod_output & (offs_m1[:, None] < Q_LEN) & (offs_n1[None, :] < KV_LEN)
{% endif %}

{% if TORCHINDUCTOR_FLEXATTENTION_MASKOUT %}
    if CHECK_BLOCK_BOUNDARY:
        qkT = tl.where(offs_n1[None, :] < KV_LEN, qkT, float("-inf"))
    pT = tl.math.exp(qkT - lse[:, None])
{% else %}
    pre_mod_scores = qkT
    {{ modification(
        subgraph_number=0,
        output_name="post_mod_scores",
        score="qkT",
        b="off_z",
        h="off_hq",
        m="m",
        n="n",
        out="qkT"
    ) | indent_except_first(1) }}

    if CHECK_BLOCK_BOUNDARY:
        post_mod_scores = tl.where(offs_n1[None, :] < KV_LEN, post_mod_scores, float("-inf"))

    post_mod_scores = tl.where(
        mask_mod_output,
        post_mod_scores,
        float("-inf"),
    )

    pT = tl.math.exp(post_mod_scores - lse[:, None])
{% endif %}
    do = tl.load(
        DO + offs_m1[:, None] * stride_dom + offs_v[None, :] * stride_dod,
        mask=(offs_m1[:, None] < Q_LEN) & (offs_v[None, :] < V_HEAD_DIM),
        other=0.0,
    )
    index_n = offs_n1[:, None]
    dv = tl.dot(tl.trans(pT.to(MATMUL_PRECISION)), do, input_precision="ieee")
    index_v = offs_v[None, :]
    dv_ptrs = DV + index_n * stride_dvm + index_v * stride_dvd
    tl.atomic_add(
        dv_ptrs,
        dv,
        mask=(index_n < KV_LEN) & (index_v < V_HEAD_DIM),
    )
    if IS_DIVISIBLE:
        Di = tl.load(DELTA + offs_m1)
    else:
        Di = tl.load(DELTA + offs_m1, mask=offs_m1 < Q_LEN, other=0.0)
    dpT = tl.dot(do, tl.trans(v), input_precision="ieee")
    dsT = (pT * (dpT - Di[:, None])).to(MATMUL_PRECISION)
{% if not TORCHINDUCTOR_FLEXATTENTION_MASKOUT %}
    {{ modification(
        subgraph_number=1,
        output_name="grad_scores",
        score="pre_mod_scores",
        b="off_z",
        h="off_hq",
        m="m",
        n="n",
        grad_score_mod="dsT"
    ) | indent_except_first(1) }}
{% endif %}

{% if RUN_CAPTURED_GRADS %}
    idx_b = off_z
    idx_h = off_hq
    idx_m = m
    idx_n = n
    scatter_mask = (offs_m1[:, None] < Q_LEN) & (offs_n1[None, :] < KV_LEN)
    {{ modification(
        subgraph_number=3,
        output_name=None,
        mask="scatter_mask",
        score="pre_mod_scores",
        b="idx_b",
        h="idx_h",
        m="idx_m",
        n="idx_n",
        grad_score_mod="dsT"
    ) | indent_except_first(1) }}
{% endif %}

{% if not TORCHINDUCTOR_FLEXATTENTION_MASKOUT %}
    dsT = grad_scores
    dsT = tl.where(mask_mod_output, dsT, 0.0)
    dsT = tl.where(offs_m1[:, None] < Q_LEN, dsT, 0.0)
{% endif %}
    index_k = offs_k[None, :]

    dk = tl.dot(tl.trans(dsT).to(MATMUL_PRECISION), qT, input_precision="ieee")
{% if PRESCALE_QK %}
    dk *= SM_SCALE
{% endif %}
    if SAFE_HEAD_DIM:
        dk_mask = index_n < KV_LEN
    else:
        dk_mask = (index_n < KV_LEN) & (index_k < QK_HEAD_DIM)
    dk_ptrs = DK + tl.broadcast_to(
        index_n * stride_kn + index_k * stride_kd + stride_kh * off_hkv + stride_kz * off_z,
        dk.shape,
    )
    tl.atomic_add(dk_ptrs, dk, mask=dk_mask)


@triton.jit
def get_bounded_indices(indices, max_len=None):
    return indices % max_len if max_len is not None else indices
"""

_FLEX_ATTENTION_BACKWARD_DKDV_HELPERS_MARKER = (
    "\n@triton.jit\ndef bwd_dkdv_block_mn("
)
_FLEX_ATTENTION_BACKWARD_DKDV_HELPERS_SOURCE = (
    flex_attention_backward_dkdv_only_source[
        flex_attention_backward_dkdv_only_source.index(
            _FLEX_ATTENTION_BACKWARD_DKDV_HELPERS_MARKER
        ) :
    ]
)

flex_attention_backward_dkdv_tasklist_source = (
    r"""
{{def_kernel("Q", "K", "V", "LSE", "DELTA", "DO", "DV", "DK", "SPARSE_MASK", "Q_OFFSETS", "SPARSE_MASK_BLOCK_POS", "KV_NUM_BLKS", "KV_IDX", "Q_NUM_BLKS", "Q_IDX", "FULL_KV_NUM_BLKS", "FULL_KV_IDX", "FULL_Q_NUM_BLKS", "FULL_Q_IDX", "WORK_ITEMS", "TASK_OFFSETS", "DK_PARTIAL", "DV_PARTIAL")}}  # noqa: B950
    stride_qz, stride_qh, stride_qm, stride_qd = {{stride("Q")}}
    stride_kz, stride_kh, stride_kn, stride_kd = {{stride("K")}}
    stride_vz, stride_vh, stride_vn, stride_vd = {{stride("V")}}
    stride_doz, stride_doh, stride_dom, stride_dod = {{stride("DO")}}
    stride_dvz, stride_dvh, stride_dvm, stride_dvd = {{stride("DV")}}

    stride_q_num_blks_z = {{stride("Q_NUM_BLKS", 0)}}
    stride_q_num_blks_h = {{stride("Q_NUM_BLKS", 1)}}
    stride_q_idx_z = {{stride("Q_IDX", 0)}}
    stride_q_idx_h = {{stride("Q_IDX", 1)}}
    stride_q_idx_n = {{stride("Q_IDX", 2)}}
    stride_full_q_num_blks_z = {{stride("FULL_Q_NUM_BLKS", 0)}}
    stride_full_q_num_blks_h = {{stride("FULL_Q_NUM_BLKS", 1)}}
    stride_full_q_idx_z = {{stride("FULL_Q_IDX", 0)}}
    stride_full_q_idx_h = {{stride("FULL_Q_IDX", 1)}}
    stride_full_q_idx_n = {{stride("FULL_Q_IDX", 2)}}

    ZQ = {{size("Q", 0)}}
    HQ = {{size("Q", 1)}}
    HKV = {{size("K", 1)}}
    Q_LEN = {{size("Q", 2)}}
    ZKV = {{size("K", 0)}}
    KV_LEN = {{size("K", 2)}}
    SPARSE_Z = {{size("KV_NUM_BLKS", 0)}}
    SPARSE_HQ = {{size("KV_NUM_BLKS", 1)}}

    MATMUL_PRECISION = Q.dtype.element_ty
    SPARSE_Q_MULTIPLE = SPARSE_Q_BLOCK_SIZE // BLOCK_M1
    SPARSE_KV_MULTIPLE = SPARSE_KV_BLOCK_SIZE // BLOCK_N1
    offs_k = tl.arange(0, QK_HEAD_DIM)
    offs_v = tl.arange(0, V_HEAD_DIM)

    meta_id = tl.program_id(0).to(tl.int32)
    work_start = tl.load(TASK_OFFSETS + meta_id)
    work_end = tl.load(TASK_OFFSETS + meta_id + 1)
    for work_idx in range(work_start, work_end):
        off_hkv = tl.load(WORK_ITEMS + work_idx * 5 + 0).to(tl.int64)
        kv_start_block = tl.load(WORK_ITEMS + work_idx * 5 + 1)
{% if not TASKLIST_NO_SPLIT %}
        sub_id = tl.load(WORK_ITEMS + work_idx * 5 + 2)
        split_count = tl.load(WORK_ITEMS + work_idx * 5 + 3)
        is_split = tl.load(WORK_ITEMS + work_idx * 5 + 4)
{% endif %}

        off_zq = tl.zeros_like(off_hkv)
        off_zkv = tl.zeros_like(off_hkv)
        sparse_idx_z = tl.zeros_like(off_hkv)
        pid_mask = kv_start_block // SPARSE_KV_MULTIPLE
        start_n1 = kv_start_block * BLOCK_N1
        offs_n1 = start_n1 + tl.arange(0, BLOCK_N1)

        k_adj = (stride_kh * off_hkv + stride_kz * off_zkv).to(tl.int64)
        v_adj = (stride_vh * off_hkv + stride_vz * off_zkv).to(tl.int64)
        dv_adj = (stride_dvh * off_hkv + stride_dvz * off_zq).to(tl.int64)
        K1 = K + k_adj
        V1 = V + v_adj
{% if TASKLIST_NO_SPLIT %}
        DV_OUT = DV + dv_adj
{% else %}
        DV_DIRECT = DV + dv_adj
        DK_SPLIT = DK_PARTIAL + sub_id * PARTIAL_DK_STRIDE
        DV_SPLIT = DV_PARTIAL + sub_id * PARTIAL_DV_STRIDE + dv_adj
{% endif %}

        k = tl.load(
            K1 + offs_n1[:, None] * stride_kn + offs_k[None, :] * stride_kd,
            mask=(offs_n1[:, None] < KV_LEN) & (offs_k[None, :] < QK_HEAD_DIM),
            other=0.0,
        )
        if PRESCALE_QK:
            k = (k * SM_SCALE).to(MATMUL_PRECISION)
        v = tl.load(
            V1 + offs_n1[:, None] * stride_vn + offs_v[None, :] * stride_vd,
            mask=(offs_n1[:, None] < KV_LEN) & (offs_v[None, :] < V_HEAD_DIM),
            other=0.0,
        )

        for off_g in range(0, GQA_SHARED_HEADS):
            off_hq1 = off_hkv * GQA_SHARED_HEADS + off_g
            sparse_idx_hq1 = off_hq1 % SPARSE_HQ
            q_adj1 = (stride_qh * off_hq1 + stride_qz * off_zq).to(tl.int64)
            do_adj1 = (stride_doh * off_hq1 + stride_doz * off_zq).to(tl.int64)
            off_chz1 = ((off_zq * HQ + off_hq1) * Q_LEN).to(tl.int64)
            Q1 = Q + q_adj1
            DO1 = DO + do_adj1
            LSE1 = LSE + off_chz1
            DELTA1 = DELTA + off_chz1

            q_num_offset = (
                sparse_idx_z * stride_q_num_blks_z
                + sparse_idx_hq1 * stride_q_num_blks_h
                + pid_mask
            )
            q_idx_offset = (
                sparse_idx_z * stride_q_idx_z
                + sparse_idx_hq1 * stride_q_idx_h
                + pid_mask * stride_q_idx_n
            )
            q_indices = Q_IDX + q_idx_offset
            q_count = tl.load(Q_NUM_BLKS + q_num_offset)
            q_hi = tl.minimum(
                q_count * SPARSE_Q_MULTIPLE,
                tl.maximum(tl.cdiv(Q_LEN, BLOCK_M1), 1),
            )
{% if TASKLIST_NO_SPLIT %}
            q_begin = 0
            q_end = q_hi
{% else %}
            if is_split == 0:
                q_begin = 0
                q_end = q_hi
            else:
                q_begin = sub_id * q_hi // split_count
                q_end = (sub_id + 1) * q_hi // split_count
{% endif %}
            for start_m in range(q_begin, q_end):
                blk_idx_in_list = start_m // SPARSE_Q_MULTIPLE
                q_block = tl.load(q_indices + blk_idx_in_list)
                q_start = (
                    q_block * SPARSE_Q_BLOCK_SIZE
                    + (start_m % SPARSE_Q_MULTIPLE) * BLOCK_M1
                )
                offs_m1 = q_start + tl.arange(0, BLOCK_M1)
{% if TASKLIST_NO_SPLIT %}
                bwd_dkdv_block_mn(
                    {{gen_argdefs()}},
                    Q1, DO1, DK, DELTA1, LSE1, DV_OUT,
                    k, v, Q_LEN, KV_LEN,
                    off_zq, off_hq1, off_hkv, offs_n1, offs_m1,
                    q_start, q_block, pid_mask, offs_k, offs_v,
                    stride_qm, stride_qd, stride_dom, stride_dod,
                    stride_dvm, stride_dvd, stride_kz, stride_kh,
                    stride_kn, stride_kd, MATMUL_PRECISION,
                    False, CHECK_BLOCK_BOUNDARY=not IS_DIVISIBLE,
                )
{% else %}
                if is_split == 0:
                    bwd_dkdv_block_mn(
                        {{gen_argdefs()}},
                        Q1, DO1, DK, DELTA1, LSE1, DV_DIRECT,
                        k, v, Q_LEN, KV_LEN,
                        off_zq, off_hq1, off_hkv, offs_n1, offs_m1,
                        q_start, q_block, pid_mask, offs_k, offs_v,
                        stride_qm, stride_qd, stride_dom, stride_dod,
                        stride_dvm, stride_dvd, stride_kz, stride_kh,
                        stride_kn, stride_kd, MATMUL_PRECISION,
                        False, CHECK_BLOCK_BOUNDARY=not IS_DIVISIBLE,
                    )
                else:
                    bwd_dkdv_block_mn(
                        {{gen_argdefs()}},
                        Q1, DO1, DK_SPLIT, DELTA1, LSE1, DV_SPLIT,
                        k, v, Q_LEN, KV_LEN,
                        off_zq, off_hq1, off_hkv, offs_n1, offs_m1,
                        q_start, q_block, pid_mask, offs_k, offs_v,
                        stride_qm, stride_qd, stride_dom, stride_dod,
                        stride_dvm, stride_dvd, stride_kz, stride_kh,
                        stride_kn, stride_kd, MATMUL_PRECISION,
                        False, CHECK_BLOCK_BOUNDARY=not IS_DIVISIBLE,
                    )
{% endif %}

            if HAS_FULL_BLOCKS:
                full_q_num_offset = (
                    sparse_idx_z * stride_full_q_num_blks_z
                    + sparse_idx_hq1 * stride_full_q_num_blks_h
                    + pid_mask
                )
                full_q_idx_offset = (
                    sparse_idx_z * stride_full_q_idx_z
                    + sparse_idx_hq1 * stride_full_q_idx_h
                    + pid_mask * stride_full_q_idx_n
                )
                full_q_indices = FULL_Q_IDX + full_q_idx_offset
                full_q_count = tl.load(FULL_Q_NUM_BLKS + full_q_num_offset)
                full_q_hi = tl.minimum(
                    full_q_count * SPARSE_Q_MULTIPLE,
                    tl.maximum(tl.cdiv(Q_LEN, BLOCK_M1), 1),
                )
{% if TASKLIST_NO_SPLIT %}
                full_q_begin = 0
                full_q_end = full_q_hi
{% else %}
                if is_split == 0:
                    full_q_begin = 0
                    full_q_end = full_q_hi
                else:
                    full_q_begin = sub_id * full_q_hi // split_count
                    full_q_end = (sub_id + 1) * full_q_hi // split_count
{% endif %}
                for start_m in range(full_q_begin, full_q_end):
                    blk_idx_in_list = start_m // SPARSE_Q_MULTIPLE
                    q_block = tl.load(full_q_indices + blk_idx_in_list)
                    q_start = (
                        q_block * SPARSE_Q_BLOCK_SIZE
                        + (start_m % SPARSE_Q_MULTIPLE) * BLOCK_M1
                    )
                    offs_m1 = q_start + tl.arange(0, BLOCK_M1)
{% if not PRESCALE_QK %}
{% if TASKLIST_NO_SPLIT %}
                    bwd_dkdv_full_block_mn(
                        {{gen_argdefs()}},
                        Q1, DO1, DK, DELTA1, LSE1, DV_OUT,
                        k, v, Q_LEN, KV_LEN,
                        off_zq, off_hq1, off_hkv, offs_n1, offs_m1,
                        q_start, offs_k, offs_v,
                        stride_qm, stride_qd, stride_dom, stride_dod,
                        stride_dvm, stride_dvd, stride_kz, stride_kh,
                        stride_kn, stride_kd, MATMUL_PRECISION,
                        CHECK_BLOCK_BOUNDARY=False,
                    )
{% else %}
                    if is_split == 0:
                        bwd_dkdv_full_block_mn(
                            {{gen_argdefs()}},
                            Q1, DO1, DK, DELTA1, LSE1, DV_DIRECT,
                            k, v, Q_LEN, KV_LEN,
                            off_zq, off_hq1, off_hkv, offs_n1, offs_m1,
                            q_start, offs_k, offs_v,
                            stride_qm, stride_qd, stride_dom, stride_dod,
                            stride_dvm, stride_dvd, stride_kz, stride_kh,
                            stride_kn, stride_kd, MATMUL_PRECISION,
                            CHECK_BLOCK_BOUNDARY=False,
                        )
                    else:
                        bwd_dkdv_full_block_mn(
                            {{gen_argdefs()}},
                            Q1, DO1, DK_SPLIT, DELTA1, LSE1, DV_SPLIT,
                            k, v, Q_LEN, KV_LEN,
                            off_zq, off_hq1, off_hkv, offs_n1, offs_m1,
                            q_start, offs_k, offs_v,
                            stride_qm, stride_qd, stride_dom, stride_dod,
                            stride_dvm, stride_dvd, stride_kz, stride_kh,
                            stride_kn, stride_kd, MATMUL_PRECISION,
                            CHECK_BLOCK_BOUNDARY=False,
                        )
{% endif %}
{% else %}
{% if TASKLIST_NO_SPLIT %}
                    bwd_dkdv_block_mn(
                        {{gen_argdefs()}},
                        Q1, DO1, DK, DELTA1, LSE1, DV_OUT,
                        k, v, Q_LEN, KV_LEN,
                        off_zq, off_hq1, off_hkv, offs_n1, offs_m1,
                        q_start, q_block, pid_mask, offs_k, offs_v,
                        stride_qm, stride_qd, stride_dom, stride_dod,
                        stride_dvm, stride_dvd, stride_kz, stride_kh,
                        stride_kn, stride_kd, MATMUL_PRECISION,
                        True, CHECK_BLOCK_BOUNDARY=not IS_DIVISIBLE,
                    )
{% else %}
                    if is_split == 0:
                        bwd_dkdv_block_mn(
                            {{gen_argdefs()}},
                            Q1, DO1, DK, DELTA1, LSE1, DV_DIRECT,
                            k, v, Q_LEN, KV_LEN,
                            off_zq, off_hq1, off_hkv, offs_n1, offs_m1,
                            q_start, q_block, pid_mask, offs_k, offs_v,
                            stride_qm, stride_qd, stride_dom, stride_dod,
                            stride_dvm, stride_dvd, stride_kz, stride_kh,
                            stride_kn, stride_kd, MATMUL_PRECISION,
                            True, CHECK_BLOCK_BOUNDARY=not IS_DIVISIBLE,
                        )
                    else:
                        bwd_dkdv_block_mn(
                            {{gen_argdefs()}},
                            Q1, DO1, DK_SPLIT, DELTA1, LSE1, DV_SPLIT,
                            k, v, Q_LEN, KV_LEN,
                            off_zq, off_hq1, off_hkv, offs_n1, offs_m1,
                            q_start, q_block, pid_mask, offs_k, offs_v,
                            stride_qm, stride_qd, stride_dom, stride_dod,
                            stride_dvm, stride_dvd, stride_kz, stride_kh,
                            stride_kn, stride_kd, MATMUL_PRECISION,
                            True, CHECK_BLOCK_BOUNDARY=not IS_DIVISIBLE,
                        )
{% endif %}
{% endif %}
"""
    + _FLEX_ATTENTION_BACKWARD_DKDV_HELPERS_SOURCE
)

flex_attention_backward_dkdv_reduce_source = r"""
{{def_kernel("DK", "DV", "DK_PARTIAL", "DV_PARTIAL", "SPLIT_BASES")}}
    stride_dkz, stride_dkh, stride_dkn, stride_dkd = {{stride("DK")}}
    stride_dvz, stride_dvh, stride_dvn, stride_dvd = {{stride("DV")}}
    KV_LEN = {{size("DK", 2)}}

    split_base_id = tl.program_id(0).to(tl.int32)
    off_hkv = tl.load(SPLIT_BASES + split_base_id * 3 + 0).to(tl.int64)
    kv_block = tl.load(SPLIT_BASES + split_base_id * 3 + 1)
    split_count = tl.load(SPLIT_BASES + split_base_id * 3 + 2)
    off_z = tl.zeros_like(off_hkv)

    start_n = kv_block * BLOCK_N1
    offs_n = start_n + tl.arange(0, BLOCK_N1)
    offs_k = tl.arange(0, QK_HEAD_DIM)
    offs_v = tl.arange(0, V_HEAD_DIM)
    dk_base = off_z * stride_dkz + off_hkv * stride_dkh
    dv_base = off_z * stride_dvz + off_hkv * stride_dvh

    dk_sum = tl.zeros([BLOCK_N1, QK_HEAD_DIM], tl.float32)
    for sub_id in range(split_count):
        dk_sum += tl.load(
            DK_PARTIAL
            + sub_id * PARTIAL_DK_STRIDE
            + dk_base
            + offs_n[:, None] * stride_dkn
            + offs_k[None, :] * stride_dkd,
            mask=(offs_n[:, None] < KV_LEN)
            & (offs_k[None, :] < QK_HEAD_DIM),
            other=0.0,
        )
    tl.store(
        DK
        + dk_base
        + offs_n[:, None] * stride_dkn
        + offs_k[None, :] * stride_dkd,
        dk_sum,
        mask=(offs_n[:, None] < KV_LEN)
        & (offs_k[None, :] < QK_HEAD_DIM),
    )

    dv_sum = tl.zeros([BLOCK_N1, V_HEAD_DIM], tl.float32)
    for sub_id in range(split_count):
        dv_sum += tl.load(
            DV_PARTIAL
            + sub_id * PARTIAL_DV_STRIDE
            + dv_base
            + offs_n[:, None] * stride_dvn
            + offs_v[None, :] * stride_dvd,
            mask=(offs_n[:, None] < KV_LEN)
            & (offs_v[None, :] < V_HEAD_DIM),
            other=0.0,
        )
    tl.store(
        DV
        + dv_base
        + offs_n[:, None] * stride_dvn
        + offs_v[None, :] * stride_dvd,
        dv_sum,
        mask=(offs_n[:, None] < KV_LEN)
        & (offs_v[None, :] < V_HEAD_DIM),
    )
"""

_BWD_DQ_MASK_OUT_SIGNATURE = (
    '{{def_kernel("Q", "K", "V", "LSE", "DELTA", "DO", "DQ", '
    '"SPARSE_MASK", "Q_OFFSETS", "SPARSE_MASK_BLOCK_POS", "KV_NUM_BLKS", '
    '"KV_IDX", "Q_NUM_BLKS", "Q_IDX", "FULL_KV_NUM_BLKS", "FULL_KV_IDX", '
    '"FULL_Q_NUM_BLKS", "FULL_Q_IDX")}}'
)
_BWD_DQ_MASK_IN_SIGNATURE = (
    '{{def_kernel("Q", "K", "V", "LSE", "DELTA", "DO", "DQ", '
    '"KV_NUM_BLKS", "KV_IDX", "Q_NUM_BLKS", "Q_IDX", "FULL_KV_NUM_BLKS", '
    '"FULL_KV_IDX", "FULL_Q_NUM_BLKS", "FULL_Q_IDX")}}'
)
_BWD_DQ_MASK_IN_SOURCE = _with_kernel_signature(
    flex_attention_backward_qmajor_dq_source,
    _BWD_DQ_MASK_OUT_SIGNATURE,
    _BWD_DQ_MASK_IN_SIGNATURE,
)

_BWD_DKDV_MASK_OUT_SIGNATURE = (
    '{{def_kernel("Q", "K", "V", "LSE", "DELTA", "DO", "DV", "DK", '
    '"SPARSE_MASK", "Q_OFFSETS", "SPARSE_MASK_BLOCK_POS", "KV_NUM_BLKS", '
    '"KV_IDX", "Q_NUM_BLKS", "Q_IDX", "FULL_KV_NUM_BLKS", "FULL_KV_IDX", '
    '"FULL_Q_NUM_BLKS", "FULL_Q_IDX")}}'
)
_BWD_DKDV_MASK_IN_SIGNATURE = (
    '{{def_kernel("Q", "K", "V", "LSE", "DELTA", "DO", "DV", "DK", '
    '"KV_NUM_BLKS", "KV_IDX", "Q_NUM_BLKS", "Q_IDX", "FULL_KV_NUM_BLKS", '
    '"FULL_KV_IDX", "FULL_Q_NUM_BLKS", "FULL_Q_IDX")}}'
)
_BWD_DKDV_MASK_IN_SOURCE = _with_kernel_signature(
    flex_attention_backward_dkdv_only_source,
    _BWD_DKDV_MASK_OUT_SIGNATURE,
    _BWD_DKDV_MASK_IN_SIGNATURE,
)

flex_attention_bwd_dq_mask_out = NPUTritonTemplate(
    name="flex_attention_bwd_dq_mask_out",
    grid=flex_attention_backward_dq_grid,
    source=flex_attention_backward_qmajor_dq_source,
)

flex_attention_bwd_dq_mask_in = NPUTritonTemplate(
    name="flex_attention_bwd_dq_mask_in",
    grid=flex_attention_backward_dq_grid,
    source=_BWD_DQ_MASK_IN_SOURCE,
)

flex_attention_bwd_dkdv_mask_out = NPUTritonTemplate(
    name="flex_attention_bwd_dkdv_mask_out",
    grid=flex_attention_backward_dkdv_grid,
    source=flex_attention_backward_dkdv_only_source,
)

flex_attention_bwd_dkdv_mask_in = NPUTritonTemplate(
    name="flex_attention_bwd_dkdv_mask_in",
    grid=flex_attention_backward_dkdv_grid,
    source=_BWD_DKDV_MASK_IN_SOURCE,
)

flex_attention_bwd_dkdv_tasklist = NPUTritonTemplate(
    name="flex_attention_bwd_dkdv_tasklist",
    grid=flex_attention_backward_dkdv_grid,
    source=flex_attention_backward_dkdv_tasklist_source,
)

flex_attention_bwd_dkdv_tasklist_no_split = NPUTritonTemplate(
    name="flex_attention_bwd_dkdv_tasklist_no_split",
    grid=flex_attention_backward_dkdv_grid,
    source=flex_attention_backward_dkdv_tasklist_source,
)

flex_attention_bwd_dkdv_reduce = NPUTritonTemplate(
    name="flex_attention_bwd_dkdv_reduce",
    grid=flex_attention_backward_dkdv_grid,
    source=flex_attention_backward_dkdv_reduce_source,
)
