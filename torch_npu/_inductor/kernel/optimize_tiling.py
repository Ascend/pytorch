import math
from dataclasses import dataclass, field
from typing import Optional


# ═══════════════════════════
#  Platform constants
# ═══════════════════════════

AIC_NUM       = 28
AIV_NUM       = 56
UB_SIZE       = 253952
L1_SIZE       = 524288       # 512 KB
L2_SIZE       = 134_217_728  # 128 MB
L0A_SIZE      = 65_536       # 64 KB
L0B_SIZE      = 65_536       # 64 KB
L0C_SIZE      = 262_144      # 256 KB
BT_SIZE       = 4096

# Platform performance parameters (queried at runtime; using defaults from source)
CUBE_FREQ_MHZ = 1650         # default from GetCoreFreq
DDR_RATE      = 31           # default from GetHbmBW
L2_RATE       = 100          # default from GetL2BW

# Derived bandwidth values
CORE_FREQ     = CUBE_FREQ_MHZ / 1000.0                 # 1.65
HBM_BW        = CORE_FREQ * AIC_NUM * DDR_RATE / 1024  # ~1.398
L2_BW         = CORE_FREQ * AIC_NUM * L2_RATE / 1024   # ~4.512

DB_SIZE            = 2
DB_OFF_SIZE        = 1
BASIC_BLOCK_16     = 16
BASIC_BLOCK_64     = 64
BASIC_BLOCK_128    = 128
BASIC_BLOCK_256    = 256
K_128_BYTE         = 128    # BASIC_BLOCK_K_128_BYTE (in bytes)
K_256_BYTE         = 256    # BASIC_BLOCK_K_256_BYTE
K_512_BYTE         = 512    # BASIC_BLOCK_K_512_BYTE
DATA_SIZE_FP32     = 4
BASE_STEP          = 1
ITER_COL_FIRST     = 0
BASIC_L1_BUFFER_4  = 4      # BASIC_L1_BUFFER_NUM
L1_SINGLE_SIZE_LIM = 48 * 1024
FP32_MIN_BASE      = 64
CUBE_BOUND_RATIO   = 0.85
EPSILON            = 1e-9
FP32_SPLIT_K_FACTOR = 3
MIN_TAIL_BLOCK     = 4096


def ceil_div(a, b):
    return (a + b - 1) // b

def ceil_align(a, align):
    return (a + align - 1) // align * align

def floor_align(a, align):
    return a // align * align

def floor_div(a, b):
    return a // b


@dataclass
class RunInfo:
    used_core_num: int = 1
    single_core_m: int = 1
    single_core_n: int = 1
    single_core_k: int = 1
    split_k_base: int = 1
    base_m: int = 1
    base_n: int = 1
    base_k: int = 1
    step_ka: int = 1
    step_kb: int = 1
    depth_a1: int = 1
    depth_b1: int = 1
    step_m: int = 1
    step_n: int = 1
    iterate_order: int = 0
    db_l0c: int = 0
    l1_buffer_num: int = 2
    ub_db: int = 1
    m_base_tail_split: int = 1
    n_base_tail_split: int = 1
    m_tail_main: int = 0
    n_tail_main: int = 0
    m_cnt: int = 1   # tailInfo.mCnt
    n_cnt: int = 1   # tailInfo.nCnt
    cube_bound_param: float = 0.0
    cube_bound_edge: float = 0.0


# ═══════════════════════════════════════════════════════════════════════════
#  Phase 1: ResetBase
# ═══════════════════════════════════════════════════════════════════════════

def reset_base(a_dtype_size: int, k_value: int, run: RunInfo):
    run.used_core_num = AIC_NUM
    run.base_m = BASIC_BLOCK_256
    run.base_n = BASIC_BLOCK_256
    run.base_k = K_128_BYTE // a_dtype_size
    run.step_m = BASE_STEP
    run.step_n = BASE_STEP
    run.iterate_order = ITER_COL_FIRST
    run.db_l0c = DB_OFF_SIZE
    run.single_core_k = k_value
    run.single_core_m = run.base_m
    run.single_core_n = run.base_n
    run.m_base_tail_split = 1
    run.n_base_tail_split = 1
    run.m_tail_main = 0
    run.n_tail_main = 0
    run.m_cnt = 1
    run.n_cnt = 1


# ═══════════════════════════════════════════════════════════════════════════
#  Phase 2a: GetMaxBaseWithLimit
# ═══════════════════════════════════════════════════════════════════════════

def get_max_base_with_limit(
    base_mn_buffer_limit: int,
    base_align_unit: int,
    is_right_matrix: bool,
    is_memory_bound: bool,
    m_value: int, n_value: int, k_value: int,
    a_dtype_size: int, b_dtype_size: int,
    has_bias: bool,
    is_a_trans: bool, is_b_trans: bool,
    is_matmul_tiling: bool,
) -> int:
    shape_value = n_value if is_right_matrix else m_value

    k_align_value = ceil_align(k_value, BASIC_BLOCK_16)
    k_limit_value = BASIC_BLOCK_16 if is_memory_bound else (K_128_BYTE // a_dtype_size)
    min_kl0 = min(k_limit_value, k_align_value) * a_dtype_size

    max_base_mn_with_buffer = base_mn_buffer_limit // DATA_SIZE_FP32 // BASIC_BLOCK_16
    max_base_block = min(L0A_SIZE // DB_SIZE // min_kl0, max_base_mn_with_buffer)

    if is_right_matrix and has_bias:
        max_base_block = min(max_base_block, BT_SIZE // DB_SIZE // DATA_SIZE_FP32)

    # K-inner alignment constraint
    is_k_inner = (not is_a_trans) or is_b_trans
    if is_k_inner:
        if is_memory_bound and is_matmul_tiling:
            k_align_unit = K_256_BYTE // a_dtype_size
        else:
            k_align_unit = K_512_BYTE // a_dtype_size
    else:
        k_align_unit = BASIC_BLOCK_16  # NOT divided by dtype size

    max_base_mn_with_k_inner = L1_SIZE // (2 * DB_SIZE * a_dtype_size * min(k_align_unit, k_align_value))
    max_base_block = min(max_base_block, max_base_mn_with_k_inner)

    # Shape constraint
    max_base_block = min(ceil_align(shape_value, base_align_unit), floor_align(max_base_block, base_align_unit))
    if shape_value < base_align_unit:
        max_base_block = min(max_base_block, ceil_align(shape_value, BASIC_BLOCK_16))

    return max_base_block


# ═══════════════════════════════════════════════════════════════════════════
#  Phase 2b: GetBalanceRateWithTail
# ═══════════════════════════════════════════════════════════════════════════

def get_balance_rate_with_tail(
    m_value: int, n_value: int,
    used_core_num: int, base_m: int, base_n: int,
    is_matmul_tiling: bool,
) -> float:
    """Faithful port of GetBalanceRateWithTail from C++."""
    batch = 1  # always 1 for MatMul (non-batch)
    total_round = batch * ceil_div(m_value, base_m) * ceil_div(n_value, base_n)
    if total_round == 0:
        return 0.0
    main_round = ceil_div(total_round, used_core_num) - 1
    # Guard against main_round underflow (totalRound < usedCoreNum → mainRound = 0 after ceil_div-1)
    # In C++ uint64_t, ceil_div(totalRound, usedCoreNum) is 1 when totalRound <= usedCoreNum,
    # so mainRound = 0. When totalRound == 0, mainRound wraps to UINT64_MAX, but we guard above.
    tail_remainder = total_round - used_core_num * main_round
    if tail_remainder <= 0:
        tail_remainder = total_round  # fallback
    total_tail_split = floor_div(used_core_num, tail_remainder) if tail_remainder > 0 else 1
    bm, bn = base_m, base_n
    if m_value <= BASIC_BLOCK_16:
        bm = m_value
    if n_value <= BASIC_BLOCK_16:
        bn = n_value
    # C++ condition: mainRound == 0 || floorDiv(baseM*baseN, totalTailSplit) < MIN_TAIL ||
    #                !IsMatMulTiling
    if main_round == 0 or floor_div(bm * bn, max(total_tail_split, 1)) < MIN_TAIL_BLOCK or not is_matmul_tiling:
        return (batch * m_value * n_value / used_core_num) / ((main_round + 1) * bm * bn)
    tail_split_sqrt = int(math.sqrt(total_tail_split))
    if tail_split_sqrt == 0:
        return (batch * m_value * n_value / used_core_num) / ((main_round + 1) * bm * bn)
    offset = floor_div(total_tail_split - tail_split_sqrt * tail_split_sqrt, tail_split_sqrt) + 1
    denom = tail_split_sqrt * (tail_split_sqrt + offset - 1)
    if denom == 0:
        return (batch * m_value * n_value / used_core_num) / ((main_round + 1) * bm * bn)
    tail_round = 1.0 / denom
    return (m_value * n_value / used_core_num) / ((main_round + tail_round) * bm * bn)


# ═══════════════════════════════════════════════════════════════════════════
#  Phase 2c: GetBaseK
# ═══════════════════════════════════════════════════════════════════════════

def get_base_k(run: RunInfo, k_value: int, a_dtype_size: int, is_a_trans: bool, is_b_trans: bool):
    k_value_align = ceil_align(k_value, BASIC_BLOCK_16)
    max_base_k = L0A_SIZE // DB_SIZE // a_dtype_size // max(run.base_m, run.base_n)
    if k_value_align <= max_base_k:
        run.base_k = k_value_align
        return
    if is_a_trans and not is_b_trans:
        run.base_k = floor_align(max_base_k, BASIC_BLOCK_16)
        return
    if max_base_k * a_dtype_size >= K_256_BYTE:
        run.base_k = floor_align(max_base_k, K_256_BYTE // a_dtype_size)
        return
    for base_k_cand in (128, 64, 32, 16):
        if max_base_k >= base_k_cand:
            run.base_k = base_k_cand
            return
    run.base_k = 16  # fallback


# ═══════════════════════════════════════════════════════════════════════════
#  Phase 2: GetRebalanceBlock  (the core 2-D search)
# ═══════════════════════════════════════════════════════════════════════════

def get_rebalance_block(
    run: RunInfo,
    m_value: int, n_value: int, k_value: int,
    a_dtype_size: int, b_dtype_size: int,
    is_fp32: bool, has_bias: bool,
    is_a_trans: bool, is_b_trans: bool,
    is_matmul_tiling: bool,
):
    single_core_compute_power = CORE_FREQ * 8
    if is_fp32:  # DAV_3510 FP32 has 1/16 compute
        single_core_compute_power /= BASIC_BLOCK_16

    balance_rate_edge = 0.9
    cmr = (m_value + n_value) / (m_value * n_value)
    compute_power = single_core_compute_power * AIC_NUM

    base_mn_buffer_limit = L0C_SIZE  # usedCoreNum == aicNum at this point

    batch_num = 1
    l2_cache_usage = max(
        batch_num * (m_value + n_value) * k_value * a_dtype_size / L2_SIZE,
        1.0,
    )
    run.cube_bound_edge = (
        (L2_BW / compute_power)
        + l2_cache_usage * (1 - L2_BW / HBM_BW) * cmr
        - (1 + L2_BW / HBM_BW) / k_value
    )

    base_m_best = min(ceil_align(m_value, BASIC_BLOCK_16), BASIC_BLOCK_256)
    base_n_best = max(
        BASIC_BLOCK_16,
        min(ceil_align(n_value, BASIC_BLOCK_16),
            floor_align(base_mn_buffer_limit // DATA_SIZE_FP32 // max(base_m_best, 1), BASIC_BLOCK_16)),
    )
    cube_bound_param_best = 1.0 / max(base_m_best, 1) + 1.0 / max(base_n_best, 1)
    is_memory_bound = cube_bound_param_best > run.cube_bound_edge

    inner_align_unit = BASIC_BLOCK_128 if is_memory_bound else BASIC_BLOCK_64

    fixp_bound_edge = (m_value * n_value * HBM_BW) / ((m_value + n_value) * L2_BW)
    base_m_align_unit = (inner_align_unit // a_dtype_size) if is_a_trans else BASIC_BLOCK_16
    if k_value < fixp_bound_edge:
        base_n_align_unit = K_256_BYTE // b_dtype_size
    else:
        base_n_align_unit = BASIC_BLOCK_16 if is_b_trans else (inner_align_unit // b_dtype_size)

    max_base_m = get_max_base_with_limit(
        base_mn_buffer_limit, base_m_align_unit, False, is_memory_bound,
        m_value, n_value, k_value, a_dtype_size, b_dtype_size,
        has_bias, is_a_trans, is_b_trans, is_matmul_tiling,
    )
    max_base_n = get_max_base_with_limit(
        base_mn_buffer_limit, base_n_align_unit, True, is_memory_bound,
        m_value, n_value, k_value, a_dtype_size, b_dtype_size,
        has_bias, is_a_trans, is_b_trans, is_matmul_tiling,
    )

    run.base_m = max(BASIC_BLOCK_16, min(max_base_m, BASIC_BLOCK_256))
    run.base_n = max(
        BASIC_BLOCK_16,
        min(max_base_n, floor_align(base_mn_buffer_limit // DATA_SIZE_FP32 // run.base_m, base_n_align_unit)),
    )
    run.cube_bound_param = 1.0 / run.base_m + 1.0 / run.base_n
    run.cube_bound_edge = run.cube_bound_edge * CUBE_BOUND_RATIO
    balance_rate = get_balance_rate_with_tail(m_value, n_value, run.used_core_num, run.base_m, run.base_n, is_matmul_tiling)

    # 2-D search loop
    cur_base_m = max_base_m
    while cur_base_m >= 1:
        cur_max_base_n = min(
            max_base_n,
            floor_align(base_mn_buffer_limit // DATA_SIZE_FP32 // cur_base_m, base_n_align_unit),
        )
        cur_base_n = cur_max_base_n
        while cur_base_n >= 1:
            cur_cube_bound_param = 1.0 / cur_base_m + 1.0 / cur_base_n
            cur_balance_rate = get_balance_rate_with_tail(
                m_value, n_value, run.used_core_num, cur_base_m, cur_base_n, is_matmul_tiling
            )

            skip_cond = (
                balance_rate >= balance_rate_edge
                and cur_cube_bound_param > run.cube_bound_param
                and cur_cube_bound_param > run.cube_bound_edge
                and run.cube_bound_edge > 0
            )
            used_core_tmp = ceil_div(m_value, cur_base_m) * ceil_div(n_value, cur_base_n)
            skip_cond = skip_cond or (
                is_fp32
                and cur_cube_bound_param < run.cube_bound_edge
                and (
                    (cur_base_m < FP32_MIN_BASE and m_value > cur_base_m)
                    or (cur_base_n < FP32_MIN_BASE and n_value > cur_base_n)
                )
                and used_core_tmp > AIC_NUM
            )
            if skip_cond:
                cur_base_n -= base_n_align_unit
                continue

            cube_bound_cond = cur_cube_bound_param <= run.cube_bound_edge and cur_balance_rate > balance_rate
            balance_cond = (
                (cur_cube_bound_param / max(cur_balance_rate, EPSILON)) < (run.cube_bound_param / max(balance_rate, EPSILON))
                or (
                    abs(cur_cube_bound_param / max(cur_balance_rate, EPSILON) - run.cube_bound_param / max(balance_rate, EPSILON)) < EPSILON
                    and cur_balance_rate > balance_rate
                )
            )
            update_cond = cube_bound_cond or balance_cond
            if update_cond:
                if cube_bound_cond:
                    run.cube_bound_edge = cur_cube_bound_param
                run.base_m = cur_base_m
                run.base_n = cur_base_n
                run.cube_bound_param = cur_cube_bound_param
                balance_rate = cur_balance_rate

            cur_base_n -= base_n_align_unit
        cur_base_m -= base_m_align_unit

    run.base_m = min(ceil_align(m_value, BASIC_BLOCK_16), run.base_m)
    run.base_n = min(ceil_align(n_value, BASIC_BLOCK_16), run.base_n)
    get_base_k(run, k_value, a_dtype_size, is_a_trans, is_b_trans)

    m_core = ceil_div(m_value, run.base_m)
    n_core = ceil_div(n_value, run.base_n)
    run.used_core_num = min(batch_num * m_core * n_core, run.used_core_num)
    run.db_l0c = DB_SIZE if run.base_m * run.base_n * DATA_SIZE_FP32 * DB_SIZE <= L0C_SIZE else 1
    run.ub_db = DB_SIZE if run.base_m * run.base_n * DATA_SIZE_FP32 <= UB_SIZE else 1


# ═══════════════════════════════════════════════════════════════════════════
#  Phase 3: CalcTailBasicBlock
# ═══════════════════════════════════════════════════════════════════════════

def calc_tail_basic_block(run: RunInfo, m_value: int, n_value: int, a_dtype_size: int, b_dtype_size: int, is_a_trans: bool, is_b_trans: bool):
    m_cnt = ceil_div(m_value, run.base_m)
    n_cnt = ceil_div(n_value, run.base_n)
    mn_cnt = m_cnt * n_cnt
    tail_cnt = 0 if mn_cnt <= AIC_NUM else mn_cnt % AIC_NUM
    run.m_cnt = 1
    run.n_cnt = 1
    if tail_cnt != 0:
        while (
            (run.m_cnt + 1) * run.n_cnt * tail_cnt <= AIC_NUM
            and (
                is_a_trans
                or ceil_div(run.base_m, run.m_cnt) * a_dtype_size > K_128_BYTE
            )
        ):
            run.m_cnt += 1
            if (
                run.m_cnt * (run.n_cnt + 1) * tail_cnt <= AIC_NUM
                and (
                    is_b_trans
                    or ceil_div(run.base_n, run.n_cnt) * b_dtype_size > K_128_BYTE
                )
            ):
                run.n_cnt += 1


# ═══════════════════════════════════════════════════════════════════════════
#  Phase 4: CalL1Tiling  (CalL1TilingDefault)
# ═══════════════════════════════════════════════════════════════════════════

def cal_l1_tiling(
    run: RunInfo,
    m_value: int, n_value: int, k_value: int,
    a_dtype_size: int, b_dtype_size: int,
    has_bias: bool, has_scale: bool,
    is_a_trans: bool, is_b_trans: bool,
):
    is_k_inner = (not is_a_trans) or is_b_trans
    total_l1 = L1_SIZE - (run.base_n * DB_SIZE * DATA_SIZE_FP32 if has_bias else 0)
    total_l1 -= run.base_n * 8 if has_scale else 0  # sizeof(uint64_t) = 8

    max_step_k = min(ceil_div(k_value, run.base_k), 8)
    k_align_unit = (K_512_BYTE // a_dtype_size) if is_k_inner else BASIC_BLOCK_16

    res_kl1 = 0
    single_mte_size = 0
    for step_k in range(1, max_step_k + 1):
        cur_kl1 = run.base_k * step_k
        a_l1_size = run.base_m * cur_kl1 * a_dtype_size
        b_l1_size = run.base_n * cur_kl1 * b_dtype_size
        if (a_l1_size + b_l1_size) * DB_SIZE > total_l1:
            break
        if max(a_l1_size, b_l1_size) * DB_SIZE * 2 > L1_SIZE:
            break

        cond_no_res = res_kl1 == 0
        cond_k_align_256b = cur_kl1 % (K_256_BYTE // a_dtype_size) == 0
        cond_k_align = (
            res_kl1 % k_align_unit != 0
            and (cond_k_align_256b or (not cond_k_align_256b and single_mte_size < L1_SINGLE_SIZE_LIM))
        )
        cond_mte_size = (
            res_kl1 % k_align_unit == 0
            and cur_kl1 % k_align_unit == 0
            and single_mte_size < L1_SINGLE_SIZE_LIM
        )
        if cond_no_res or cond_k_align or cond_mte_size:
            res_kl1 = cur_kl1
            single_mte_size = max(a_l1_size, b_l1_size)

    run.step_ka = res_kl1 // run.base_k if run.base_k > 0 else 1
    run.step_kb = run.step_ka
    run.depth_a1 = run.step_ka * DB_SIZE
    run.depth_b1 = run.step_kb * DB_SIZE
    run.single_core_m = run.base_m
    run.single_core_n = run.base_n


# ═══════════════════════════════════════════════════════════════════════════
#  Full-load checks and overrides
# ═══════════════════════════════════════════════════════════════════════════

def check_al1_full_load(
    run: RunInfo,
    m_value: int, n_value: int, k_value: int,
    a_dtype_size: int, is_a_trans: bool, is_b_trans: bool,
    has_bias: bool, is_split_k: bool, l0c2out_on_the_fly: bool,
) -> bool:
    if is_split_k or not l0c2out_on_the_fly:
        return False
    if run.cube_bound_param <= run.cube_bound_edge:
        return False
    m_cnt = ceil_div(m_value, run.single_core_m)
    n_cnt = ceil_div(n_value, run.single_core_n)
    is_k_inner = (not is_a_trans) and is_b_trans
    is_k_aligned = is_k_inner and (k_value * a_dtype_size % K_512_BYTE == 0)
    if n_cnt < AIC_NUM or (not is_k_aligned and n_cnt == AIC_NUM):
        return False
    if k_value <= BASIC_BLOCK_128 and m_cnt != 1:
        return False
    m_aligned = ceil_align(m_value, BASIC_BLOCK_16)
    k_aligned = ceil_align(k_value, BASIC_BLOCK_16)
    a_l1_size = m_aligned * k_aligned * a_dtype_size
    bias_size = run.base_n * DB_SIZE * DATA_SIZE_FP32 if has_bias else 0
    if (a_l1_size + bias_size) > L1_SIZE * 3 // 4:
        return False
    return True


def check_bl1_full_load(
    run: RunInfo,
    m_value: int, n_value: int, k_value: int,
    b_dtype_size: int, is_a_trans: bool, is_b_trans: bool,
    has_bias: bool, is_split_k: bool, l0c2out_on_the_fly: bool,
) -> bool:
    if is_split_k or not l0c2out_on_the_fly:
        return False
    if run.cube_bound_param <= run.cube_bound_edge:
        return False
    m_cnt = ceil_div(m_value, run.single_core_m)
    n_cnt = ceil_div(n_value, run.single_core_n)
    is_k_inner = (not is_a_trans) and is_b_trans
    is_k_aligned = is_k_inner and (k_value * b_dtype_size % K_512_BYTE == 0)
    if m_cnt < AIC_NUM or (not is_k_aligned and m_cnt == AIC_NUM):
        return False
    n_aligned = ceil_align(n_value, BASIC_BLOCK_16)
    k_aligned = ceil_align(k_value, BASIC_BLOCK_16)
    b_l1_size = n_aligned * k_aligned * b_dtype_size
    bias_size = n_aligned * DATA_SIZE_FP32 if has_bias else 0
    if (b_l1_size + bias_size) > L1_SIZE * 3 // 4:
        return False
    return True


def do_al1_full_load(
    run: RunInfo,
    m_value: int, n_value: int, k_value: int,
    a_dtype_size: int, b_dtype_size: int,
    has_bias: bool, is_a_trans: bool, is_b_trans: bool,
    is_matmul_tiling: bool,
):
    """Override tiling for A-L1 full-load scenario."""
    # Reset load balance
    run.m_base_tail_split = 1
    run.n_base_tail_split = 1
    run.m_tail_main = 0
    run.n_tail_main = 0

    m_aligned = ceil_align(m_value, BASIC_BLOCK_16)
    k_aligned = ceil_align(k_value, BASIC_BLOCK_16)
    run.step_m = ceil_div(m_aligned, run.base_m)
    run.step_ka = ceil_div(k_aligned, run.base_k)
    a_l1_size = m_aligned * k_aligned * a_dtype_size
    bias_size = run.base_n * DB_SIZE * DATA_SIZE_FP32 if has_bias else 0
    remain_l1 = L1_SIZE - (a_l1_size + bias_size)
    max_base_n_l1 = remain_l1 // (run.base_k * b_dtype_size * DB_SIZE)
    max_base_n_l0c = L0C_SIZE // (run.base_m * DATA_SIZE_FP32 * DB_SIZE)
    max_base_n = floor_align(min(max_base_n_l1, max_base_n_l0c), BASIC_BLOCK_16)
    if is_matmul_tiling:
        balance_base_n = ceil_align(ceil_div(n_value, AIC_NUM), BASIC_BLOCK_16)
    else:
        balance_base_n = max_base_n
    run.base_n = min(run.base_n, min(max_base_n, balance_base_n))

    base_align_unit = K_128_BYTE // b_dtype_size
    if not is_b_trans and run.base_n > base_align_unit:
        run.base_n = floor_align(run.base_n, base_align_unit)

    max_step_k_buf = remain_l1 // (run.base_k * run.base_n * DB_SIZE * b_dtype_size)
    max_step_k = min(min(ceil_div(k_value, run.base_k), max_step_k_buf), 4)

    if (
        is_b_trans
        and max_step_k == max_step_k_buf
        and max_step_k * run.base_k % K_256_BYTE != 0
        and (run.base_k << 1) * run.base_m * a_dtype_size * DB_SIZE <= L0A_SIZE
    ):
        run.base_k <<= 1
        max_base_n_l1 = remain_l1 // (run.base_k * b_dtype_size * DB_SIZE)
        max_base_n_l0b = L0B_SIZE // (run.base_k * b_dtype_size * DB_SIZE)
        run.base_n = min(run.base_n, max_base_n_l0b)
        run.step_ka = ceil_div(k_aligned, run.base_k)

    step_k = 1
    while step_k < max_step_k:
        cur_kl1 = run.base_k * step_k
        used_l1 = run.base_n * cur_kl1 * b_dtype_size
        if used_l1 < L1_SINGLE_SIZE_LIM:
            step_k += 1
            continue
        if is_b_trans and cur_kl1 % (K_256_BYTE // b_dtype_size) != 0:
            step_k += 1
            continue
        break

    run.step_kb = step_k
    run.depth_a1 = run.step_m * run.step_ka
    run.depth_b1 = run.step_kb * DB_SIZE
    run.single_core_m = m_value
    run.single_core_n = run.base_n

    bl1_4buf = run.base_k * run.step_kb * run.base_n * b_dtype_size * BASIC_L1_BUFFER_4
    bias_4buf = (bias_size // DB_SIZE * BASIC_L1_BUFFER_4) if has_bias else 0
    run.l1_buffer_num = DB_SIZE if (bl1_4buf + a_l1_size + bias_4buf) > L1_SIZE else BASIC_L1_BUFFER_4
    run.db_l0c = DB_SIZE if run.base_m * run.base_n * DATA_SIZE_FP32 * DB_SIZE <= L0C_SIZE else 1

    n_core = ceil_div(n_value, run.base_n)
    run.used_core_num = min(n_core, AIC_NUM)

    # CalcTailBasicBlockAL1Full
    n_cnt_total = ceil_div(n_value, run.base_n)
    tail_cnt = 0 if n_cnt_total <= AIC_NUM else n_cnt_total % AIC_NUM
    run.m_cnt = 1
    run.n_cnt = 1
    if tail_cnt != 0:
        while (
            (run.n_cnt + 1) * tail_cnt <= AIC_NUM
            and (is_b_trans or ceil_div(run.base_n, run.n_cnt) * b_dtype_size > K_128_BYTE)
        ):
            run.n_cnt += 1


def do_bl1_full_load(
    run: RunInfo,
    m_value: int, n_value: int, k_value: int,
    a_dtype_size: int, b_dtype_size: int,
    has_bias: bool, is_a_trans: bool, is_b_trans: bool,
    is_matmul_tiling: bool,
):
    """Override tiling for B-L1 full-load scenario."""
    run.m_base_tail_split = 1
    run.n_base_tail_split = 1
    run.m_tail_main = 0
    run.n_tail_main = 0

    n_aligned = ceil_align(n_value, BASIC_BLOCK_16)
    k_aligned = ceil_align(k_value, BASIC_BLOCK_16)
    run.step_n = ceil_div(n_aligned, run.base_n)
    run.step_kb = ceil_div(k_aligned, run.base_k)
    b_l1_size = n_aligned * k_aligned * b_dtype_size
    bias_size = n_aligned * DATA_SIZE_FP32 if has_bias else 0
    remain_l1 = L1_SIZE - (b_l1_size + bias_size)
    max_base_m_l1 = remain_l1 // (run.base_k * a_dtype_size * DB_SIZE)
    max_base_m_l0c = L0C_SIZE // (run.base_n * DATA_SIZE_FP32 * DB_SIZE)
    max_base_m = floor_align(min(max_base_m_l1, max_base_m_l0c), BASIC_BLOCK_16)
    if is_matmul_tiling:
        balance_base_m = ceil_align(ceil_div(m_value, AIC_NUM), BASIC_BLOCK_16)
    else:
        balance_base_m = max_base_m
    run.base_m = min(run.base_m, min(max_base_m, balance_base_m))

    base_align_unit = K_128_BYTE // b_dtype_size
    if is_a_trans and run.base_m > base_align_unit:
        run.base_m = floor_align(run.base_m, base_align_unit)

    max_step_k_buf = remain_l1 // (run.base_k * run.base_m * DB_SIZE * a_dtype_size)
    max_step_k = min(min(ceil_div(k_value, run.base_k), max_step_k_buf), 4)

    if (
        not is_a_trans
        and max_step_k == max_step_k_buf
        and max_step_k * run.base_k % K_256_BYTE != 0
        and (run.base_k << 1) * run.base_n * b_dtype_size * DB_SIZE <= L0B_SIZE
    ):
        run.base_k <<= 1
        max_base_m_l1 = remain_l1 // (run.base_k * a_dtype_size * DB_SIZE)
        max_base_m_l0a = L0A_SIZE // (run.base_k * a_dtype_size * DB_SIZE)
        run.base_m = min(run.base_m, max_base_m_l0a)
        run.step_kb = ceil_div(k_aligned, run.base_k)

    step_k = 1
    while step_k < max_step_k:
        cur_kl1 = run.base_k * step_k
        used_l1 = run.base_m * cur_kl1 * a_dtype_size
        if used_l1 < L1_SINGLE_SIZE_LIM:
            step_k += 1
            continue
        if not is_a_trans and cur_kl1 % (K_256_BYTE // a_dtype_size) != 0:
            step_k += 1
            continue
        break

    run.step_ka = step_k
    run.depth_a1 = run.step_ka * DB_SIZE
    run.depth_b1 = run.step_n * run.step_kb
    run.single_core_m = run.base_m
    run.single_core_n = n_value

    al1_4buf = run.base_k * run.step_ka * run.base_m * a_dtype_size * BASIC_L1_BUFFER_4
    bias_4buf = (bias_size // DB_SIZE * BASIC_L1_BUFFER_4) if has_bias else 0
    run.l1_buffer_num = DB_SIZE if (al1_4buf + b_l1_size + bias_4buf) > L1_SIZE else BASIC_L1_BUFFER_4
    run.db_l0c = DB_SIZE if run.base_m * run.base_n * DATA_SIZE_FP32 * DB_SIZE <= L0C_SIZE else 1
    run.ub_db = DB_SIZE if run.base_m * run.base_n * DATA_SIZE_FP32 <= UB_SIZE else 1

    m_core = ceil_div(m_value, run.base_m)
    run.used_core_num = min(m_core, AIC_NUM)

    # CalcTailBasicBlockBL1Full
    m_cnt_total = ceil_div(m_value, run.base_m)
    tail_cnt = 0 if m_cnt_total <= AIC_NUM else m_cnt_total % AIC_NUM
    run.m_cnt = 1
    run.n_cnt = 1
    if tail_cnt != 0:
        while (
            (run.m_cnt + 1) * tail_cnt <= AIC_NUM
            and (is_a_trans or ceil_div(run.base_m, run.m_cnt) * a_dtype_size > K_128_BYTE)
        ):
            run.m_cnt += 1


# ═══════════════════════════════════════════════════════════════════════════
#  L0C2Out determination
# ═══════════════════════════════════════════════════════════════════════════

def get_l0c2out_dav3510(
    m_value: int, n_value: int, k_value: int,
    c_dtype_size: int, a_type_is_fp16_or_bf16: bool,
    is_split_k: bool,
    run: RunInfo,
) -> str:
    """Returns 'ON_THE_FLY', 'ND_FIXPIPE_1_1', or 'ND_FIXPIPE_1_2'."""
    is_valid_mkn = k_value <= BASIC_BLOCK_256 and m_value >= BASIC_BLOCK_256
    m_cnt = ceil_div(m_value, run.single_core_m)
    n_cnt = ceil_div(n_value, run.single_core_n)
    is_multi_round = m_cnt * n_cnt > AIC_NUM
    is_unaligned_n = (n_value * c_dtype_size) % 128 != 0 and (n_value * c_dtype_size) > BASIC_BLOCK_256
    fixpipe_bound = is_valid_mkn and is_multi_round and is_unaligned_n
    if not fixpipe_bound or (AIV_NUM != AIC_NUM * 2) or is_split_k:
        return "ON_THE_FLY"
    if a_type_is_fp16_or_bf16:
        return "ND_FIXPIPE_1_1"
    return "ND_FIXPIPE_1_2"


# ═══════════════════════════════════════════════════════════════════════════
#  SplitK check  (IsSplitSingleCoreK)
# ═══════════════════════════════════════════════════════════════════════════

def is_split_single_core_k(k_value: int, is_fp32: bool, is_nd_format: bool, is_tensor_api: bool) -> bool:
    real_single_core_k = k_value
    if is_fp32 and is_nd_format and is_tensor_api and real_single_core_k >= AIC_NUM * BASIC_BLOCK_16:
        return True
    return False


# ═══════════════════════════════════════════════════════════════════════════
#  SelectSplitKBase
# ═══════════════════════════════════════════════════════════════════════════

def select_split_k_base(m_value: int, n_value: int, k_value: int, base_k: int, deterministic_level: int = 0) -> int:
    split_k_list = [64, 128, 256, 512, 1024, 2048, 4096, 8192]
    is_small_mnk = (
        ceil_align(m_value, BASIC_BLOCK_16) * ceil_align(n_value, BASIC_BLOCK_16) <= BASIC_BLOCK_256 * BASIC_BLOCK_256
        and k_value < FP32_SPLIT_K_FACTOR * split_k_list[0] * split_k_list[0]
        and deterministic_level <= 1
    )
    if is_small_mnk:
        return max(split_k_list[1], base_k)
    for i in range(len(split_k_list) - 1):
        threshold = FP32_SPLIT_K_FACTOR * split_k_list[i] * split_k_list[i]
        if k_value < threshold:
            return max(split_k_list[i], base_k)
    return max(split_k_list[-1], base_k)


# ═══════════════════════════════════════════════════════════════════════════
#  StreamK tiling
# ═══════════════════════════════════════════════════════════════════════════

def check_streamk_sk_tiling(m_value, n_value, k_value, a_dtype_size, is_fp32, is_hf32):
    """CheckStreamKSKTiling — returns True if StreamK-SK should be selected."""
    stream_k_min_k = 8192
    k_aligned_256 = ceil_align(k_value, BASIC_BLOCK_256)
    threshold = max(stream_k_min_k, AIC_NUM * K_256_BYTE) // a_dtype_size
    if k_aligned_256 < threshold:
        return False

    m_aligned = ceil_align(m_value, BASIC_BLOCK_16)
    n_aligned = ceil_align(n_value, BASIC_BLOCK_16)
    k_aligned_8 = ceil_align(k_value, 8)
    align_value = BASIC_BLOCK_256
    if is_fp32 and not is_hf32:
        if k_aligned_8 >= BASIC_BLOCK_16 * AIC_NUM * AIC_NUM:
            align_value = FP32_MIN_BASE  # 64
        else:
            align_value = 32  # BLOCK_BYTE_SIZE

    is_sk_support = (
        m_aligned * n_aligned < 65536  # FP32_MN_SHAPE_THRESHOLD
        and k_aligned_8 >= BASIC_BLOCK_16 * AIC_NUM * AIC_NUM
        and is_fp32
    )
    m_cnt = ceil_div(m_value, align_value)
    n_cnt = ceil_div(n_value, align_value)
    if m_cnt * n_cnt > AIC_NUM // 2 and not is_sk_support:
        return False
    return True


def check_streamk_dpsk_tiling(m_value, n_value, k_value, a_dtype_size):
    """CheckStreamKDPSKTiling — returns True if StreamK-DPSK should be selected."""
    stream_k_min_k = 8192
    if m_value % BASIC_BLOCK_256 != 0 or n_value % BASIC_BLOCK_256 != 0:
        return False
    if k_value < max(stream_k_min_k, AIC_NUM * K_128_BYTE) // a_dtype_size:
        return False
    m_cnt = ceil_div(m_value, BASIC_BLOCK_256)
    n_cnt = ceil_div(n_value, BASIC_BLOCK_256)
    total_mn = m_cnt * n_cnt
    return total_mn >= AIC_NUM and total_mn % AIC_NUM != 0 and total_mn % AIC_NUM <= AIC_NUM // 2


def is_streamk_capable(m_value, n_value, k_value, dtype, is_a_trans, is_b_trans, deterministic_level=0):
    """BasicStreamKTiling::IsCapable."""
    if deterministic_level > 1:
        return False
    # aFormat == ND is assumed (inductor always uses ND)
    # IsSelfNonContiguous is assumed false for standard mm
    if AIV_NUM != AIC_NUM * 2:
        return False
    dtype_size_map = {"float16": 2, "bfloat16": 2, "float32": 4, "int8": 1}
    a_dtype_size = dtype_size_map.get(dtype, 2)
    is_fp32 = (dtype == "float32")
    is_hf32 = False  # hf32 not used by default
    return (
        check_streamk_sk_tiling(m_value, n_value, k_value, a_dtype_size, is_fp32, is_hf32)
        or check_streamk_dpsk_tiling(m_value, n_value, k_value, a_dtype_size)
    )


def get_streamk_l0c2out_flag(m_value, n_value):
    """GetL0C2OutFlag for StreamK."""
    if n_value > BASIC_BLOCK_64 and n_value % BASIC_BLOCK_16 != 0 and m_value > 2 and m_value * n_value >= BASIC_BLOCK_256:
        return "ND_FIXPIPE_1_2"
    return "ON_THE_FLY"


def is_halve_base_profitable(
    m_value, n_value, k_value, a_dtype_size,
    m_cnt, n_cnt, total_mn_cnt, halve_n,
):
    """IsHalveBaseProfitable from StreamK tiling."""
    k_cnt_old = floor_div(AIC_NUM, total_mn_cnt)
    k_cnt_new = floor_div(AIC_NUM, total_mn_cnt * 2)
    if k_cnt_new == 0 or k_cnt_new >= k_cnt_old:
        return False
    # fp32 precision splitK threshold check
    if a_dtype_size == DATA_SIZE_FP32:
        if ceil_div(k_value, k_cnt_new) >= AIC_NUM * BASIC_BLOCK_16:
            return False
    is_nd_fixpipe = get_streamk_l0c2out_flag(m_value, n_value) == "ND_FIXPIPE_1_2"
    fix_width_old = n_value
    fix_width_new = n_value
    if is_nd_fixpipe:
        fix_width_old = _get_padded_fix_width(n_value, n_cnt)
        fix_width_new = _get_padded_fix_width(n_value, n_cnt * 2) if halve_n else fix_width_old
    fixup_old = k_cnt_old * (n_value + fix_width_old)
    fixup_new = k_cnt_new * (n_value + fix_width_new)
    if fixup_new >= fixup_old:
        return False
    save_bytes = 4 * m_value * (fixup_old - fixup_new)
    dup_bytes = n_cnt * m_value * k_value * a_dtype_size if halve_n else m_cnt * n_value * k_value * a_dtype_size
    if is_nd_fixpipe:
        return save_bytes > dup_bytes * 2
    return save_bytes > dup_bytes * 2  # deployed CANN 9.2.0.B111 requires 2x margin universally


def _get_padded_fix_width(n_value, n_cnt):
    """GetPaddedFixWidth for ND_FIXPIPE_1_2 alignment padding."""
    base_n = ceil_align(ceil_div(n_value, n_cnt), BASIC_BLOCK_16)
    full_cnt = ceil_div(n_value, base_n) - 1
    tail_n = n_value - full_cnt * base_n
    return full_cnt * ceil_align(base_n, 32) + ceil_align(tail_n, 32)


def do_streamk_tiling(
    run: RunInfo,
    m_value: int, n_value: int, k_value: int,
    a_dtype_size: int, b_dtype_size: int,
    is_fp32: bool, has_bias: bool,
    is_a_trans: bool, is_b_trans: bool,
):
    """BasicStreamKTiling::DoOpTiling."""
    # ResetBase already called by caller

    m_cnt = ceil_div(m_value, run.base_m)
    n_cnt = ceil_div(n_value, run.base_n)
    total_mn_cnt = m_cnt * n_cnt

    if total_mn_cnt <= AIC_NUM // 2:
        # Adjust mCnt/nCnt to near aicNum/2 when in the sweet zone
        if m_cnt > AIC_NUM // 3 and m_cnt < AIC_NUM // 2:
            m_cnt = AIC_NUM // 2
        if n_cnt > AIC_NUM // 3 and n_cnt < AIC_NUM // 2:
            n_cnt = AIC_NUM // 2
        total_mn_cnt = m_cnt * n_cnt

        if total_mn_cnt <= 2:
            base_m_tmp = ceil_align(ceil_div(m_value, m_cnt), BASIC_BLOCK_16)
            base_n_tmp = ceil_align(ceil_div(n_value, n_cnt), BASIC_BLOCK_16)
            if base_n_tmp >= base_m_tmp and base_n_tmp > BASIC_BLOCK_16:
                if is_halve_base_profitable(m_value, n_value, k_value, a_dtype_size,
                                            m_cnt, n_cnt, total_mn_cnt, True):
                    n_cnt *= 2
            elif base_m_tmp > BASIC_BLOCK_16:
                if is_halve_base_profitable(m_value, n_value, k_value, a_dtype_size,
                                            m_cnt, n_cnt, total_mn_cnt, False):
                    m_cnt *= 2
            total_mn_cnt = m_cnt * n_cnt

        run.base_m = ceil_align(ceil_div(m_value, m_cnt), BASIC_BLOCK_16)
        run.base_n = ceil_align(ceil_div(n_value, n_cnt), BASIC_BLOCK_16)
        k_cnt = floor_div(AIC_NUM, total_mn_cnt)
        run.single_core_k = ceil_div(k_value, k_cnt)
    else:
        # DPSK path
        k_cnt = AIC_NUM // (total_mn_cnt % AIC_NUM) if (total_mn_cnt % AIC_NUM) != 0 else 1
        sk_single_core_k = ceil_div(k_value, k_cnt)
        k_cnt = ceil_div(k_value, sk_single_core_k)
        run.single_core_k = sk_single_core_k

    # baseK computation
    base_k_align = (BASIC_BLOCK_128 // a_dtype_size) if (not is_a_trans or is_b_trans) else BASIC_BLOCK_16
    k_value_max = floor_align(
        L0A_SIZE // DB_SIZE // a_dtype_size // max(run.base_m, run.base_n),
        base_k_align,
    )
    run.base_k = min(run.single_core_k, k_value_max)

    # CalL1Tiling
    cal_l1_tiling(run, m_value, n_value, k_value, a_dtype_size, b_dtype_size, has_bias, False, is_a_trans, is_b_trans)

    # depthb1 < deptha1 adjustment
    if run.base_m == run.base_n and run.depth_b1 == run.depth_a1 * 2:
        run.depth_a1 = run.depth_a1 * 2
        run.depth_b1 = run.depth_b1 // 2
        run.step_kb = run.depth_b1 // DB_SIZE
        run.step_ka = run.depth_a1 // DB_SIZE

    # Reserve L1 for bias when many tiles
    if total_mn_cnt > AIC_NUM and has_bias:
        run.step_kb = 3
        run.step_ka = 3

    # usedCoreNum
    run.used_core_num = AIC_NUM

    # dbL0C and ubDB
    run.db_l0c = DB_SIZE if run.base_m * run.base_n * DATA_SIZE_FP32 * DB_SIZE <= L0C_SIZE else 1

    # splitK check
    is_tensor_api = True  # MatMulV3 with tensor API
    is_nd_format = True
    real_single_core_k = run.single_core_k if run.single_core_k != 0 else k_value
    if is_fp32 and is_nd_format and is_tensor_api and real_single_core_k >= AIC_NUM * BASIC_BLOCK_16:
        run.split_k_base = select_split_k_base(m_value, n_value, k_value, run.base_k)
    else:
        run.split_k_base = k_value


# ═══════════════════════════════════════════════════════════════════════════
#  Main entry point
# ═══════════════════════════════════════════════════════════════════════════

def _get_optimized_tiling(
    m: int,
    n: int,
    k: int,
    dtype: str = "float16",
    op_type: str = "mm",
    is_a_trans: bool = False,
    is_b_trans: bool = False,
    has_bias: bool = False,
) -> dict:
    """
    Compute tiling.
    Parameters
    ----------
    m, n, k : int
        Matrix dimensions.  A is [M, K], B is [K, N], C is [M, N].
    dtype : str
        "float16", "bfloat16", "float32", "int8".
    op_type : str
        "mm" or "addmm" (currently treated identically).
    is_a_trans : bool
        Whether A is transposed (K inner → M inner).
    is_b_trans : bool
        Whether B is transposed (N inner → K inner).
    has_bias : bool
        Whether bias is present.

    Returns
    -------
    dict with keys: mL1, nL1, kL1, l1BufferNum, l0cDB,
                    baseM, baseN, baseK, stepM, stepN, stepKa, stepKb,
                    usedCoreNum, dbL0C, ubDB
    """
    # Dtype sizes
    dtype_size_map = {"float16": 2, "bfloat16": 2, "float32": 4, "int8": 1}
    a_dtype_size = b_dtype_size = dtype_size_map.get(dtype, 2)
    is_fp32 = (dtype == "float32")
    c_dtype_size = 4 if is_fp32 else 2  # output dtype (fp32 accumulates to fp16/fp32)
    if dtype == "float32":
        c_dtype_size = 4

    is_matmul_tiling = True   # batchA == 1 for mm
    is_nd_format = True       # inductor always uses ND format
    is_tensor_api = True      # MatMulV3 with tensor API

    run = RunInfo()

    # ── Check if StreamK strategy should be used (priority 3 > BASIC_ASWT priority 4) ──
    use_streamk = is_streamk_capable(m, n, k, dtype, is_a_trans, is_b_trans)

    if use_streamk:
        # ── StreamK path ──
        reset_base(a_dtype_size, k, run)
        do_streamk_tiling(run, m, n, k, a_dtype_size, b_dtype_size, is_fp32, has_bias, is_a_trans, is_b_trans)
    else:
        # ── BASIC_ASWT path ──
        # ── Phase 1: ResetBase ──
        reset_base(a_dtype_size, k, run)

        # ── Phase 2: GetRebalanceBlock ──
        get_rebalance_block(
            run, m, n, k, a_dtype_size, b_dtype_size,
            is_fp32, has_bias, is_a_trans, is_b_trans, is_matmul_tiling,
        )

        # ── Phase 3: CalcTailBasicBlock ──
        calc_tail_basic_block(run, m, n, a_dtype_size, b_dtype_size, is_a_trans, is_b_trans)

        # ── Phase 4: CalL1Tiling ──
        cal_l1_tiling(run, m, n, k, a_dtype_size, b_dtype_size, has_bias, False, is_a_trans, is_b_trans)

        # ── Phase 5: BasicAswt overrides (full-load, fixpipe, splitK) ──
        is_split_k = is_split_single_core_k(k, is_fp32, is_nd_format, is_tensor_api)
        l0c2out = get_l0c2out_dav3510(m, n, k, c_dtype_size, dtype in ("float16", "bfloat16"), is_split_k, run)

        if check_al1_full_load(run, m, n, k, a_dtype_size, is_a_trans, is_b_trans, has_bias, is_split_k, l0c2out == "ON_THE_FLY"):
            do_al1_full_load(run, m, n, k, a_dtype_size, b_dtype_size, has_bias, is_a_trans, is_b_trans, is_matmul_tiling)
        elif check_bl1_full_load(run, m, n, k, b_dtype_size, is_a_trans, is_b_trans, has_bias, is_split_k, l0c2out == "ON_THE_FLY"):
            do_bl1_full_load(run, m, n, k, a_dtype_size, b_dtype_size, has_bias, is_a_trans, is_b_trans, is_matmul_tiling)
        elif l0c2out == "ON_THE_FLY":
            # ATrans stepK adjustment
            if is_a_trans and not is_b_trans:
                remain_for_al1_bl1 = (L1_SIZE - BASIC_BLOCK_256 * DATA_SIZE_FP32) if has_bias else L1_SIZE
                new_step_ka = remain_for_al1_bl1 // 2 // ((run.base_m + run.base_n) * run.base_k) // a_dtype_size
                if new_step_ka > 0:
                    run.step_ka = new_step_ka
                    run.step_kb = run.step_ka
                    run.depth_a1 = run.step_ka * DB_SIZE
                    run.depth_b1 = run.step_kb * DB_SIZE
            if is_split_k:
                run.split_k_base = select_split_k_base(m, n, k, run.base_k)
            else:
                run.split_k_base = k
        else:
            # fixpipe optimization path
            run.split_k_base = k

    # ── Phase 6: GetTilingDataProcess (MatMulV3BasicTilingData) ──
    mL1 = min(ceil_align(m, BASIC_BLOCK_16), run.base_m * run.step_m)
    nL1 = min(ceil_align(n, BASIC_BLOCK_16), run.base_n * run.step_n)
    step_ka_min = min(run.step_kb, run.step_ka)
    step_ka_min = min(step_ka_min, 4)
    kL1 = run.base_k * step_ka_min

    return {
        "mL1": mL1,
        "nL1": nL1,
        "kL1": kL1,
        "l1BufferNum": run.l1_buffer_num,
        "l0cDB": run.db_l0c,
        "baseM": run.base_m,
        "baseN": run.base_n,
        "baseK": run.base_k,
        "stepM": run.step_m,
        "stepN": run.step_n,
        "stepKa": run.step_ka,
        "stepKb": run.step_kb,
        "usedCoreNum": run.used_core_num,
        "ubDB": run.ub_db,
        "splitKBase": run.split_k_base,
    }


# ═══════════════════════════════════════════════════════════════════════════
#  Wrapper for inductor integration (accepts torch.dtype)
# ═══════════════════════════════════════════════════════════════════════════

# Mapping from torch.dtype string representation to aclnn dtype string
_DTYPE_STR_MAP = {
    "torch.float16": "float16",
    "torch.bfloat16": "bfloat16",
    "torch.float32": "float32",
    "torch.int8": "int8",
    "float16": "float16",
    "bfloat16": "bfloat16",
    "float32": "float32",
    "int8": "int8",
}


def get_optimized_tiling(
    m: int,
    n: int,
    k: int,
    dtype=None,
    op_type: str = "mm",
    is_a_trans: bool = False,
    is_b_trans: bool = False,
    has_bias: bool = False,
) -> dict:
    """Wrapper for _get_optimized_tiling that accepts torch.dtype or str.

    Used by inductor's mm.py to get aclnn-quality tiling configs.
    Returns the same dict as _get_optimized_tiling.
    """
    # Convert dtype to string
    if dtype is None:
        dtype_str = "float16"
    elif isinstance(dtype, str):
        dtype_str = dtype
    else:
        # torch.dtype → str(dtype) gives "torch.float16" etc.
        dtype_str = _DTYPE_STR_MAP.get(str(dtype), "float16")

    return _get_optimized_tiling(
        m, n, k, dtype_str, op_type, is_a_trans, is_b_trans, has_bias
    )
