import math
import warnings

from catlass_cppgen.catlass.arch.arch import Arch
from catlass_cppgen.catlass.gemm_coord import GemmShape
from catlass_cppgen.catlass.gemm.dispatch_policy import MmadMultiBatch
from torch_npu._inductor import config


def _generate_tile_desc(Block_M, Block_N, Block_K_l1, Block_K_l0):
    return [GemmShape(Block_M, Block_N, Block_K_l1), GemmShape(Block_M, Block_N, Block_K_l0)]


def _is_support_tile_autotune(kernel):
    return kernel.slice_axis != "K" and "GroupedMatmul" not in kernel.gen_kernel_name()


def may_adjust_l1_tileshape(dtype_size, l1_m, l1_n, l1_k, l1_stages=2, l1_size=1 << 19):
    l1_bias_size = l1_n * dtype_size
    while ((l1_m + l1_n) * l1_k * dtype_size * l1_stages + l1_bias_size) > l1_size:
        if l1_m <= 16 and l1_n <= 16:
            l1_k -= 16
        elif l1_m >= l1_n:
            l1_m -= 16
        else:
            l1_n -= 16
    return (l1_m, l1_n, l1_k)


class Config:
    __slots__ = ["tile_desc", "block_swizzle"]

    def __init__(self, tile_desc, blk_swizzle):
        self.tile_desc = tile_desc
        self.block_swizzle = blk_swizzle


class TileAutotune:

    _supported_archs = [
        Arch.AtlasA2,
        Arch.AtlasA5,
    ]

    def __init__(self, arch_type):
        self.arch_type = arch_type
        # default stages
        self.L1Stages: int = 2
        self.L0AStages: int = 2
        self.L0BStages: int = 2
        self.L0CStages: int = 1
        self.aicCoreNum: int = config.num_cube_core
        self.init_l1_l0_size(arch_type)

    def init_l1_l0_size(self, arch_type):
        if arch_type not in self._supported_archs:
            warnings.warn(
                f"Unknown arch type to get specific tile size: {arch_type}."
                f"Will use the default tile size to generate tile configs."
            )
            arch_type = Arch.AtlasA2

        if arch_type == Arch.AtlasA2:
            self.L1Size = 512 * 1024
            self.L0CSize = 128 * 1024
            self.L0ASize = 64 * 1024
            self.L0BSize = 64 * 1024
        if arch_type == Arch.AtlasA5:
            self.L1Size = 512 * 1024
            self.L0CSize = 256 * 1024
            self.L0ASize = 64 * 1024
            self.L0BSize = 64 * 1024

    @staticmethod
    def floor_power_of_2(n):
        if n <= 1:
            return n
        try:
            return 1 << (n.bit_length() - 1)
        except AttributeError:
            import sympy
            
            exp = sympy.floor(sympy.log(n, 2))
            return 2 ** exp

    @staticmethod
    def align_down(n, alignment=16):
        return n // alignment * alignment

    @staticmethod
    def align_up(n, alignment=16):
        return (n + alignment - 1) // alignment * alignment

    @staticmethod
    def aligned_decrease_iterator(value, alignment=16, ratio=0.5, down_limit=0):
        if value < alignment:
            yield alignment
            return

        current = (value // alignment) * alignment
        target = max(TileAutotune.align_down(int(value * ratio), alignment=16), 16)
        if down_limit > 0:
            target = min(target, down_limit)
        while current >= target:
            yield current
            current -= alignment

    def tile_heuristics(self, shape_desc, configs, num_left=4):
        # NB: This heuristics only consider m&n splitting situation
        # Heuristics principle:
        # 1. Calculate the num iteration that needs to scheduler and the ratio of tail_blocks/aicCoreNum
        # 2. When the iter is the same, comparison priority a). ratio>=0.8; b) l1_k; c) l0_k
        # 3. If the above comparson is equal, select smallest l1_n when l1_m is the same;
        #    select smallest l1_m when l1_n is the same
        # 4. Use `num_left` to limit the final number of configs

        M, N, _, _ = shape_desc

        def calculate_blocks(cfg):
            m_blk = math.ceil(M / cfg[0])
            n_blk = math.ceil(N / cfg[1])
            return m_blk * n_blk

        def calculate_iter(cfg):
            m_iter = math.ceil(M / cfg[0])
            n_iter = math.ceil(N / cfg[1])
            return math.ceil((m_iter * n_iter) / self.aicCoreNum)

        def update_list(cfg_list, new_cfg):
            l1m = new_cfg[0]
            l1n = new_cfg[1]
            m2n = {cfg[0]: (i, cfg[1]) for i, cfg in enumerate(cfg_list)}
            n2m = {cfg[1]: (i, cfg[0]) for i, cfg in enumerate(cfg_list)}

            if l1m not in m2n and l1n not in n2m:
                cfg_list.append(cfg)
            elif l1m in m2n:
                if l1n < m2n[l1m][1]:
                    idx = m2n[l1m][0]
                    del cfg_list[idx]
                    cfg_list.append(new_cfg)
            elif l1n in n2m:
                if l1m < n2m[l1n][1]:
                    idx = n2m[l1n][0]
                    del cfg_list[idx]
                    cfg_list.append(new_cfg)

        best_by_iter = {}
        for cfg in configs:
            num_iter = calculate_iter(cfg)
            tail_blocks = calculate_blocks(cfg) % self.aicCoreNum
            use_ratio = 1 if tail_blocks == 0 else float(tail_blocks) / self.aicCoreNum
            metric = 1 if use_ratio >= 0.8 else 0
            key = (metric, cfg[2], cfg[3])

            if num_iter not in best_by_iter:
                best_by_iter[num_iter] = (key, [cfg])
            else:
                best_key, cfg_list = best_by_iter[num_iter]
                if key > best_key:
                    best_by_iter[num_iter] = (key, [cfg])
                elif key == best_key:
                    # l1k & l0k are the same
                    update_list(cfg_list, cfg)

        final_configs = []
        for num_iter in sorted(best_by_iter.keys()):
            _, cfg_list = best_by_iter[num_iter]
            curr_num = len(final_configs)
            if num_left is not None:
                if (curr_num + len(cfg_list)) >= num_left:
                    final_configs.extend(cfg_list)
                    break
            final_configs.extend(cfg_list)

        return final_configs

    def filter_configs(self, shape_desc, dispatch_policy, configs):
        M, N, K, _ = shape_desc

        def filter_mmad_multi_batch(tile_config) -> bool:
            """Filter for MmadMultiBatch: l1_m >= M && l1_n >= N && l1_k >= K"""
            return tile_config[0] >= M and tile_config[1] >= N and tile_config[2] >= K

        def filter_small(tile_config) -> bool:
            """Filter for MmadSmall: checks core count constraints and dimensions"""
            a = math.ceil(M / tile_config[0])
            b = math.ceil(N / tile_config[1])
            return a * b <= self.aicCoreNum and tile_config[2] >= K

        filter_funcs = {
            MmadMultiBatch: filter_mmad_multi_batch,
        }
        if dispatch_policy not in filter_funcs:
            return configs

        filter_func = filter_funcs[dispatch_policy]
        return [cfg for cfg in configs if filter_func(cfg)]

    def gen_tile_configs(self, dispatch_policy, dtype_size, shape_desc):
        self.L1Stages = dispatch_policy.l1a_stages
        self.L0AStages = dispatch_policy.l0a_stages
        self.L0BStages = dispatch_policy.l0b_stages
        self.L0CStages = dispatch_policy.l0c_stages

        L1Size = self.L1Size // self.L1Stages
        L0CSize = self.L0CSize // self.L0CStages
        L0BSize = self.L0BSize // self.L0BStages
        L0ASize = self.L0ASize // self.L0AStages
        L0Size = min(L0ASize, L0BSize)

        configs = set()
        M, N, K, _ = shape_desc
        min_mn, max_mn = min(M, N), max(M, N)

        # helper method
        def add_block_sizes_to_configs(BLOCK_min_mn):
            BLOCK_max_mn_start = self.align_down(
                min(max_mn, int(L0CSize // BLOCK_min_mn // 4))
            )
            down_limit = self.align_down(math.ceil(BLOCK_max_mn_start / 16))
            for BLOCK_max_mn in self.aligned_decrease_iterator(
                BLOCK_max_mn_start, down_limit=down_limit
            ):
                BLOCK_K_start = self.floor_power_of_2(
                    min(K, L1Size // (BLOCK_min_mn + BLOCK_max_mn) // dtype_size)
                )

                BLOCK_K = max(BLOCK_K_start, 16)
                SUB_BLOCK_K = L0Size // max(BLOCK_min_mn, BLOCK_max_mn) // dtype_size
                if SUB_BLOCK_K < 16:
                    continue

                SUB_BLOCK_K = max(min(self.align_down(SUB_BLOCK_K), BLOCK_K), 16)
                if M < N:
                    BLOCK_M, BLOCK_N = BLOCK_min_mn, BLOCK_max_mn
                else:
                    BLOCK_M, BLOCK_N = BLOCK_max_mn, BLOCK_min_mn

                if isinstance(dispatch_policy, MmadMultiBatch):
                    # this policy need l1_k == l0_k
                    configs.add((BLOCK_M, BLOCK_N, SUB_BLOCK_K, SUB_BLOCK_K))
                else:
                    configs.add((BLOCK_M, BLOCK_N, BLOCK_K, SUB_BLOCK_K))

        # accumulator type is float or int32
        upper_limit = self.floor_power_of_2(int((L0CSize // 4) ** 0.5))
        if min_mn < 128:
            # non-split case
            min_mn = self.align_up(min_mn)
            add_block_sizes_to_configs(min_mn)

        BLOCK_min_mn_start = min(self.align_down(min_mn), upper_limit)
        down_limit = self.align_down(math.ceil(min_mn / 8))
        for BLOCK_min_mn in self.aligned_decrease_iterator(
            BLOCK_min_mn_start, down_limit=down_limit
        ):
            add_block_sizes_to_configs(BLOCK_min_mn)

        if not isinstance(dispatch_policy, MmadMultiBatch):
            # Tile heuristics to decrease the number of tile configs
            configs = self.tile_heuristics(shape_desc, configs)

        # NB: for specific dispatch policy, some tile shapes are illegal,
        # so we need to filter them out
        configs = self.filter_configs(shape_desc, dispatch_policy, configs)
        return [_generate_tile_desc(*tile_cfg) for tile_cfg in configs]


class GemmAutotune:
    def __init__(self, arch_type):
        self.arch_type = arch_type
        self.tile_autotune = TileAutotune(arch_type)
        self.caches = {}

    def gen_configs(self, kernel, dispatch_policy, dtype, shape_desc):
        dtype_size = dtype.data_size()
        op_kind = type(kernel)

        key = (op_kind, dtype_size, dispatch_policy, shape_desc)
        if key in self.caches:
            return self.caches[key]

        tile_cfgs = self._get_tile_configs(
            kernel, dispatch_policy, dtype_size, shape_desc
        )

        self.caches[key] = tile_cfgs
        return tile_cfgs


    def set_l1_tile(self, tile, new_l1_tile):
        # 1st element of tile is l1_tile, 2nd element of tile is l0_tile
        tile[0].m = new_l1_tile[0]
        tile[0].n = new_l1_tile[1]
        tile[0].k = new_l1_tile[2]
        tile[1].m = new_l1_tile[0]
        tile[1].n = new_l1_tile[1]
        # the new l1k may be less than l0k
        if tile[1].k > tile[0].k:
            tile[1].k = tile[0].k

    
    def may_adjust_l1_tile_for_bias(self, dtype_size, tile):
        # default l1 stages & size
        l1_stages = 2
        l1_size = 1 << 19  # 512k
        if self.tile_autotune:
            l1_stages = self.tile_autotune.L1Stages
            l1_size = self.tile_autotune.L1Size

        l1_tile = tile[0]
        new_l1_tile = may_adjust_l1_tileshape(
            dtype_size, l1_tile.m, l1_tile.n, l1_tile.k, l1_stages=l1_stages, l1_size=l1_size
        )
        self.set_l1_tile(tile, new_l1_tile)

    def _get_tile_configs(self, kernel, dispatch_policy, dtype_size, shape_desc):
        if (
            self.tile_autotune is not None
            and shape_desc is not None
            and _is_support_tile_autotune(kernel)
        ):
            res = self.tile_autotune.gen_tile_configs(
                dispatch_policy, dtype_size, shape_desc
            )
        else:
            res = []
        # may adjust l1 tile if has bias
        if shape_desc[-1]:
            for tile_desc in res:
                self.may_adjust_l1_tile_for_bias(dtype_size, tile_desc)
        return res


def get_gemm_autotune(arch_type=None):
    if not hasattr(get_gemm_autotune, "instance"):
        if arch_type is None:
            arch_type = Arch.AtlasA2
        get_gemm_autotune.instance = GemmAutotune(arch_type)
    return get_gemm_autotune.instance


def generate_configs(arch_type, kernel):
    autotune = get_gemm_autotune(arch_type)
    dispatch_policy = kernel.get_dispatch_policy()[0] # currently using the first default policy, may expand in future.
    data_type = kernel.element_A
    shape_desc = (kernel.M, kernel.N, kernel.K, kernel.layout_Bias)
    return autotune.gen_configs(kernel, dispatch_policy, data_type, shape_desc)


# Fragile, we suppose this func is called after generate_configs
# since bias can be fused in scheduling, we cannot know it in advance
def may_adjust_l1_tile_for_bias(op, arch_type=None):
    from catlass_cppgen.common.data_type import DataType

    autotune = get_gemm_autotune(arch_type)
    element = op.element_Bias if op.element_Bias != DataType.UNDEFINED else op.element_A
    dtype_size = op.element_A.data_size()
    autotune.may_adjust_l1_tile_for_bias(dtype_size, [op.l1_tile_shape, op.l0_tile_shape])
