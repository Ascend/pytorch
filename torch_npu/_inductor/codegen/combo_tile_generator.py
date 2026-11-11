import math

from .tile_generator import TileGenerator
from .triton_utils import NPUKernelType, get_byte_per_numel
from .. import config


class ComboTileGenerator(TileGenerator):
    """Search shared tiles, counting each subkernel's boundary independently.

    TileGenerator supplies the split descent, subblock enumeration, alignment,
    duplicate removal and compiler options. Only the shape envelope, program
    counting and per-branch resident tile footprint differ for a combo launch.
    """

    def __init__(self, subkernels, input_ptr_num=0, npu_kernel_type=NPUKernelType.SIMD):
        self.subkernels = subkernels
        self.max_waves = config.combo_kernel_max_waves
        self._program_limit = config.num_vector_core * self.max_waves
        first = subkernels[0]
        axis_names = first["axis_names"]
        split_axis = first["split_axis"]
        if len(split_axis) > 1:
            raise ValueError("Combo tile search supports at most one split axis")

        numels = []
        for axis, name in enumerate(axis_names):
            extents = [item["size_hints"].get(name, 1) for item in subkernels]
            numels.append(sum(extents) if axis in split_axis else max(extents))
        dtype = max(
            (item["split_axis_dtype"] for item in subkernels),
            key=get_byte_per_numel,
        )
        super().__init__(
            numels, axis_names, first["tiling_axis"], first["no_loop_axis"],
            split_axis, first["low_dims"], first["persistent_reduction"], dtype,
            npu_kernel_type=npu_kernel_type, input_ptr_num=input_ptr_num,
            dual_reduction=any(item["dual_reduction"] for item in subkernels),
        )
        self.max_total_numel = sum(
            math.prod(item["size_hints"].get(name, 1) for name in axis_names)
            for item in subkernels
        )
        self.tiny_kernel = self.max_total_numel < 128 * 1024
        self.stop_numel = min(
            self.max_numel_threshold, self.max_total_numel // config.num_vector_core,
        ) // 8

    def rebuild_dynamic_axis(self):
        # Optional pins derived from an aggregate shape need not be valid for
        # every branch. Keep the no-loop decisions already made by codegen.
        return

    def calculate_total_programs(self, blocks):
        programs = 0
        for item in self.subkernels:
            if not item["split_axis"]:
                programs += 1
                continue
            axis = item["split_axis"][0]
            extent = item["size_hints"][self.axis_name[axis]]
            programs += (extent + blocks[axis] - 1) // blocks[axis]
        return programs

    def calcu_last_split_blocks(self, axis):
        """Find the smallest shared block fitting the wave budget, including tails.

        ceil(sum(extents) / max_programs) is only a lower bound: every subkernel has
        its own final partial block. Program count is monotone in block size,
        so a binary search finds the exact boundary without enumerating sizes.
        """
        lower = max(1, (self.numels[axis] + self.max_programs - 1)
                    // self.max_programs)
        upper = max(item["size_hints"][self.axis_name[axis]]
                    for item in self.subkernels)
        blocks = self.blocks.copy()
        blocks[axis] = upper
        if self.calculate_total_programs(blocks) > self.max_programs:
            raise ValueError("Combo has more nonempty subkernels than the program limit")
        while lower < upper:
            middle = (lower + upper) // 2
            blocks[axis] = middle
            if self.calculate_total_programs(blocks) > self.max_programs:
                lower = middle + 1
            else:
                upper = middle
        return lower

    def calculate_config_numel(self, cfg):
        largest_tile = 1
        for item in self.subkernels:
            tile_numel = 1
            for axis in item["tiling_axis"]:
                name = self.axis_name[axis]
                if axis in item["no_loop_axis"] or (
                    name.startswith("r") and item["persistent_reduction"]
                ):
                    extent = item["size_hints"][name]
                    if self.npu_kernel_type == NPUKernelType.SIMT_ONLY:
                        extent = 1 << (extent - 1).bit_length()
                else:
                    extent = cfg.get(f"{name.upper()}BLOCK_SUB", 1)
                tile_numel *= extent
            # Compare each branch's resident bytes in the common dtype units
            # consumed by the inherited resource checks.
            branch_bytes = get_byte_per_numel(item["split_axis_dtype"])
            normalized_numel = (tile_numel * branch_bytes + self.dtype_bytes - 1) // self.dtype_bytes
            largest_tile = max(largest_tile, normalized_numel)
        return largest_tile

    def descend_split_axis_inner(self, axis_index):
        reached_stop = super().descend_split_axis_inner(axis_index)
        # A combo may be classified as non-tiny even when each branch is tiny.
        # In that case the inherited descent retains several split choices but
        # starts tiling from the final, smaller subblock. Also consider the full
        # block of each retained split choice before descending shared subtiles.
        saved_sub_blocks = self.sub_blocks.copy()
        for candidate in self.candidate_blocks:
            for axis in self.split_axis:
                self.sub_blocks[axis] = candidate[axis]
            self.add_to_configs(candidate)
        self.sub_blocks = saved_sub_blocks
        return reached_stop

    @property
    def max_programs(self):
        return self._program_limit

    def descend_split_tiling(self):
        # The parent keeps only the last few split seeds. Searching only at the
        # maximum budget drops the one-wave boundary and intermediate seeds.
        # Search every wave budget independently to retain all one-wave tiles,
        # including their subtiles and compiler options, before deduplicating.
        configs = []
        seen = set()
        minimum_programs = self.calculate_total_programs(self.numels)
        program_limit = self._program_limit
        try:
            for waves in range(1, self.max_waves + 1):
                self._program_limit = config.num_vector_core * waves
                if minimum_programs > self.max_programs:
                    continue
                for candidate in super().descend_split_tiling():
                    key = (tuple(sorted(candidate.kwargs.items())),
                           candidate.num_warps, candidate.num_stages)
                    if key not in seen:
                        seen.add(key)
                        configs.append(candidate)
        finally:
            self._program_limit = program_limit
        self.configs = configs
        return self.configs

    def add_to_configs(self, candi_block):
        if self.calculate_total_programs(candi_block) > self.max_programs:
            return False
        return super().add_to_configs(candi_block)
