import copy
import functools
import math
import sys
from torch._inductor.runtime.runtime_utils import next_power_of_2
from torch._inductor.runtime.triton_heuristics import Config

from .triton_utils import byte_per_numel
from ..config import num_vector_core


# generate tiling configs
class TileGenerator:

    def __init__(self, numels, axis_names, tiling_axis, split_axis, low_dims, persistent_reduction,
                 configs, dtype, dual_reduction=False):
        self.numels = numels.copy()

        self.blocks = [x for x in self.numels]
        self.candidate_blocks = []
        self.sub_blocks = self.blocks.copy()
        self.axis_name = axis_names
        self.tiling_axis = tiling_axis
        self.split_axis = split_axis
        self.low_dims = low_dims
        self.configs = configs
        self.dtype_bytes = self.get_byte_per_numel(dtype)
        self.stop_numel = 1024 // self.dtype_bytes
        self.block_name = {}
        self.sub_block_name = {}
        self.persistent_reduction = persistent_reduction
        self.dual_reduction = dual_reduction
        for axis, name in enumerate(self.axis_name):
            if axis not in tiling_axis and axis not in split_axis:
                self.blocks[axis] = 1
                self.sub_blocks[axis] = 1
                continue
            if axis in self.split_axis:
                self.block_name[axis] = f"{name.upper()}BLOCK"
            if axis in self.tiling_axis:
                self.sub_block_name[axis] = f"{name.upper()}BLOCK_SUB"

    @classmethod
    def aligned_numel(cls, numel):
        aligned = next_power_of_2(numel)
        return aligned

    @classmethod
    def get_byte_per_numel(cls, dtype):
        if dtype is None:
            return 1
        return byte_per_numel[dtype]

    def valid_tile_numel(self, total_numel):
        byte_num = self.dtype_bytes
        max_numel = 16384 * 4 // byte_num
        return total_numel <= max_numel

    def calculate_config_numel(self, config):
        total_numel = 1
        for axis in self.tiling_axis:
            total_numel = total_numel * config[self.sub_block_name[axis]]
        return total_numel

    def calculate_total_numel(self):
        smallest = sys.maxsize

        def calculate_total_numel_candi(blocks):
            total_numel = 1
            for axis in self.tiling_axis:
                total_numel = total_numel * self.sub_blocks[axis]
            return total_numel

        for candi_blocks in self.candidate_blocks:
            numel = calculate_total_numel_candi(candi_blocks)
            if numel < smallest:
                smallest = numel
        return smallest

    def fill_config(self, config, blocks):
        for axis in self.split_axis:
            config[self.block_name[axis]] = blocks[axis]
        for axis in self.tiling_axis:
            tiling_numel = self.aligned_numel(self.sub_blocks[axis])
            config[self.sub_block_name[axis]] = tiling_numel

    def find_config(self, cfg):
        for config in self.configs:
            if config.kwargs == cfg:
                return True
        return False

    def add_to_configs(self, candi_block):
        newcfg = {}
        self.fill_config(newcfg, candi_block)
        total_numel = self.calculate_config_numel(newcfg)
        if self.valid_tile_numel(total_numel) and not self.find_config(newcfg):
            self.configs.append(Config(newcfg, num_warps=1, num_stages=1))

    def descend_one_axis(self, axis, is_split=False):
        def calc_total_programs():
            grids = []
            for axis in self.split_axis:
                numel = self.numels[axis]
                block_size = self.blocks[axis]
                programs = (numel + block_size - 1) // block_size
                grids.append(programs)

            total_programs = functools.reduce(lambda x, y: x * y, grids) if grids else 1
            return total_programs

        reached_stop_numel = False
        slow_decend_split = False

        while True:
            total_numel = self.stop_numel + 100
            for candi_block in self.candidate_blocks:
                self.add_to_configs(candi_block)

            # tile numel reached threshold     
            total_numel = self.calculate_total_numel()
            if total_numel <= self.stop_numel:
                self.add_to_configs(self.blocks)
                reached_stop_numel = True
                break

            numel = self.blocks[axis] if is_split else self.sub_blocks[axis]
            if numel == 1:
                self.add_to_configs(self.blocks)
                break

            if is_split:
                if self.persistent_reduction and self.axis_name[axis][0] == "r":
                    reached_stop_numel = True
                    break
                total_programs = calc_total_programs()
                if total_programs > num_vector_core:
                    break
                if total_programs > num_vector_core // 2 or self.dual_reduction:
                    if len(self.candidate_blocks) > 2:
                        self.candidate_blocks.pop(0)
                    self.candidate_blocks.append(tuple(self.blocks))

                self.blocks[axis] = numel // 2
                self.sub_blocks[axis] = self.blocks[axis]
                total_programs = calc_total_programs()
                if total_programs > num_vector_core:
                    slow_decend_split = True
                step = numel // 4 if numel // 4 > 1 else 1
                self.blocks[axis] = numel // 2 if not slow_decend_split else numel - step
                self.sub_blocks[axis] = self.blocks[axis]
            else:
                if numel >= 128:
                    self.sub_blocks[axis] = next_power_of_2(numel // 2)
                else:  # numel >4 and numel < 128 :
                    self.slow_descend_axis(axis)
        return reached_stop_numel

    def slow_descend_axis(self, axis):
        numel = self.sub_blocks[axis]
        self.sub_blocks[axis] = self.aligned_numel(numel // 2)

    def descend_all_low_dims(self):
        low_dim_numels = [self.sub_blocks[x] for x in self.low_dims]
        if not low_dim_numels:
            return

        def descent_all_axis(min_numel):
            for axis in self.low_dims:
                if self.axis_name[axis][0] == "r" and self.persistent_reduction:
                    continue
                numel = self.sub_blocks[axis]
                if numel == 1:
                    continue
                if min_numel > 1 and abs(numel - min_numel) / min_numel < 0.2:
                    continue
                if numel >= 128:
                    self.sub_blocks[axis] = next_power_of_2(numel // 2)
                else:  # numel >4 and numel < 128 :
                    self.slow_descend_axis(axis)

        count = 0
        total_numel = self.calculate_total_numel()
        while total_numel > self.stop_numel and count < 100:
            count += 1
            total_numel = self.calculate_total_numel()
            for candi_block in self.candidate_blocks:
                self.add_to_configs(candi_block)
            min_numel = min(low_dim_numels)
            descent_all_axis(min_numel)
            total_numel_2 = self.calculate_total_numel()
            if total_numel == total_numel_2:
                descent_all_axis(0)

        return total_numel < self.stop_numel

    def descend_split_tiling(self):

        tiling_not_low_dims = [x for x in self.tiling_axis if x not in self.low_dims]

        def descend_split_axis():

            for axis in self.split_axis:
                if self.descend_one_axis(axis, is_split=True):
                    return True

            total = self.calculate_total_numel()
            return total <= self.stop_numel

        def desceond_tiling_not_low_dims():
            for axis in tiling_not_low_dims:
                if self.axis_name[axis][0] == "r" and self.persistent_reduction:
                    continue
                if self.descend_one_axis(axis):
                    return True
            total = self.calculate_total_numel()
            return total <= self.stop_numel

        #  need to all low dims fairly
        def descend_low_dims():
            for axis in self.tiling_axis:
                if self.axis_name[axis][0] == "r" and self.persistent_reduction:
                    continue
                if axis in tiling_not_low_dims:
                    continue
                if self.descend_one_axis(axis):
                    return True
            total = self.calculate_total_numel()
            return total <= self.stop_numel

        while True:
            # descend split axis
            if descend_split_axis():
                break
            if len(self.candidate_blocks) > 0:
                self.sub_blocks = list(self.candidate_blocks[0])
            # descend tiling but not low dims
            if desceond_tiling_not_low_dims():
                break
                # descend low dims, need to descend all axis at the same time
            self.descend_all_low_dims()
            break
