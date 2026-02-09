import copy
import functools
import math
import os
import sys
from torch._inductor.runtime.runtime_utils import next_power_of_2
from torch._inductor.runtime.triton_heuristics import Config

from .triton_utils import get_byte_per_numel, NPUKernelType
from .. import config


# generate tiling configs
class TileGenerator:

    def __init__(self, numels, axis_names, tiling_axis, no_loop_axis, split_axis, low_dims, persistent_reduction,
                  dtype, npu_kernel_type=NPUKernelType.SIMD, input_ptr_num=0, dual_reduction=False):
        self.numels = numels.copy()

        self.blocks = [x for x in self.numels]
        self.candidate_blocks = []
        self.sub_blocks = self.blocks.copy()
        self.axis_name = axis_names
        self.tiling_axis = tiling_axis
        self.no_loop_axis = no_loop_axis
        self.split_axis = split_axis
        self.low_dims = low_dims
        self.configs = []
        self.dtype_bytes = get_byte_per_numel(dtype)
        self.block_name = {}
        self.sub_block_name = {}
        self.persistent_reduction = persistent_reduction
        self.dual_reduction = dual_reduction
        self.npu_kernel_type = npu_kernel_type
        self.max_total_numel = functools.reduce(lambda x, y: x * y, self.blocks) if self.blocks else 1
        self.tiny_kernel = self.max_total_numel < 128 * 1024
        self.input_ptr_num = 3 if input_ptr_num == 0 else min(input_ptr_num, 3)

        local_mem_size = 128 * 1024 if self.npu_kernel_type == NPUKernelType.SIMT_ONLY else config.ub_size
        self.max_numel_threshold = local_mem_size // self.input_ptr_num // self.dtype_bytes
        self.stop_numel = min(self.max_numel_threshold, self.max_total_numel // (config.num_vector_core * self.dtype_bytes)) // 8
        for axis, name in enumerate(self.axis_name):
            if axis not in tiling_axis and axis not in split_axis:
                self.blocks[axis] = 1
                self.sub_blocks[axis] = 1
                continue
            if axis in self.split_axis:
                self.block_name[axis] = f"{name.upper()}BLOCK"
            if axis in self.tiling_axis:
                self.sub_block_name[axis] = f"{name.upper()}BLOCK_SUB"

        self.program_threshold = config.num_vector_core // 8 if self.tiny_kernel and self.npu_kernel_type == NPUKernelType.SIMD else config.num_vector_core // 2
        self.program_threshold = 0 if self.max_total_numel < 128 else self.program_threshold
        self.npu_kernel_type = npu_kernel_type
        self.real_tiling_axis = []
        for tiling_axis in self.tiling_axis:
            if tiling_axis in self.no_loop_axis:
                continue
            if self.axis_name[tiling_axis][0] == "r" and self.persistent_reduction:
                continue
            self.real_tiling_axis.append(tiling_axis)   
        self.split_axis_num = len(self.split_axis)

    def reset_configs(self):
        self.config = []
        self.blocks = [x for x in self.numels]
        self.candidate_blocks = []
        self.sub_blocks = self.blocks.copy()
        for axis, _ in enumerate(self.axis_name):
            if axis not in self.tiling_axis and axis not in self.split_axis:
                self.blocks[axis] = 1
                self.sub_blocks[axis] = 1
        self.real_tiling_axis = []
        for tiling_axis in self.tiling_axis:
            if tiling_axis in self.no_loop_axis:
                continue
            if self.axis_name[tiling_axis][0] == "r" and self.persistent_reduction:
                continue
            self.real_tiling_axis.append(tiling_axis)   
        self.split_axis_num = len(self.split_axis)

    def calcu_last_split_blocks(self, axis):
        splits = 1
        for x in self.split_axis:
            if x != axis:
                splits = splits * ((self.numels[x] + self.blocks[x] - 1) // self.blocks[x])
            else:
                break

        last_splits = config.num_vector_core // splits
        last_blocks = (self.numels[axis] + last_splits - 1) // last_splits
        return last_blocks


    def aligned_numel(self, numel):
        min_numel = 32 // self.dtype_bytes
        if numel <= min_numel:
            return numel
        aligned = ((numel + min_numel - 1) // min_numel) * min_numel
        return aligned

    def valid_tile_numel(self, total_numel):
        max_numel = self.max_numel_threshold
        return total_numel <= max_numel

    def calculate_config_numel(self, cfg):
        total_numel = 1
        for i, _ in enumerate(self.sub_blocks):
            sub_block_name = self.sub_block_name.get(i, "")
            config_sub_block = cfg.get(sub_block_name, None)
            if config_sub_block is None:
                total_numel = total_numel * 1
            else:
                total_numel = total_numel * config_sub_block
        return total_numel

    def calculate_total_numel(self):
        smallest = sys.maxsize

        def calculate_total_numel_candi(blocks):
            total_numel = 1
            for axis in self.sub_blocks:
                total_numel = total_numel * axis
            return total_numel

        for candi_blocks in self.candidate_blocks:
            numel = calculate_total_numel_candi(candi_blocks)
            if numel < smallest:
                smallest = numel
        return smallest

    def fill_config(self, cfg, blocks):
        for axis in self.split_axis:
            cfg[self.block_name[axis]] = blocks[axis]
        for axis in self.tiling_axis:
            if self.npu_kernel_type == NPUKernelType.SIMT_ONLY:
                tiling_numel = next_power_of_2(self.sub_blocks[axis])
                while tiling_numel > blocks[axis]:
                    tiling_numel = tiling_numel // 2
            else:
                tiling_numel = min(self.aligned_numel(self.sub_blocks[axis]), blocks[axis])
            cfg[self.sub_block_name[axis]] = tiling_numel
        cfg["compile_mode"] = self.npu_kernel_type.compile_mode()
        cfg["remain_programs"] = self.cal_cfg_remain_programs(cfg)
        cfg["using_programs"] = self.calc_cfg_programs(cfg)

    def cal_cfg_remain_programs(self, cfg):
        remain_programs = 0
        for i, block in enumerate(self.blocks):
            block_name = self.block_name.get(i, "")
            config_block = cfg.get(block_name, None)
            if config_block is None:
                remain_programs += self.numels[i] % block
            else:
                remain_programs += self.numels[i] % config_block
        return remain_programs

    def calc_cfg_programs(self, cfg):
        grids = []
        for i, _ in enumerate(self.blocks):
            block_name = self.block_name.get(i, "")
            config_block = cfg.get(block_name, None)
            if config_block is None:
                grids.append(1)
            else:
                grids.append((self.numels[i] + config_block - 1) // config_block)
        total_programs = functools.reduce(lambda x, y: x * y, grids) if grids else 1
        return total_programs

    def find_config(self, cfg):
        for config_var in self.configs:
            if config_var.kwargs == cfg:
                return True
        return False

    def add_to_configs(self, candi_block):
        newcfg = {}
        self.fill_config(newcfg, candi_block)
        total_numel = self.calculate_config_numel(newcfg)
        stop_numel_threshold = 0 if len(self.configs) < 10 or self.tiny_kernel else self.stop_numel + 100
        if not self.valid_tile_numel(total_numel):
            return False
        if self.find_config(newcfg):
            return False
        if total_numel < stop_numel_threshold:
            return False

        # This is tmp check for simt overflow, and will be removed latter
        if self.npu_kernel_type == NPUKernelType.SIMT_ONLY and total_numel * self.dtype_bytes > 8 * 1024:
            return False
        self.configs.append(Config(newcfg, num_warps=1, num_stages=1))
        return True

    def desecnd_all_low_dims_with_all_blocks(self):
        restore_sub_blocks = {}
        for x in self.low_dims:
            restore_sub_blocks[x] = self.sub_blocks[x]
        self.descend_all_low_dims()
        for x in self.low_dims:
            self.sub_blocks[x] = restore_sub_blocks[x]

    def descend_tiling_axis(self, axis):
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

        while True:
            if axis in self.real_tiling_axis:
                do_tiling_axis_index = self.real_tiling_axis.index(axis)
                if do_tiling_axis_index + 1 < len(self.real_tiling_axis):
                    tmp_block = copy.deepcopy(self.blocks)
                    tmp_sub_block = copy.deepcopy(self.sub_blocks)
                    self.descend_tiling_axis(self.real_tiling_axis[do_tiling_axis_index + 1])
                    self.blocks = tmp_block
                    self.sub_blocks = tmp_sub_block

            for candi_block in self.candidate_blocks:
                if self.add_to_configs(candi_block):
                    self.desecnd_all_low_dims_with_all_blocks()

            if not self.candidate_blocks:
                if self.add_to_configs(self.blocks):
                    self.desecnd_all_low_dims_with_all_blocks()

            # tile numel reached threshold
            total_numel = self.calculate_total_numel()
            if total_numel <= self.stop_numel:
                if self.add_to_configs(self.blocks):
                    self.desecnd_all_low_dims_with_all_blocks()
                reached_stop_numel = True
                break

            numel = self.sub_blocks[axis]
            if numel == 1:
                if self.add_to_configs(self.blocks):
                    self.desecnd_all_low_dims_with_all_blocks()
                break

            if numel >= 32:
                self.sub_blocks[axis] = next_power_of_2(numel // 2)

            else:  # numel < 32 :
                numel = self.sub_blocks[axis]
                self.sub_blocks[axis] = numel - 1
        return reached_stop_numel

    def descend_split_axis_inner(self, axis_index):
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
        axis = self.split_axis[axis_index]
        while True:
            if axis_index + 1 < self.split_axis_num and not slow_decend_split:
                tmp_block = copy.deepcopy(self.blocks)
                tmp_sub_block = copy.deepcopy(self.sub_blocks)
                self.descend_split_axis_inner(axis_index + 1)
                self.blocks = tmp_block
                self.sub_blocks = tmp_sub_block

            for candi_block in self.candidate_blocks:
                if self.add_to_configs(candi_block):
                    self.desecnd_all_low_dims_with_all_blocks()

            # tile numel reached threshold
            total_numel = self.calculate_total_numel()
            if total_numel <= self.stop_numel:
                if self.add_to_configs(self.blocks):
                    self.desecnd_all_low_dims_with_all_blocks()
                reached_stop_numel = True
                break

            numel = self.blocks[axis]
            if numel == 1:
                if self.add_to_configs(self.blocks):
                    self.desecnd_all_low_dims_with_all_blocks()
                break

            if self.persistent_reduction and self.axis_name[axis][0] == "r":
                reached_stop_numel = True
                break
            total_programs = calc_total_programs()
            if total_programs > config.num_vector_core:
                last_blocks = self.calcu_last_split_blocks(axis)
                if last_blocks != self.blocks[axis]:
                    self.blocks[axis] = last_blocks
                if tuple(self.blocks) not in self.candidate_blocks:
                    self.candidate_blocks.append(tuple(self.blocks))
                    self.add_to_configs(self.candidate_blocks[-1])
                break

            if total_programs > self.program_threshold or self.dual_reduction:
                if len(self.candidate_blocks) > 2:
                    self.candidate_blocks.pop(0)
                if tuple(self.blocks) not in self.candidate_blocks:
                    self.candidate_blocks.append(tuple(self.blocks))
                if self.tiny_kernel:
                    self.add_to_configs(list(tuple(self.blocks)))
                slow_decend_split = (total_programs > config.num_vector_core // 2)

            if not slow_decend_split or self.npu_kernel_type == NPUKernelType.SIMT_ONLY:
                self.blocks[axis] = numel // 2
                self.sub_blocks[axis] = self.blocks[axis]

            else:
                step = numel // 4 if numel // 4 > 1 else 1
                self.blocks[axis] = numel - step
                self.sub_blocks[axis] = self.blocks[axis]

            total_programs = calc_total_programs()
            if self.blocks[axis] == 1 and (total_programs > self.program_threshold or self.dual_reduction) and tuple(
                    self.blocks) not in self.candidate_blocks:
                if total_programs > config.num_vector_core:
                    last_blocks = self.calcu_last_split_blocks(axis)
                    if last_blocks != self.blocks[axis]:
                        self.blocks[axis] = last_blocks
                self.candidate_blocks.append(tuple(self.blocks))
                self.add_to_configs(self.candidate_blocks[-1])
                break

        return reached_stop_numel


    def descend_all_low_dims(self):
        low_dim_numels = [self.sub_blocks[x] for x in self.low_dims]
        if not low_dim_numels:
            return

        def descent_all_axis(min_numel):
            for axis in self.low_dims:
                if self.axis_name[axis][0] == "r" and self.persistent_reduction:
                    continue
                if axis in self.no_loop_axis:
                    continue

                numel = self.sub_blocks[axis]
                if numel == 1:
                    continue
                if min_numel > 1 and abs(numel - min_numel) / min_numel < 0.2:
                    continue
                if numel >= 128:
                    self.sub_blocks[axis] = next_power_of_2(numel // 2)
                else:  # numel >4 and numel < 128 :
                    numel = self.sub_blocks[axis]
                    numel = numel // 2
                    self.sub_blocks[axis] = min(self.aligned_numel(numel), next_power_of_2(numel))

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

    def tune_multibuffer(self):
        new_cfg = []
        for self_config in self.configs:
            self_config.kwargs['multibuffer'] = False
            config_copied = copy.deepcopy(self_config)
            config_copied.kwargs['multibuffer'] = True
            new_cfg.append(config_copied)
        self.configs.extend(new_cfg)

    def tune_simt_num_warps(self, tune_num_warps_list=None):
        num_warps_list = num_warps_list if tune_num_warps_list else [8, 16, 32, 64]
        configs_without_num_warps = copy.deepcopy(self.configs)
        self.configs = []
        for self_config in configs_without_num_warps:
            for cfg_num_warps in num_warps_list:
                new_cfg = copy.deepcopy(self_config)
                new_cfg.num_warps = cfg_num_warps
                self.configs.append(new_cfg)

    def add_extra_options(self):
        if self.npu_kernel_type != NPUKernelType.SIMT_ONLY:
            self.tune_multibuffer()
        if self.npu_kernel_type == NPUKernelType.SIMT_ONLY:
            self.tune_simt_num_warps()
    
    def set_kernel_type(self, npu_kernel_type):
        self.npu_kernel_type = npu_kernel_type

    def descend_split_tiling(self):
        self.reset_configs()
        tiling_not_low_dims = [x for x in self.tiling_axis if x not in self.low_dims]

        def descend_split_axis():

            for axis_index in range(len(self.split_axis)):
                if self.descend_split_axis_inner(axis_index):
                    return True

            total = self.calculate_total_numel()
            return total <= self.stop_numel

        def desceond_tiling_not_low_dims():
            for axis in tiling_not_low_dims:
                if self.axis_name[axis][0] == "r" and self.persistent_reduction:
                    continue
                if axis in self.no_loop_axis:
                    continue

                if self.descend_tiling_axis(axis):
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
        self.add_extra_options()
        if self.npu_kernel_type == NPUKernelType.SIMT_ONLY:
            self.configs.sort(key=lambda x: x.kwargs['remain_programs'], reverse=False)
            if self.configs[0].kwargs['remain_programs'] == 0:
                split_index = len(self.configs)
                for i, conf in enumerate(self.configs):
                    if conf.kwargs['remain_programs'] != 0:
                        split_index = i
                        break
                self.configs = self.configs[:split_index]
        return self.configs
