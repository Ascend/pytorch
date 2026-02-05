import bisect
import copy
import math
import heapq
import os
import time
import logging
import re
import dataclasses
import functools
import itertools
import shutil
import hashlib
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed

import triton
import torch
import numpy as np
import pandas as pd
from torch._inductor.runtime.triton_heuristics import Config
from torch._inductor.runtime.runtime_utils import next_power_of_2
from torch._inductor import config

import torch_npu
from .codegen.tile_generator import TileGenerator
from .config import log
from .npu_triton_heuristics import NPUCachingAutotuner
from . import config as npu_config
from .codegen.triton_utils import get_byte_per_numel, NPUKernelType
from .profiler import simple_trace_handler


def fast_a_log_message(content, tag='autotuner', level='debug'):
    full_content = '[fastA {}] '.format(tag) + content
    if level == 'debug':
        log.debug(full_content)
    elif level == 'info':
        log.info(full_content)
    elif level == 'warning':
        log.warning(full_content)
    elif level == 'error':
        log.error(full_content)


def ceil_divide(a, b):
    return (a + b - 1) // b


class FastASetting:
    def __init__(self):
        self.fast_a_on = npu_config.fasta_autotune
        self.autotune_method = npu_config.fasta_autotune_method
        self.use_tile_exp = True
        self.fasta_method_lib = ["SampleStack", "Expert"]
        self.profiling_clear_L2 = False

        self.re_profiling = False
        self.re_profiling_max_times = 500

        # 超参数集合
        # for bucket
        self.bucket_max_config_num = 10  # for sample stack
        self.find_bucket_max_ub_max_times = float('inf')   # for sample stack

        # for sample stack
        self.tolerance_rate = 0.05  # both for model
        self.max_tolerance = float('inf')
        self.bucket_max_find_times = 2
        self.autotuner_max_find_times = 3

        # for expert
        self.expert_min_bucket_num = 5
        self.expert_find_ub_right_index = 3

        self.kernel_no_fastA = []

        if self.fast_a_on:
            fast_a_log_message(content='fastA autotuner with tile_gen online, set ENABLE_PRINT_UB_BITS=1',
                               tag='setting', level='info')
            if self.use_tile_exp:
                fast_a_log_message(content='use tiling generation expert: {}'.format(self.use_tile_exp), tag='setting')
            elif self.kernel_no_fastA:
                self.use_tile_exp = True
                fast_a_log_message(content='use tiling generation expert on because of kernel_no_fastA',
                                   tag='setting', level='info')
                fast_a_log_message(content=f"fastA ignores kernel: {self.kernel_no_fastA}", tag='setting', level='info')

            fast_a_log_message(content="profiling clear L2:{}".format(self.profiling_clear_L2),
                               tag='setting')
            if self.re_profiling:
                fast_a_log_message(content='fastA re_profiling: {}, max times: {}'.format(
                    self.re_profiling, self.re_profiling_max_times), tag='setting')

            if self.autotune_method not in self.fasta_method_lib:
                fast_a_log_message(content=f"no {self.autotune_method}, use SampleStack method instead",
                                   tag='setting', level='warning')
                self.autotune_method = "Expert"
            else:
                fast_a_log_message(content=f'fastA autotune method: {self.autotune_method}',
                                   tag="setting", level='info')

            if self.autotune_method == 'SampleStack':
                fast_a_log_message(content=f"for iterative method: bucket_max_config_num: {self.bucket_max_config_num}",
                                   tag='setting')
                fast_a_log_message(
                    content=f"for iterative method: find_bucket_max_ub_max_times: {self.find_bucket_max_ub_max_times}",
                    tag='setting')
                fast_a_log_message(content=f"Sample Stack: tolerance_rate: {self.tolerance_rate}", tag='setting')
                fast_a_log_message(content=f"Sample Stack: max_tolerance: {self.max_tolerance}", tag='setting')
                fast_a_log_message(content=f"Sample Stack: bucket_max_find_times: {self.bucket_max_find_times}",
                                   tag='setting')
                fast_a_log_message(content=f"Sample Stack: max_find_times: {self.autotuner_max_find_times}",
                                   tag='setting')

            if self.autotune_method == 'Expert':
                fast_a_log_message(
                    content='Expert: the minimus number of kernel buckets selected by expert method is {}'.format(
                        self.expert_min_bucket_num), tag='setting')
                fast_a_log_message(
                    content='Expert: the second ub search will find {} right index'.format(
                        self.expert_find_ub_right_index), tag='setting')
        else:
            fast_a_log_message(content='fastA autotune off', tag='setting', level='info')

    def is_in_kernel_black_list(self, kernel_name):
        for name in self.kernel_no_fastA:
            if name in kernel_name:
                return True
        return False


FASTA_SETTING = FastASetting()


def get_ub_size():
    if "910b" in npu_config.target.arch.lower():
        return 192
    else:
        return 256



class FastAConfig(Config):
    def __init__(self, kwargs, num_warps=4, num_stages=2, num_ctas=1, num_buffers_warp_spec=0, num_consumer_groups=0,
                 reg_dec_producer=0, reg_inc_consumer=0, maxnreg=None, pre_hook=None):
        super().__init__(kwargs, num_warps, num_stages, num_ctas, num_buffers_warp_spec, num_consumer_groups,
                         reg_dec_producer, reg_inc_consumer, maxnreg, pre_hook)
        self.num_vector_core = None
        self.ub_size = None
        self.ub_usage = None
        self.vector_core_num = None
        self.core_usage = None
        self.from_expert = False  # from baseline
        self.real_ub_size = -1
        self.circle_num = -1

    def add_core_ub_info(self, num_vector_core, ub_size, ub_usage, core_num):
        self.num_vector_core = num_vector_core
        self.ub_size = ub_size
        self.ub_usage = ub_usage
        self.vector_core_num = core_num
        self.core_usage = self.vector_core_num / self.num_vector_core

    def get_config_info(self):
        return "config: {}, core num {}, UB usage ratio {:.5f}, from expert {}".format(
            self.kwargs, self.vector_core_num, self.ub_usage, self.from_expert)


class TileConfig:
    def __init__(self):
        self.blocks = []
        self.sub_blocks = []
        self.vector_core_num = 0
        self.ub = 0

    def cal_vector_core_num(self, origin_numels, split_axis_set):
        grids = []
        for axis in split_axis_set:
            numel = origin_numels[axis]
            block_size = self.blocks[axis]
            programs = ceil_divide(numel, block_size)
            grids.append(programs)
        self.vector_core_num = functools.reduce(lambda x, y: x * y, grids) if grids else 1

    def cal_ub_size(self, tiling_axis_set):
        total_numel = 1
        for axis in tiling_axis_set:
            total_numel = total_numel * self.sub_blocks[axis]
        self.ub = total_numel

    def cal_circle_num(self):
        now_circle = 1
        for block_numel, sub_block_numel in zip(self.blocks, self.sub_blocks):
            now_circle *= ceil_divide(block_numel, sub_block_numel)
        return now_circle

    def generate_base_config(self, block_name, sub_block_name, split_axis_set, tiling_axis_set, deal_function=None):
        cfg = {}

        def default_deal_function(x):
            return x

        if deal_function is None:
            deal_function = default_deal_function
        for axis in split_axis_set:
            cfg[block_name[axis]] = self.blocks[axis]
        for axis in tiling_axis_set:
            cfg[sub_block_name[axis]] = deal_function(self.sub_blocks[axis])
        return cfg


class FastATileGenerator(TileGenerator):
    def __init__(self, numels, axis_names, tiling_axis, no_loop_axis, split_axis, low_dims, persistent_reduction,
                 dtype, npu_kernel_type=NPUKernelType.SIMD, input_ptr_num=0, dual_reduction=False):
        super().__init__(numels, axis_names, tiling_axis, no_loop_axis, split_axis, low_dims, persistent_reduction,
                 dtype, npu_kernel_type, input_ptr_num, dual_reduction)

        self.expert_configs = None
        self.aligned_numel_num = None
        self.num_vector_core = None
        self.max_ub_size_numel = None
        self.configs = []

    def add_multibuffer(self):
        """
        This function does not work within tilegen;
        its purpose is to reduce the number of configs and the compilation time of subsequent steps.
        Returns:
        """
        return

    def calc_total_programs_by_config(self, this_config):
        grids = []
        for axis in self.split_axis:
            numel = self.numels[axis]
            block_size = this_config[self.block_name[axis]]
            programs = (numel + block_size - 1) // block_size
            grids.append(programs)
        total_programs = functools.reduce(lambda x, y: x * y, grids) if grids else 1
        return total_programs

    def old_descend_split_tiling(self):
        super().descend_split_tiling()
        return

    def get_block_and_sub_list(self, start_core_num, stop_core_num, core_num_step,
                               start_ub, stop_ub, sub_block_step):
        """
        generate block and sub block list with parameters
        start_core_num(int): for block, start means a bigger value
        stop_core_num(int): for block, stop means a smaller value
        core_num_step(int): for block, the step
        start_ub(float): for sub_block, start means a bigger value
        stop_ub(float): for sub_block, stop means a smaller value
        sub_block_step_times(int): the aligned step times for sub_block
        """

        block_and_sub_list_in_function = []
        for axis, _ in enumerate(self.axis_name):
            final_list = []
            if not self.is_axis_in_tiling_or_split(axis):
                final_list.append([1, 1])
                block_and_sub_list_in_function.append(copy.deepcopy(final_list))
                continue
            elif axis in self.split_axis:
                if self.persistent_reduction and self.axis_name[axis][0] == "r":
                    final_list = [[self.blocks[axis], self.blocks[axis]]]
                    block_and_sub_list_in_function.append(copy.deepcopy(final_list))
                    continue
                else:
                    block_list = []
                    for use_core_num in range(start_core_num, stop_core_num - core_num_step, -core_num_step):
                        block_list.append(math.ceil(self.numels[axis] / use_core_num))
                    block_set = set(block_list)
                    block_list = sorted(list(block_set), reverse=True)
            else:
                block_list = [self.blocks[axis]]

            if self.persistent_reduction and self.axis_name[axis][0] == "r":
                for b in block_list:
                    final_list.append([b, b])

            elif axis in self.tiling_axis and axis not in self.no_loop_axis:
                for b in block_list:
                    start_sub_block = min(self.aligned_numel(b), start_ub)
                    if b > 32:

                        stop_sub_block = stop_ub

                        sub_block_list = list(
                            range(start_sub_block, stop_sub_block - sub_block_step, -sub_block_step))
                        if self.persistent_reduction:
                            add_list = list(range(stop_sub_block - 1, 0, -1))
                            sub_block_list += add_list
                    else:
                        stop_sub_block = 1
                        sub_block_list = list(range(start_sub_block, stop_sub_block - 1, -1))

                    for sb in sub_block_list:
                        final_list.append([b, sb])
            else:
                for b in block_list:
                    final_list.append([b, b])
            block_and_sub_list_in_function.append(copy.deepcopy(final_list))
        return block_and_sub_list_in_function

    def strict_aligned_numel(self, numel):
        min_numel = 32 // self.dtype_bytes
        if numel <= min_numel:
            return next_power_of_2(numel)
        aligned = ((numel + min_numel - 1) // min_numel) * min_numel
        return aligned

    def cal_sub_blocks_with_circle_num(self, block_size, this_stop_ub, aligned=False):
        this_sub_block_set = set()
        last_sub_block = block_size + 2 * self.aligned_numel_num
        for this_circle_num in range(1, block_size + 1):
            this_sub_block = ceil_divide(block_size, this_circle_num)
            if this_sub_block < this_stop_ub:
                break
            if (last_sub_block - this_sub_block >= self.aligned_numel_num) or this_sub_block < 32:
                if aligned:
                    this_sub_block_set.add(self.strict_aligned_numel(this_sub_block))
                else:
                    this_sub_block_set.add(this_sub_block)
                last_sub_block = this_sub_block
        return sorted(list(this_sub_block_set), reverse=True)

    def get_block_and_sub_list_ceil_divide(self, start_core_num, stop_core_num, core_num_step, stop_ub):
        """
        generate block and sub block list with parameters
        start_core_num(int): for block, start means a bigger value
        stop_core_num(int): for block, stop means a smaller value
        core_num_step(int): for block, the step,
        stop_ub(float): for sub_block, stop means a smaller value
        """


        aligned_axis = []
        not_r_axis = []
        r_axis = []
        for axis, name in enumerate(self.axis_name):
            if self.persistent_reduction and name[0] == "r":
                r_axis.append(axis)
            else:
                not_r_axis.append(axis)
        if (len(r_axis) > 0 and self.numels[r_axis[0]] % self.aligned_numel_num == 0)\
                or len(not_r_axis) == 0:
            aligned_axis.append(r_axis[0])
        else:
            aligned_axis.append(max(not_r_axis))

        block_and_sub_list_in_function = []
        for axis, _ in enumerate(self.axis_name):
            final_list = []
            if (not self.is_axis_in_tiling_or_split(axis)) or (
                    self.persistent_reduction and self.axis_name[axis][0] == "r"):
                final_list = [[self.blocks[axis], self.blocks[axis]]]
                block_and_sub_list_in_function.append(copy.deepcopy(final_list))
                continue

            block_list = []
            if axis in self.split_axis:
                for use_core_num in range(start_core_num, stop_core_num - core_num_step, -core_num_step):
                    block_list.append(ceil_divide(self.numels[axis], use_core_num))
                block_set = set(block_list)
                block_list = sorted(list(block_set), reverse=True)
            else:
                block_list = [self.blocks[axis]]

            if axis in self.tiling_axis and axis not in self.no_loop_axis:
                for b in block_list:
                    aligned_flag = axis in aligned_axis
                    if b > 32:
                        if aligned_flag:
                            this_time_stop_ub = self.aligned_numel(stop_ub)
                        else:
                            this_time_stop_ub = stop_ub
                    else:
                        this_time_stop_ub = 1

                    sub_block_list = self.cal_sub_blocks_with_circle_num(block_size=b,
                                                                    this_stop_ub=this_time_stop_ub,
                                                                    aligned=aligned_flag)
                    for sb in sub_block_list:
                        final_list.append([b, sb])
            else:
                for b in block_list:
                    final_list.append([b, b])
            block_and_sub_list_in_function.append(copy.deepcopy(final_list))
        return block_and_sub_list_in_function

    def is_axis_in_tiling_or_split(self, axis: int):
        return axis in self.tiling_axis or axis in self.split_axis

    def descend_split_tiling(self):
        max_ub_size_byte = get_ub_size() * 1024  # A2:192kb A5:256kb
        self.max_ub_size_numel = max_ub_size_byte // self.dtype_bytes
        self.num_vector_core = npu_config.num_vector_core
        self.aligned_numel_num = 32 // self.dtype_bytes  # step size need to adjust

        origin_core_num_stop_times = self.num_vector_core
        min_stop_ub_ration = self.aligned_numel_num / self.max_ub_size_numel
        min_avg_core_config_num = 3

        if FASTA_SETTING.use_tile_exp:
            self.old_descend_split_tiling()
            self.expert_configs = copy.deepcopy(self.configs)
            self.configs.clear()
            for _, cfg in enumerate(self.expert_configs):
                temp_config = FastAConfig(cfg.kwargs, num_warps=1, num_stages=1)
                temp_config.from_expert = True
                temp_config.add_core_ub_info(num_vector_core=npu_config.num_vector_core,
                                             ub_size=self.calculate_config_numel(cfg.kwargs),
                                             ub_usage=self.calculate_config_numel(cfg.kwargs) / self.max_ub_size_numel,
                                             core_num=self.calc_total_programs_by_config(cfg.kwargs))
                self.configs.append(temp_config)

        # fasta do not support simt/simt_mix/simt_template
        if self.npu_kernel_type != NPUKernelType.SIMD:
            return self.configs

        if len(self.axis_name) == 0:
            return self.configs

        if FASTA_SETTING.autotune_method == "Expert":
            block_and_sub_list = self.get_block_and_sub_list_ceil_divide(start_core_num=self.num_vector_core,
                                                                    stop_core_num=1, core_num_step=1,
                                                                    stop_ub=1)

        elif not self.tiny_kernel:
            sub_block_step_set = self.aligned_numel_num
            block_and_sub_list = self.get_block_and_sub_list(start_core_num=self.num_vector_core,
                                                        stop_core_num=self.num_vector_core // origin_core_num_stop_times,
                                                        core_num_step=1,
                                                        start_ub=self.aligned_numel(int(self.max_ub_size_numel)),
                                                        stop_ub=self.aligned_numel(
                                                            int(self.max_ub_size_numel * min_stop_ub_ration)),
                                                        sub_block_step=sub_block_step_set)
            all_configs_num = 1
            for i in range(len(self.axis_name)):
                all_configs_num *= len(block_and_sub_list[i])
            if all_configs_num <= (
                    self.num_vector_core - self.num_vector_core // origin_core_num_stop_times) * min_avg_core_config_num:
                block_and_sub_list = self.get_block_and_sub_list(start_core_num=self.num_vector_core,
                                                            stop_core_num=1,
                                                            core_num_step=1,
                                                            start_ub=self.aligned_numel(int(self.max_ub_size_numel)),
                                                            stop_ub=1,
                                                            sub_block_step=1)
        else:
            block_and_sub_list = self.get_block_and_sub_list(start_core_num=self.num_vector_core,
                                                        stop_core_num=1,
                                                        core_num_step=1,
                                                        start_ub=self.aligned_numel(int(self.max_ub_size_numel)),
                                                        stop_ub=1,
                                                        sub_block_step=1)

        core_bucket = [[] for _ in range(self.num_vector_core)]

        all_configs = [list(item) for item in itertools.product(*block_and_sub_list)]
        configs_now = []
        for c_item in all_configs:
            this_config = TileConfig()
            for _, content in enumerate(c_item):
                this_config.blocks.append(content[0])
                this_config.sub_blocks.append(content[1])
            this_config.cal_vector_core_num(origin_numels=self.numels, split_axis_set=self.split_axis)
            this_config.cal_ub_size(tiling_axis_set=self.tiling_axis)

            if this_config.vector_core_num <= self.num_vector_core and this_config.ub <= self.max_ub_size_numel:
                configs_now.append(this_config)
                core_bucket[this_config.vector_core_num - 1].append(this_config.ub)  # debug info

        fast_a_log_message(content=f"num vector core is {self.num_vector_core}", tag='tile_gen')
        fast_a_log_message(content=f"numel info is {self.numels}", tag='tile_gen')
        fast_a_log_message(content=f"split axis is {self.split_axis}", tag='tile_gen')
        fast_a_log_message(content=f"tiling axis is {self.tiling_axis}", tag='tile_gen')
        fast_a_log_message(content=f"low dim axis is {self.low_dims}", tag='tile_gen')

        use_deal = None
        for _, cfg in enumerate(configs_now):
            base_cfg = cfg.generate_base_config(block_name=self.block_name,
                                                sub_block_name=self.sub_block_name,
                                                split_axis_set=self.split_axis,
                                                tiling_axis_set=self.tiling_axis,
                                                deal_function=use_deal)
            temp_config = FastAConfig(base_cfg, num_warps=1, num_stages=1)
            temp_config.from_expert = False
            temp_config.add_core_ub_info(num_vector_core=npu_config.num_vector_core,
                                         ub_size=cfg.ub,
                                         ub_usage=cfg.ub / self.max_ub_size_numel,
                                         core_num=cfg.vector_core_num)
            temp_config.circle_num = cfg.cal_circle_num()
            self.configs.append(temp_config)
        return self.configs


class NPUFastABucket:
    def __init__(self, core_num, config_list, ub_max_valid_usage=1., config_list_length=10, triton_meta=None,
                 inductor_meta=None, device_props=None, fn=None):
        self.core_num = core_num
        self.config_list = config_list
        self.state = "start"  # start/running/end
        self.profile_value_list = []
        self.best_config_profile_value = None
        self.need_profile_configs = []
        sorted(self.config_list, key=lambda x: x.ub_usage, reverse=False)
        self.method_name = FASTA_SETTING.autotune_method

        self.triton_meta = triton_meta
        self.inductor_meta = inductor_meta
        self.device_props = device_props
        self.fn = fn

        self.ub_max_valid_usage = ub_max_valid_usage
        self.config_list_length = config_list_length
        self.start_deal_end = False
        if not self.config_list:
            self.valid_data_num = 0
            self.state = "end"
            self.have_tested = None
            return

        elif self.config_list[-1].ub_usage <= ub_max_valid_usage:
            self.valid_data_num = len(self.config_list)

        else:
            self.valid_data_num = bisect.bisect_right(
                self.config_list, self.ub_max_valid_usage, key=lambda x: x.ub_usage)
            if self.valid_data_num <= 0:
                self.state = "end"
                self.have_tested = None
                return

        self.have_tested = [False for _ in range(self.valid_data_num)]
        self.target_ub_point = [x * self.ub_max_valid_usage for x in [0.5, 0.6, 0.8, 0.9, 1.]]
        self.start_search_interval = True
        self.info_stack = []
        self.find_times = 0

        #  超参数集合
        self.max_find_times = FASTA_SETTING.bucket_max_find_times
        self.tolerance_rate = FASTA_SETTING.tolerance_rate
        self.max_tolerance = FASTA_SETTING.max_tolerance
        fast_a_log_message(content="bucket {}: method: {}, config num: {}, max_ub_usage: {:.4}".format(
            self.core_num, self.method_name, len(self.config_list), self.ub_max_valid_usage))

    def dynamic_generate_config(self, ref_ub_usage=None):
        if self.state == "end":
            return []

        self.find_times += 1
        if self.find_times > self.max_find_times:
            self.state = "end"
            return []

        if self.state == "start":
            self.start(ref_ub_usage=ref_ub_usage)

        if self.state == "running":
            self.find_best()

        return self.need_profile_configs

    def get_configs_in_bucket_expert(self, get_circle_num=1):
        this_config_list = sorted(self.config_list, key=lambda x: x.circle_num, reverse=False)
        expert_config_list = []
        circle_num_set = set()
        for cfg in this_config_list:
            if not cfg.from_expert:
                circle_num_set.add(cfg.circle_num)
                if len(circle_num_set) > get_circle_num:
                    break
                expert_config_list.append(cfg)
        return expert_config_list

    def start(self, ref_ub_usage=None):
        if self.valid_data_num <= self.config_list_length:
            self.need_profile_configs = copy.deepcopy(self.config_list[:self.valid_data_num])
            self.start_deal_end = True

        elif not ref_ub_usage:
            if self.start_search_interval:
                point_index = []
                for _, rp in enumerate(self.target_ub_point):
                    this_index = bisect.bisect_left(self.config_list[:self.valid_data_num],
                                                    rp, key=lambda x: x.ub_usage)
                    point_index.append(this_index)

                interval_num = [self.config_list_length // len(point_index) - 1] * (len(point_index) - 1)
                interval_num[-1] += self.config_list_length - sum(interval_num)
                for i in range(len(point_index) - 1):
                    choose_indexes_here, remain_num = self._get_equal_spaced_points(left=point_index[i],
                                                                                    right=point_index[i + 1],
                                                                                    n=interval_num[i])
                    for pi in choose_indexes_here:
                        if pi > self.valid_data_num - 1:
                            continue
                        self.need_profile_configs.append(self.config_list[pi])
                        self.have_tested[pi] = True
                    interval_num[-1] += remain_num
                if len(self.need_profile_configs) < self.config_list_length:
                    choose_indexes_here, _ = self._get_equal_spaced_points(left=0, right=point_index[0],
                                                                           n=self.config_list_length - len(
                                                                               self.need_profile_configs))
                    for pi in choose_indexes_here:
                        self.need_profile_configs.append(self.config_list[pi])
                        self.have_tested[pi] = True
            else:
                self.need_profile_configs = []
                interval_index_count = self.valid_data_num / (self.config_list_length - 1)
                real_float_index = 0.
                this_index = round(real_float_index)
                for _ in range(self.config_list_length):
                    self.need_profile_configs.append(self.config_list[this_index])
                    self.have_tested[this_index] = True
                    real_float_index += interval_index_count
                    this_index = round(real_float_index)
                    this_index = min(this_index, self.valid_data_num - 1)

        else:
            self._get_next_request_list(target_ub_usage=ref_ub_usage, direction='mid')
        self.state = "running"
        return self.need_profile_configs

    def find_best(self):
        valid_data = self._deal_return_profiling_data()
        if len(valid_data) == 0:
            self.state = 'end'
            return []

        if not self.start_deal_end:
            self._deal_sample_data(valid_data=valid_data)
            self.start_deal_end = True
        else:
            self._deal_data_use_interval(valid_data=valid_data)

        while len(self.info_stack) > 0:
            this_time, this_data = heapq.heappop(self.info_stack)
            if this_time <= self._get_tolerable_degradation_value(self.best_config_profile_value.profiler_time):
                deal_state = self._get_next_request_list(target_ub_usage=this_data[0], direction=this_data[3])
                if deal_state == "running":
                    return self.need_profile_configs

        self.state = 'end'
        return []

    def _deal_return_profiling_data(self):
        valid_data = []
        for pv in self.profile_value_list:
            if pv.config is not None and pv.profiler_time is not None:
                valid_data.append([pv.config.ub_usage, pv.profiler_time, pv.profiler_time_sem])

        this_valid_num = len(valid_data)
        if this_valid_num < 3:
            self.state = 'end'
            return []
        return valid_data

    def _deal_sample_data(self, valid_data):
        valid_data.sort(key=lambda x: x[1])
        up_limit = self._get_tolerable_degradation_value(valid_data[0][1])
        for _, this_data in enumerate(valid_data):
            if this_data[1] <= up_limit:
                this_data.append('mid')
                heapq.heappush(self.info_stack, (this_data[1], this_data))

    def _deal_data_use_interval(self, valid_data):
        this_valid_num = len(valid_data)
        ub_usages, times, stds = zip(*valid_data)
        interval_ratio = [0.2, 0.8, 1.]
        interval_flag = [max(1, int(this_valid_num * interval_ratio[0])),
                         max(2, min(this_valid_num - 1, int(this_valid_num * interval_ratio[1]))),
                         this_valid_num]
        intervals_min_index = [np.argmin(times[0:interval_flag[0]]),
                               np.argmin(times[interval_flag[0]:interval_flag[1]]),
                               np.argmin(times[interval_flag[1]:this_valid_num])]
        intervals_min_value = [times[intervals_min_index[0]],
                               times[intervals_min_index[1]],
                               times[intervals_min_index[2]]]
        if intervals_min_value[0] <= self._get_tolerable_degradation_value(
                self.best_config_profile_value.profiler_time):
            heapq.heappush(self.info_stack, (intervals_min_value[0], [ub_usages[intervals_min_index[0]],
                                                                      times[intervals_min_index[0]],
                                                                      stds[intervals_min_index[0]],
                                                                      'left']))
        if intervals_min_value[2] <= self._get_tolerable_degradation_value(
                self.best_config_profile_value.profiler_time):
            heapq.heappush(self.info_stack, (intervals_min_value[2], [ub_usages[intervals_min_index[2]],
                                                                      times[intervals_min_index[2]],
                                                                      stds[intervals_min_index[2]],
                                                                      'right']))

    def _get_tolerable_degradation_value(self, the_better_time):
        level = min(self.max_tolerance, the_better_time * self.tolerance_rate)
        return the_better_time + level

    @staticmethod
    def _get_equal_spaced_points(left: int, right: int, n: int):
        if n <= 0:
            return None, 0
        if n == 1:
            return [left], n - 1
        if right - left < n:
            return list(range(left, right)), n - (right - left)

        step = (right - 1 - left) // (n - 1)
        points = [left + i * step for i in range(n)]
        return points, 0

    def _get_next_request_list(self, target_ub_usage, direction):
        if direction not in ['left', 'mid', 'right']:
            direction = 'mid'
        self.need_profile_configs = []
        choose_indexes = []
        if direction == 'left':
            target_index = bisect.bisect_left(self.config_list, target_ub_usage, key=lambda x: x.ub_usage)
            this_index = target_index
            for _ in range(self.config_list_length):
                if this_index < 0:
                    break
                choose_indexes.append(this_index)
                this_index -= 1
        elif direction == 'right':
            target_index = bisect.bisect_right(self.config_list, target_ub_usage, key=lambda x: x.ub_usage)
            this_index = target_index
            for _ in range(self.config_list_length):
                if this_index >= self.valid_data_num:
                    break
                choose_indexes.append(this_index)
                this_index += 1
        else:
            target_index = bisect.bisect_left(self.config_list, target_ub_usage, key=lambda x: x.ub_usage)
            target_index = min(max(target_index, 0), len(self.config_list) - 1)
            left_index = target_index - 1
            right_index = target_index + 1
            choose_indexes = [target_index]
            while len(choose_indexes) <= self.config_list_length:
                in_new_data = False
                if right_index < self.valid_data_num:
                    choose_indexes.append(right_index)
                    right_index += 1
                    in_new_data = True
                if left_index > -1:
                    choose_indexes.append(left_index)
                    left_index -= 1
                    in_new_data = True
                if not in_new_data:
                    break
            while len(choose_indexes) > self.config_list_length:
                del choose_indexes[-1]

        if len(choose_indexes) <= 1:
            self.need_profile_configs = []
            return 'end'

        have_test_num = 0
        choose_indexes.sort()
        self.need_profile_configs = []
        for this_index in choose_indexes:
            self.need_profile_configs.append(self.config_list[this_index])
            have_test_num += self.have_tested[this_index]
            self.have_tested[this_index] = True

        if have_test_num / len(self.need_profile_configs) > 0.5:
            self.need_profile_configs = []
            return 'end'
        return "running"

    def add_profile_value(self, pv):
        if isinstance(pv, list):
            self.profile_value_list.extend(pv)
        else:
            self.profile_value_list.append(pv)
        self.profile_value_list = sorted(self.profile_value_list, key=lambda x: x.config.ub_usage)
        self.update_best_config(pv)

    def update_best_config(self, profile_values):
        if self.best_config_profile_value is None:
            self.best_config_profile_value = profile_values[0]

        for pv in profile_values:
            if self.best_config_profile_value.profiler_time > pv.profiler_time:
                self.best_config_profile_value = pv


class BinaryBucket:
    def __init__(self, core_num, config_list):
        self.core_num = core_num
        self.config_list = config_list
        self.left_index = 0
        self.right_index = len(config_list) - 1
        self.mid_index = (self.left_index + self.right_index) // 2
        self.result_index = None

    def update_right_index(self, new_right_index):
        self.right_index = new_right_index
        self.mid_index = (self.left_index + self.right_index) // 2

    def update_left_index(self, new_left_index):
        self.left_index = new_left_index
        self.mid_index = (self.left_index + self.right_index) // 2

    def get_now_mid_config(self):
        return self.config_list[self.mid_index]

    def get_mid_config(self, return_result):
        # we want to find the last 1 in '11111100000' compile_result is not None
        if return_result:
            self.left_index = self.mid_index + 1
            self.result_index = self.mid_index
        else:
            self.right_index = self.mid_index - 1

        if self.left_index <= self.right_index:
            self.mid_index = (self.left_index + self.right_index) // 2
            return self.config_list[self.mid_index], False
        else:
            if self.result_index is None:
                self.result_index = max(0, self.mid_index - 1)
            return self.config_list[self.result_index], True


@dataclasses.dataclass(order=True)
class ProfileValue:
    launcher: None = dataclasses.field(default=None, compare=False)
    config: FastAConfig = dataclasses.field(default=None, compare=False)
    profiler_time: float = None
    profiler_time_sem: float = None
    profiler_time_std: float = None


class NPUFastAutotuner(NPUCachingAutotuner):
    def __init__(
            self,
            fn,
            triton_meta,  # passed directly to triton
            configs,
            save_cache_hook,
            mutated_arg_names,  # see [Note: clone mutated buffers]
            optimize_mem,
            heuristic_type,
            size_hints=None,
            inductor_meta=None,  # metadata not relevant to triton
            custom_kernel=False,  # whether the kernel is inductor-generated or custom
            filename=None,
            reset_to_zero_arg_names=None,
    ):
        super().__init__(fn, triton_meta, configs, save_cache_hook, mutated_arg_names, optimize_mem, heuristic_type,
                         size_hints, inductor_meta, custom_kernel, filename, reset_to_zero_arg_names)
        self.compile_results = None
        self._reload_kernel = None
        self.origin_configs = []
        self.expert_configs = []
        self.use_origin_autotuner = False
        self.autotune_start_time = time.perf_counter()
        if len(configs) == 1:
            self.use_origin_autotuner = True
            return

        self.skip_precompile = True
        self.launchers = []
        self.num_vector_core = npu_config.num_vector_core
        self.use_tile_experience = True
        self.expert_configs_profiling = self.use_tile_experience
        if self.use_tile_experience:
            self._config_separation(configs=configs)
        else:
            self.origin_configs = configs

        self.bucket_dict = {}
        self.max_ub_usage_each_core = {}
        self.precompiled_thread_num = npu_config.max_precompiled_thread_num
        self.ub_max_valid_usage = 1.
        self.ub_all_ok = False

        self.best_profiling_value = None
        self.record_core_best_value = None
        self.best_value_from_core = -1
        self.best_launcher = None

        # debug
        self.profiling_config_num = 0
        self.get_result_config_num = 0

        self.bucket_score = []  # [i,0] is core_num, [i, 1] is score
        self.init_best_core_ratio = 0.9
        self.check_best_bucket_neighbor_num = 2
        self.best_profiling_ub_usage = None
        self.find_best_times = 0
        self.max_find_best_times = FASTA_SETTING.autotuner_max_find_times
        self.dtype = self.inductor_meta["split_axis_dtype"]

        self.not_r_axis_num = len(set(self.inductor_meta["split_axis"]) | set(self.inductor_meta["tiling_axis"]))
        self.min_core_num_with_no_filter_ub = npu_config.num_vector_core + 1
        self.expert_method_choose_core_num_list = []

        fast_a_log_message(content='kernel name here is {}'.format(self.get_fn_name()), tag='autotuner')
        fast_a_log_message(content='total configs num is {}'.format(len(self.origin_configs)), tag='autotuner')
        fast_a_log_message(content='expert configs num is {}'.format(len(self.expert_configs)), tag='autotuner')

        if len(self.origin_configs) <= 1 or len(self.configs) == 1:
            self.use_origin_autotuner = True
        if FASTA_SETTING.is_in_kernel_black_list(self.get_fn_name()):
            self.use_origin_autotuner = True
        #  fasta is not support simt/simt_mix autotune
        if NPUKernelType(inductor_meta.get("npu_kernel_type", "simd")) != NPUKernelType.SIMD:
            self.use_origin_autotuner = True

        if self.use_origin_autotuner:
            self.configs = self.expert_configs
            return

        if FASTA_SETTING.autotune_method == "Expert":
            self.skip_precompile = False
            self._precompile_for_expert()

    def _config_separation(self, configs):
        self.expert_configs = []
        if FASTA_SETTING.autotune_method != "Expert":
            for _, cfg in enumerate(configs):
                if cfg.from_expert:
                    self.expert_configs.append(cfg)
            self.origin_configs = configs
        else:
            self.origin_configs = []
            for _, cfg in enumerate(configs):
                if cfg.from_expert:
                    self.expert_configs.append(cfg)
                else:
                    self.origin_configs.append(cfg)

    def add_mutibuffer_config(self):
        new_cfg = []
        for self_config in self.configs:
            self_config.kwargs['multibuffer'] = False
            config_copied = copy.deepcopy(self_config)
            config_copied.kwargs['multibuffer'] = True
            new_cfg.append(config_copied)
        self.configs.extend(new_cfg)

    def print_profile_value(self, this_pv: ProfileValue, full_info=False):
        if not this_pv:
            fast_a_log_message(content='no value', tag='print value')
            return
        if this_pv.profiler_time_sem is None:
            fast_a_log_message(content="config:{}, cost time:{:.3f}, expert: {}".format(
                this_pv.config.kwargs, this_pv.profiler_time, this_pv.config.from_expert))
        else:
            fast_a_log_message(content="config:{}, cost time:{:.3f}, sem:{:.3f}, expert: {}".format(
                this_pv.config.kwargs, this_pv.profiler_time,
                this_pv.profiler_time_sem, this_pv.config.from_expert
            ))

        if full_info:
            this_value_core_num = this_pv.config.vector_core_num
            if not self.max_ub_usage_each_core:
                fast_a_log_message(content="core num is {}/{}, ub usage is {}, expert: {}".format(
                    this_value_core_num, max(self.bucket_dict.keys()),
                    this_pv.config.ub_usage, this_pv.config.from_expert), tag="autotuner")
            else:
                if this_value_core_num not in self.max_ub_usage_each_core.keys():
                    fast_a_log_message(
                        content="core num is {}/{}, ub usage is {}, expert: {}".format(
                            this_value_core_num, max(self.bucket_dict.keys()),
                            this_pv.config.ub_usage, this_pv.config.from_expert), tag="autotuner")
                else:
                    fast_a_log_message(
                        content="core num is {}/{}, ub usage is {} (/{}={})%, circle num is {}, expert: {}".format(
                        this_value_core_num, max(self.bucket_dict.keys()),
                        this_pv.config.ub_usage, self.max_ub_usage_each_core[this_value_core_num],
                        this_pv.config.ub_usage / self.max_ub_usage_each_core[this_value_core_num] * 100,
                        this_pv.config.circle_num,
                        this_pv.config.from_expert), tag="autotuner")

    def _precompile_for_expert(self):
        fast_a_log_message(content='enter precompile for expert')
        self.bucket_dict = self._make_bucket_and_filter_with_binary()
        self._expert_configs_precompile()

    def autotuner(self, *args, stream, benchmark_run=False, **kwargs):
        if self.use_origin_autotuner:
            self.configs = self.expert_configs
            self.skip_precompile = False
            super().autotuner(*args, stream=stream, benchmark_run=benchmark_run, **kwargs)
            return

        if self.best_launcher is not None:
            self.launchers = [self.best_launcher]
            return

        self.skip_precompile = False
        best_launcher = self.auto_tune_by_fasta_parallel(*args, **kwargs)
        best_launcher_time = time.perf_counter() - self.autotune_start_time

        if best_launcher is None:
            super().autotuner(*args, stream=stream, benchmark_run=benchmark_run, **kwargs)
            return

        self.best_launcher = best_launcher
        self.launchers = [self.best_launcher]

        if self.save_cache_hook:
            self.save_cache_hook(self.launchers[0].config, best_launcher_time * 1e9)
        fast_a_log_message(content=f'autotuner time {best_launcher_time} s', tag='autotuner')

    def _filter_unable_compile_config_precompile(self, need_compile_configs):

        if not need_compile_configs:
            fast_a_log_message(content='no need compile configs', tag='filter precompile', level='warning')
            return None, None, None

        if not isinstance(need_compile_configs, list):
            need_compile_configs = [need_compile_configs]

        config_len = len(need_compile_configs)
        thread_num = min(config_len, npu_config.max_precompiled_thread_num)
        compile_results = [None] * config_len
        exc_results = [None] * config_len
        exc_stack_results = [None] * config_len

        with self.lock:
            with ThreadPoolExecutor(max_workers=thread_num) as executor:
                future_to_index = {
                    executor.submit(self._precompile_config, c): idx
                    for idx, c in enumerate(need_compile_configs)
                }

                for future in as_completed(future_to_index):
                    idx = future_to_index[future]
                    try:
                        result = future.result()
                        if result:
                            compile_results[idx] = result
                    except Exception as e:
                        import traceback
                        exc_stack_results[idx] = traceback.format_exc()
                        exc_results[idx] = e

        return compile_results, exc_results, exc_stack_results

    @staticmethod
    def _max_divisor_except_self(n, max_set=None):
        if n <= 1:
            return 1

        for i in range(1, int(n ** 0.5) + 1):
            if n % i == 0:

                if max_set is None or n // i <= max_set:
                    return n // i

        return n

    def _make_bucket_and_filter_with_binary(self):
        core_num_list = list(range(self.num_vector_core, 0, -1))
        max_valid_index_each_core = {}

        temp_config_dict = {}
        final_configs = []
        give_config_core_num = []
        for core_num in core_num_list:
            filter_config_by_core_num = [c for c in self.origin_configs if c.vector_core_num == core_num]
            if len(filter_config_by_core_num) > 0:
                filter_config_by_core_num.sort(key=lambda x: x.ub_usage, reverse=False)
                temp_config_dict[core_num] = BinaryBucket(core_num=core_num, config_list=filter_config_by_core_num)
                give_config_core_num.append(core_num)
                final_configs.append(filter_config_by_core_num[-1])

        right_indexes = {}
        ub_times_dict = {}
        no_filter_ub_core_nums = set()
        this_kernel_byte = get_byte_per_numel(self.inductor_meta["split_axis_dtype"])
        compile_results, exc_results, _ = self._filter_unable_compile_config_precompile(final_configs)
        for cn, cfg, er, cr in zip(give_config_core_num, final_configs, exc_results, compile_results):
            if cr is not None:
                self.max_ub_usage_each_core[cn] = cfg.ub_usage
                max_valid_index_each_core[cn] = len(temp_config_dict.get(cn).config_list) - 1
                no_filter_ub_core_nums.add(cn)
            else:
                pattern = r'requires (\d+) bits while (\d+) bits'
                results = re.findall(pattern, str(er))
                unique_results = list(set(results))
                max_req = 0
                for req, _ in unique_results:
                    max_req = max(int(req), max_req)

                if max_req < 1e-3:
                    fast_a_log_message(content="max_req is {}, error msg is {}".format(max_req, er),
                                       tag='binary filter', level='warning')
                    this_core_ub_index = len(temp_config_dict.get(cn).config_list) - 1
                else:
                    ub_times = max(int(max_req / (cfg.ub_size * this_kernel_byte * 8.)), 1)
                    threshold = 1. / ub_times
                    ub_times_dict[cn] = ub_times
                    this_core_ub_index = bisect.bisect_right(temp_config_dict.get(cn).config_list, threshold,
                                                             key=lambda x: x.ub_usage)
                    if this_core_ub_index >= len(temp_config_dict.get(cn).config_list):
                        this_core_ub_index = len(temp_config_dict.get(cn).config_list) - 1
                    elif temp_config_dict.get(cn).config_list[this_core_ub_index].ub_usage > threshold:
                        this_core_ub_index -= 1
                if this_core_ub_index <= 3:
                    this_core_ub_index = len(temp_config_dict.get(cn).config_list) - 1
                right_indexes[cn] = this_core_ub_index

        sorted_core_num_list = sorted(list(temp_config_dict.keys()), reverse=True)
        for this_cn in sorted_core_num_list:
            if this_cn in no_filter_ub_core_nums:
                self.min_core_num_with_no_filter_ub = this_cn
            else:
                break

        turn_num = 0
        if FASTA_SETTING.autotune_method == 'Expert':
            self._expert_method_get_choose_core_num_list(this_core_num_list=list(temp_config_dict.keys()))
            fast_a_log_message(content='expert_method_choose_core_num_list is {}'.format(
                self.expert_method_choose_core_num_list), tag='expert')
            if len(self.expert_method_choose_core_num_list) > 0:
                to_delete_cn = []
                for cn in right_indexes.keys():
                    if cn not in self.expert_method_choose_core_num_list:
                        to_delete_cn.append(cn)
                for cn in to_delete_cn:
                    new_right_index = max(0, right_indexes[cn])
                    self.max_ub_usage_each_core[cn] = temp_config_dict.get(cn).config_list[new_right_index].ub_usage
                    max_valid_index_each_core[cn] = new_right_index
                    right_indexes.pop(cn, None)

            if len(right_indexes) > 0:
                turn_num += 1
                temp_final_configs = []
                temp_give_config_core_num_with_index = []
                find_suitable_cn = set()
                for cn, this_right_index in right_indexes.items():
                    for i_sub in range(FASTA_SETTING.expert_find_ub_right_index):
                        now_index = this_right_index - i_sub
                        if now_index < 0:
                            break
                        temp_final_configs.append(temp_config_dict.get(cn).config_list[now_index])
                        temp_give_config_core_num_with_index.append([cn, now_index])
                compile_results, _, _ = self._filter_unable_compile_config_precompile(temp_final_configs)
                for cn_info, cfg, cr in zip(temp_give_config_core_num_with_index, temp_final_configs, compile_results):
                    if cr is not None:
                        cn = cn_info[0]
                        now_index = cn_info[1]
                        self.max_ub_usage_each_core[cn] = cfg.ub_usage
                        max_valid_index_each_core[cn] = max(max_valid_index_each_core.get(cn, -1), now_index)
                        self.max_ub_usage_each_core[cn] = temp_config_dict.get(cn).config_list[
                            max_valid_index_each_core[cn]].ub_usage
                        right_indexes.pop(cn, None)
                        find_suitable_cn.add(cn)

        temp_final_configs = []
        temp_give_config_core_num = []
        for cn, this_right_index in right_indexes.items():
            temp_config_dict.get(cn).update_right_index(new_right_index=this_right_index)
            if cn in ub_times_dict.keys():
                threshold_left = 1. / (ub_times_dict[cn] + 3)
                this_left_index = bisect.bisect_left(temp_config_dict.get(cn).config_list, threshold_left,
                                                         key=lambda x: x.ub_usage)
                this_left_index = max(0, this_left_index)
                if this_right_index - this_left_index > 3:
                    temp_config_dict.get(cn).update_left_index(new_left_index=this_left_index)
                else:
                    new_this_left_index = max(0, this_right_index - 3)
                    temp_config_dict.get(cn).update_left_index(new_left_index=new_this_left_index)

            temp_final_configs.append(temp_config_dict.get(cn).get_now_mid_config())
            temp_give_config_core_num.append(cn)
        final_configs = temp_final_configs
        give_config_core_num = temp_give_config_core_num

        if len(final_configs) > 0:
            while True:
                temp_final_configs = []
                temp_give_config_core_num = []
                compile_results, _, _ = self._filter_unable_compile_config_precompile(final_configs)
                for cn, cr in zip(give_config_core_num, compile_results):
                    new_config, bucket_stop = temp_config_dict.get(cn).get_mid_config(return_result=(cr is not None))
                    if bucket_stop:
                        self.max_ub_usage_each_core[cn] = new_config.ub_usage
                        max_valid_index_each_core[cn] = temp_config_dict.get(cn).result_index
                    else:
                        temp_final_configs.append(new_config)
                        temp_give_config_core_num.append(cn)
                if len(temp_final_configs) == 0:
                    break
                if turn_num >= FASTA_SETTING.find_bucket_max_ub_max_times:
                    for cn, cfg in zip(temp_give_config_core_num, temp_final_configs):
                        self.max_ub_usage_each_core[cn] = cfg.ub_usage
                        max_valid_index_each_core[cn] = temp_config_dict.get(cn).mid_index
                        fast_a_log_message(content='{} incorrect max ub usage is {:.4}'.format(cn, cfg.ub_usage),
                                           tag='binary filter')
                    break
                turn_num += 1
                final_configs = temp_final_configs
                give_config_core_num = temp_give_config_core_num

        fast_a_log_message(content='Expert method find ub max usage use {} times compile totally'.format(
            turn_num + 1), tag='binary filter')

        if self.precompiled_thread_num <= 10:
            bucket_config_list_setting = 10
        else:
            bucket_config_list_setting = self._max_divisor_except_self(self.precompiled_thread_num,
                                                                       max_set=FASTA_SETTING.bucket_max_config_num)
        target_core_num = self.num_vector_core * self.init_best_core_ratio
        self.record_core_best_value = {}
        ans = {}

        for core_num in temp_config_dict.keys():
            find_ub_index = max_valid_index_each_core.get(core_num)
            if find_ub_index >= 0:
                bucket = NPUFastABucket(core_num, temp_config_dict[core_num].config_list[:find_ub_index + 1],
                                        ub_max_valid_usage=self.max_ub_usage_each_core[core_num],
                                        config_list_length=bucket_config_list_setting,
                                        inductor_meta=self.inductor_meta,
                                        triton_meta=self.triton_meta,
                                        device_props=self.device_props,
                                        fn=self.fn)
                ans[core_num] = bucket
                self.bucket_score.append([core_num, abs(core_num - target_core_num)])
                self.record_core_best_value[core_num] = -1
            else:
                fast_a_log_message(content='core num {} has no valid config after binary filter, find ub index is {}'.format(
                    core_num, find_ub_index), tag='binary filter', level='warning')
        self.bucket_score.sort(key=lambda x: x[1])
        return ans

    def _update_value_record(self, bucket):
        if bucket.state == 'end':
            if bucket.best_config_profile_value:
                self.record_core_best_value[bucket.core_num] = bucket.best_config_profile_value.profiler_time
                if self.best_value_from_core < 0:
                    self.best_value_from_core = bucket.core_num
                    self.best_profiling_ub_usage = bucket.best_config_profile_value.config.ub_usage
                elif bucket.best_config_profile_value.profiler_time < self.record_core_best_value[
                    self.best_value_from_core]:
                    self.best_value_from_core = bucket.core_num
                    self.best_profiling_ub_usage = bucket.best_config_profile_value.config.ub_usage
                self.print_profile_value(bucket.best_config_profile_value, full_info=True)
            else:
                if bucket.core_num in self.record_core_best_value.keys():
                    del self.record_core_best_value[bucket.core_num]

    def dynamic_generate_config_parallel(self):
        if self.expert_configs_profiling:
            need_compile_configs = copy.deepcopy(self.expert_configs)
        else:
            need_compile_configs = []

        for _, bucket_info in enumerate(self.bucket_score):
            bucket = self.bucket_dict[bucket_info[0]]
            if self.best_profiling_ub_usage:
                ref_ub_usage = self.best_profiling_ub_usage
            else:
                ref_ub_usage = None

            t = bucket.dynamic_generate_config(ref_ub_usage)
            if t:
                if len(need_compile_configs) + len(t) <= self.precompiled_thread_num or len(need_compile_configs) == 0:
                    need_compile_configs.extend(t)
                else:
                    break
            else:
                self._update_value_record(bucket)
        return need_compile_configs

    def dispatch_profile_value(self, profile_value_list):
        temp_profile_value_dict = {}
        for pv in profile_value_list:
            temp_profile_value_dict.setdefault(pv.config.vector_core_num, [])
            temp_profile_value_dict[pv.config.vector_core_num].append(pv)

        for core_num, pv_list in temp_profile_value_dict.items():
            if core_num in self.bucket_dict.keys():
                self.bucket_dict[core_num].add_profile_value(pv_list)

    def _expert_method_get_choose_core_num_list(self, this_core_num_list):
        this_core_num_list.sort(reverse=True)
        self.expert_method_choose_core_num_list = []

        if self.min_core_num_with_no_filter_ub <= npu_config.num_vector_core:
            for cn in this_core_num_list:
                if cn < self.min_core_num_with_no_filter_ub:
                    break
                self.expert_method_choose_core_num_list.append(cn)
            if len(self.expert_method_choose_core_num_list) >= FASTA_SETTING.expert_min_bucket_num:
                return

        self.expert_method_choose_core_num_list = this_core_num_list[:FASTA_SETTING.expert_min_bucket_num]

    def _expert_configs_precompile(self):
        need_compile_configs = copy.deepcopy(self.expert_configs)

        core_num_list = sorted(list(self.bucket_dict.keys()), reverse=True)

        if len(self.expert_method_choose_core_num_list) < 1:
            self._expert_method_get_choose_core_num_list(this_core_num_list=core_num_list)
        fast_a_log_message(content='min_core_num_with_no_filter_ub is {}, choose core num {}'.format(
            self.min_core_num_with_no_filter_ub, self.expert_method_choose_core_num_list), tag='expert')

        tiling_axis_num = len(set(self.inductor_meta["tiling_axis"]))
        for core_num in self.expert_method_choose_core_num_list:
            need_compile_configs.extend(self.bucket_dict[core_num].get_configs_in_bucket_expert(
                get_circle_num=tiling_axis_num))

        self.configs = need_compile_configs
        self.add_mutibuffer_config()
        self.profiling_config_num += len(self.configs)
        self.precompile()

    def profiling_and_get_best_config(self, *args, **kwargs):
        timings = self.benchmark_all_configs_with_std(*args, **kwargs)
        profile_values = self.make_profile_values(timings)
        best_profile = None
        best_profile = self.find_best_launcher(profile_values, best_profile)
        self.clear_last_record()
        self.get_result_config_num += len(profile_values)
        fast_a_log_message(content="now our best profile is", tag='expert')
        self.print_profile_value(best_profile)
        return best_profile

    def auto_tune_by_fasta_parallel(self, *args, **kwargs):
        fast_a_log_message(content=f"======= enter fast autotuner kernel name: {self.get_fn_name()} =====")
        if FASTA_SETTING.autotune_method == 'Expert':
            best_profile = self.profiling_and_get_best_config(*args, **kwargs)
        elif FASTA_SETTING.autotune_method == "SampleStack":
            self.bucket_dict = self._make_bucket_and_filter_with_binary()
            best_profile = self._sample_stack(*args, **kwargs)
        else:
            fast_a_log_message(content=f"fast autotuner method{FASTA_SETTING.autotune_method} is err", tag='error')
            return None

        fast_a_log_message(content="finally our best profile is")
        self.print_profile_value(best_profile, full_info=True)
        fast_a_log_message(content="finally, our config num info is {}/{} (up/down), "
                                   "the kernel name: ({})".format(self.get_result_config_num,
                                                                  self.profiling_config_num, self.get_fn_name()))
        return best_profile.launcher

    def _sample_stack(self, *args, **kwargs):
        best_profile = None
        while not self._is_all_bucket_end() and self.find_best_times < self.max_find_best_times:
            self.find_best_times += 1
            need_compile_configs = self.dynamic_generate_config_parallel()
            if need_compile_configs is None or len(need_compile_configs) == 0:
                break

            self.configs = need_compile_configs
            self.add_mutibuffer_config()
            self.profiling_config_num += len(self.configs)
            ans = self.precompile()
            if ans == "NoneCompileResults":
                break

            timings = self.benchmark_all_configs_with_std(*args, **kwargs)
            profile_values = self.make_profile_values(timings)
            best_profile = self.find_best_launcher(profile_values, best_profile)
            self.dispatch_profile_value(profile_values)
            self.clear_last_record()
            self.get_result_config_num += len(profile_values)
            self.print_profile_value(best_profile)
        return best_profile

    def clear_last_record(self):
        self.configs = []
        self.launchers = []
        self.compile_results = []

    def make_profile_values(self, timings):
        ans = []
        for launcher, time_info in timings.items():
            tmp = ProfileValue(launcher=launcher, config=launcher.config,
                               profiler_time=time_info[0], profiler_time_sem=time_info[1],
                               profiler_time_std=time_info[2])
            self.print_profile_value(tmp)
            ans.append(tmp)
        if len(ans) == 0:
            fast_a_log_message(content='make profile values get no valid value',
                               tag='make profile value', level='error')
            raise ValueError("no valid profile data")
        return ans

    def find_best_launcher(self, profile_values, best_profile):
        ans = profile_values[0]
        for pv in profile_values:
            if ans.profiler_time > pv.profiler_time:
                ans = pv

        if best_profile is None:
            self.best_profiling_value = ans.profiler_time
            best_core_num = ans.config.vector_core_num
            for _, core_info in enumerate(self.bucket_score):
                core_info[1] = abs(core_info[0] - best_core_num)
            self.bucket_score.sort(key=lambda x: x[1])
            return ans
        elif best_profile.profiler_time < ans.profiler_time:
            self.best_profiling_value = best_profile.profiler_time
            return best_profile
        else:
            self.best_profiling_value = ans.profiler_time
            best_core_num = ans.config.vector_core_num
            for _, core_info in enumerate(self.bucket_score):
                core_info[1] = abs(core_info[0] - best_core_num)
            self.bucket_score.sort(key=lambda x: x[1])
            return ans

    def _is_all_bucket_end(self):
        if self.best_value_from_core > 0:
            find_num = 0
            this_core_num = self.best_value_from_core + 1
            while find_num < self.check_best_bucket_neighbor_num and this_core_num <= self.num_vector_core:
                if this_core_num in self.record_core_best_value.keys():
                    if self.record_core_best_value[this_core_num] > self.record_core_best_value[
                        self.best_value_from_core]:
                        find_num += 1
                    else:
                        return False
                this_core_num += 1

            find_num = 0
            this_core_num = self.best_value_from_core - 1
            while find_num < self.check_best_bucket_neighbor_num and this_core_num > 0:
                if this_core_num in self.record_core_best_value.keys():
                    if (self.record_core_best_value[this_core_num]
                            > self.record_core_best_value[self.best_value_from_core]):
                        find_num += 1
                    else:
                        return False
                this_core_num -= 1
            return True
        return False

    def benchmark_all_configs_with_std(self, *args, **kwargs):
        fast_a_log_message(content=f"candidate launcher count = {len(self.launchers)}", tag='benchmark profiling')

        tilling_kernel_list = []

        def kernel_call(this_launcher):
            def call_kernel():
                if this_launcher.config.pre_hook is not None:
                    this_launcher.config.pre_hook(
                        {**dict(zip(self.arg_names, args)), **this_launcher.config.kwargs}
                    )
                cloned_args, cloned_kwargs = self.clone_args(*args, **kwargs)
                this_launcher(
                    *cloned_args,
                    **cloned_kwargs,
                    stream=stream,
                )

            return call_kernel

        for launcher in self.launchers:
            if not self.custom_kernel and launcher.n_spills > config.triton.spill_threshold:
                return [float("inf"), None, None]

            device_interface = self.get_device_interface()
            stream = device_interface.get_raw_stream(device_interface.current_device())
            tilling_kernel_list.append(kernel_call(launcher))

        profiler_active_num = 10

        def delete_file(base_path):
            if os.path.exists(base_path):
                shutil.rmtree(base_path)

        experimental_config = torch_npu.profiler._ExperimentalConfig(
            aic_metrics=torch_npu.profiler.AiCMetrics.AiCoreNone,
            profiler_level=torch_npu.profiler.ProfilerLevel.Level0,
            l2_cache=False,
            data_simplification=False,
        )

        random_uuid = uuid.uuid4().hex
        md5_hash = hashlib.md5(random_uuid.encode()).hexdigest()

        def do_batch_benchmark_kernel_isolate(this_tilling_kernel_list, this_active_num):
            this_stream = torch.npu.current_stream()

            tiling_length = len(this_tilling_kernel_list)
            torch_path = os.path.join(os.getcwd(), "profile_result", f"triton_{md5_hash}")
            WAIT = 1
            WARMUP = 1
            ACTIVE = this_active_num
            SKIP_FIRST = 1
            TOTAL_STEP = WAIT + WARMUP + ACTIVE + SKIP_FIRST
            l2_cache_size = 192 * (1 << 20)  # 此处加入清L2的内容
            buffer = torch.empty(l2_cache_size // 4, dtype=torch.int, device="npu")
            with torch_npu.profiler.profile(
                    activities=[torch_npu.profiler.ProfilerActivity.NPU],
                    on_trace_ready=simple_trace_handler(torch_path),
                    record_shapes=False,
                    profile_memory=False,
                    with_stack=False,
                    with_flops=False,
                    with_modules=False,
                    experimental_config=experimental_config):
                this_stream.synchronize()
                for fn in this_tilling_kernel_list:
                    buffer.zero_()
                    for _ in range(TOTAL_STEP):
                        fn()
                        torch.npu.synchronize()
                this_stream.synchronize()
            del buffer

            for root, _, files in os.walk(torch_path):
                for file in files:
                    if file != 'kernel_details.csv':
                        continue
                    target_file = os.path.join(root, file)
                    df = pd.read_csv(target_file)
                    triton_rows = df[df['Name'].str.startswith('triton', na=False)]
                    time_cost = [0] * tiling_length
                    time_sem = [0] * tiling_length
                    time_std = [0] * tiling_length
                    if triton_rows.shape[0] % tiling_length != 0:
                        raise IndexError("triton_rows.shape[0] % tiling_length != 0")
                    avg_data_length = triton_rows.shape[0] // tiling_length
                    if avg_data_length <= ACTIVE:
                        raise IndexError("triton_rows.shape[0] // tiling_length <= ACTIVE")
                    for tiling_index in range(tiling_length):
                        start_index = tiling_index * avg_data_length + avg_data_length - ACTIVE
                        data_list = triton_rows.iloc[start_index:start_index + ACTIVE]['Duration(us)'].to_numpy()
                        time_cost[tiling_index] = np.mean(data_list)
                        time_std[tiling_index] = np.std(data_list)
                        time_sem[tiling_index] = time_std[tiling_index] / math.sqrt(len(data_list))

                    delete_file(torch_path)
                    return time_cost, time_sem, time_std

            delete_file(torch_path)
            return [], [], []

        try:
            this_benchmark_fun = do_batch_benchmark_kernel_isolate
            timing_list, timing_sem_list, timing_std_list = this_benchmark_fun(tilling_kernel_list,
                                                                               this_active_num=profiler_active_num)
            if not len(timing_list) == len(self.launchers):
                raise RuntimeError(f"not {len(timing_list)} == {len(self.launchers)}")
            timing_infos = {}
            for launcher, timing, sem, std in zip(self.launchers, timing_list, timing_sem_list, timing_std_list):
                timing_infos[launcher] = [timing, sem, std]

            if FASTA_SETTING.re_profiling:
                tim_min_index = np.argmin(timing_list)
                tim_deal_sem = np.array(timing_list) - np.array(timing_sem_list)
                tim_deal_sem[tim_min_index] += 2 * timing_sem_list[tim_min_index]
                need_test_times = 1
                need_test_index = []
                sem_to_std_para = math.sqrt(profiler_active_num)
                for idx, val in enumerate(tim_deal_sem):
                    if idx != tim_min_index and val <= tim_deal_sem[tim_min_index]:
                        diff_time = timing_list[idx] - timing_list[tim_min_index]
                        estimate_sum_std = (timing_sem_list[idx] * sem_to_std_para +
                                            timing_sem_list[tim_min_index] * sem_to_std_para)
                        this_need_test_times = int(pow((estimate_sum_std / diff_time), 2))
                        need_test_times = max(need_test_times, this_need_test_times)
                        need_test_index.append(idx)
                if need_test_index:
                    need_test_times = min(10 * (need_test_times + 10 - 1) // 10,
                                          FASTA_SETTING.re_profiling_max_times)
                    fast_a_log_message(
                        content="need re_profiling times is ({}) for index {}".format(
                            need_test_times, need_test_index), tag='re_profiling')
                    need_test_index.append(tim_min_index)
                    re_kernel_list = [tilling_kernel_list[re_i] for re_i in need_test_index]
                    profiler_active_num = need_test_times
                    re_timing_list, re_timing_sem_list, re_timing_std_list = this_benchmark_fun(re_kernel_list,
                                                                                                this_active_num=profiler_active_num)
                    for idx, re_i in enumerate(need_test_index):
                        timing_infos[self.launchers[re_i]] = [re_timing_list[idx], re_timing_sem_list[idx],
                                                              re_timing_std_list[idx]]

        except Exception as e:
            fast_a_log_message(
                content='some cases in batch benchmark has error! Logging Exception as:\n{}\nswitched to single bench...'.format(e),
                tag='benchmark profiling', level='warning')
            timing_infos = {}
            for launcher in self.launchers:
                try:
                    fast_a_log_message(content=f"single bench config is ({launcher.config})", tag='benchmark profiling')
                    timing_infos[launcher] = self.bench_event(launcher, *args, **kwargs)
                except Exception as f:
                    fast_a_log_message(content=f"the case ({launcher.config}) is error and we ignore it\n" +
                                       f"error msg is {f}", tag='benchmark profiling', level='warning')

        for k, v in timing_infos.items():
            self.coordesc_tuner.cache_benchmark_result(k.config, v[0])

        return timing_infos

    def bench_event(self, launcher, *args, **kwargs):
        """Measure the performance of a given launcher"""

        if not self.custom_kernel and launcher.n_spills > self.inductor_meta.get(
                "spill_threshold", 16
        ):
            return [float("inf"), None, None]

        device_interface = self.get_device_interface()
        stream = device_interface.get_raw_stream(device_interface.current_device())

        def kernel_call():
            cloned_args, cloned_kwargs = self.clone_args(*args, **kwargs)
            launcher(
                *cloned_args,
                **cloned_kwargs,
                stream=stream,
            )

        return self._do_bench_with_event(kernel_call)

    @staticmethod
    def _do_bench_with_event(fn, warmup_times=1, rep_times=2):
        """
        Benchmark the runtime of the provided function. By default, return the median runtime of :code:`fn` along with
        the 20-th and 80-th performance percentile.

        :param fn: Function to benchmark
        :type fn: Callable
        :param warmup_times: Warmup times
        :param rep_times: Repetition times
        """

        di = triton.runtime.driver.active.get_device_interface()

        fn()
        di.synchronize()

        cache = triton.runtime.driver.active.get_empty_cache_for_benchmark()

        # compute number of warmup and repeat
        start_event = [di.Event(enable_timing=True) for i in range(rep_times)]
        end_event = [di.Event(enable_timing=True) for i in range(rep_times)]
        # Warm-up
        cache.zero_()
        for _ in range(warmup_times):
            fn()
        # Benchmark
        for i in range(rep_times):

            if FASTA_SETTING.profiling_clear_L2:
                # we clear the L2 cache before each run
                cache.zero_()
            # record time of `fn`
            start_event[i].record()
            fn()
            end_event[i].record()
        # Record clocks
        di.synchronize()
        times = torch.tensor([s.elapsed_time(e) for s, e in zip(start_event, end_event)], dtype=torch.float)
        std, mean = torch.std_mean(times)
        sem = std.item() / math.sqrt(len(times))
        return [mean.item() * 1e3, sem * 1e3, std.item() * 1e3]