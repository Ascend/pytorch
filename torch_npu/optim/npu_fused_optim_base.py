# Copyright (c) 2023, Huawei Technologies.
# Copyright (c) 2019, Facebook CORPORATION.
# All rights reserved.
#
# Licensed under the BSD 3-Clause License  (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://opensource.org/licenses/BSD-3-Clause
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
from torch.optim.optimizer import Optimizer
from torch_npu.utils import npu_combine_tensors, get_part_combined_tensor


class NpuFusedOptimizerBase(Optimizer):

    def __init__(self, params, default):
        super().__init__(params, default)
        self.combined_params_indexed_by_group = None
        self.combined_grads_indexed_by_group = None
        self.combined_param_states_indexed_by_group = None
        self.params_lists_indexed_by_group = []
        self.params_all_group_combined = 2 * [None]
        self.grads_all_group_combined = 2 * [None]
        self.is_params_grads_combined = False
        self.is_states_combined = False

    def _maybe_init_combined_params_and_grads(self):
        if self.is_params_grads_combined:
            return

        self.combined_params_indexed_by_group = len(self.param_groups) * [[]]
        self.combined_grads_indexed_by_group = len(self.param_groups) * [[]]        

        params_list_each_group = []
        params_size_each_group = []
        grads_size_each_group = []
        params_all_group = [[], []]
        grads_all_group = [[], []]
        for group in self.param_groups:
            group_params_list = [[], []]
            group_grads_list = [[], []]
            group_params_size = [0, 0]
            group_grads_size = [0, 0]

            for p in group['params']:
                if p.grad is None:
                    continue
                if torch.get_npu_format(p) != torch.get_npu_format(p.grad):
                    p.grad = torch.npu_format_cast(
                        p.grad, torch.get_npu_format(p)).contiguous()
                param_size = p.storage().size()
                grad_size = p.grad.storage().size()
                if p.dtype == torch.float32:
                    group_params_size[0] += param_size
                    group_params_list[0].append(p)

                    group_grads_size[0] += grad_size
                    group_grads_list[0].append(p.grad)
                elif p.dtype == torch.float16:
                    group_params_size[1] += param_size
                    group_params_list[1].append(p)

                    group_grads_size[1] += grad_size
                    group_grads_list[1].append(p.grad)
                else:
                    raise TypeError(
                        "Fused optimizer's parameters must be either float32 or float16, but received {}"
                        .format(p.dtype))

            params_all_group[0] += group_params_list[0]
            params_all_group[1] += group_params_list[1]
            grads_all_group[0] += group_grads_list[0]
            grads_all_group[1] += group_grads_list[1]

            params_list_each_group.append(group_params_list)
            params_size_each_group.append(group_params_size)
            grads_size_each_group.append(group_grads_size)

        self.params_lists_indexed_by_group = params_list_each_group

        for dtype_index, _ in enumerate(params_all_group):
            self.params_all_group_combined[dtype_index] = npu_combine_tensors(params_all_group[dtype_index])
            self.grads_all_group_combined[dtype_index] = npu_combine_tensors(grads_all_group[dtype_index])

        params_offset = len(self.params_all_group_combined) * [0]
        grads_offset = len(self.grads_all_group_combined) * [0]
        for group_index, _ in enumerate(params_list_each_group):
            group_combined_params, group_combined_grads = [], []
            for dtype_index, _ in enumerate(params_list_each_group[group_index]):
                combined_params_one_dtype = get_part_combined_tensor(
                    self.params_all_group_combined[dtype_index], params_offset[dtype_index],
                    params_size_each_group[group_index][dtype_index])
                params_offset[dtype_index] += params_size_each_group[group_index][
                    dtype_index]
                group_combined_params.append(combined_params_one_dtype)

                combined_grads_one_dtype = get_part_combined_tensor(
                    self.grads_all_group_combined[dtype_index], grads_offset[dtype_index],
                    grads_size_each_group[group_index][dtype_index])
                grads_offset[dtype_index] += grads_size_each_group[group_index][dtype_index]
                group_combined_grads.append(combined_grads_one_dtype)

            self.combined_params_indexed_by_group[group_index] = group_combined_params
            self.combined_grads_indexed_by_group[group_index] = group_combined_grads
        
        if not all(value is None for value in self.params_all_group_combined):
            self.is_params_grads_combined = True        

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        if not self.is_params_grads_combined:
            self._maybe_init_combined_params_and_grads()
        
        if not self.is_states_combined:
            self._maybe_init_combined_states()

        for i, _ in enumerate(self.param_groups):
            self._group_step(i)

        return loss

    @torch.no_grad()
    def zero_grad(self, set_to_none=False):
        if set_to_none:
            raise ValueError(
                "set_to_none is not supported in fused optimizers")

        if not self.is_params_grads_combined:
            self._maybe_init_combined_params_and_grads()
            if not self.is_params_grads_combined:
                super().zero_grad(set_to_none)
                return
                
        for grads_combined_one_dtype in self.grads_all_group_combined:
            if grads_combined_one_dtype is None:
                continue
            grads_combined_one_dtype.zero_()

    def _maybe_init_combined_states(self):
        raise NotImplementedError

    def _group_step(self, group_index):
        raise NotImplementedError

    def get_combined_params(self):
        return self.params_all_group_combined

    def get_combined_grads(self):
        return self.grads_all_group_combined
