import torch
from torch import inf
from torch.optim.optimizer import Optimizer

import torch_npu
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
        self.is_grads_masks_combined = False
        self.is_fused_optimizer = True

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
                if torch_npu.get_npu_format(p) != torch_npu.get_npu_format(p.grad):
                    p.grad = torch_npu.npu_format_cast(
                        p.grad, torch_npu.get_npu_format(p)).contiguous()
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
        self.params_all_group = params_all_group

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

    def _clip_grad_norm_fused_(self, combined_grads, combined_grads_masks, max_norm, norm_type):
        if len(combined_grads) != len(combined_grads_masks):
            raise ValueError("Length of combined_grads and combined_grads_masks must be equal.")
        if len(combined_grads) == 0 or all(i is None for i in combined_grads):
            return torch.tensor(0.)

        max_norm = float(max_norm)
        norm_type = float(norm_type)
        masked_grad_list = []
        if norm_type == inf:
            for combined_grad, combined_grad_mask in zip(combined_grads, combined_grads_masks):
                if combined_grad is not None:
                    masked_grad_list.append(combined_grad.float().abs().mul_(combined_grad_mask).max())
            total_norm = max(masked_grad_list)
        else:
            for combined_grad, combined_grad_mask in zip(combined_grads, combined_grads_masks):
                if combined_grad is not None:
                    masked_grad_list.append(combined_grad.float().abs().pow(norm_type).mul_(combined_grad_mask).sum())
            total_norm = torch.stack(masked_grad_list).sum().pow(1 / norm_type)
        clip_coef = max_norm / (total_norm + 1e-6)
        if clip_coef < 1:
            for combined_grad in combined_grads:
                if combined_grad is not None:
                    combined_grad.mul_(clip_coef)
        return total_norm

    def _combine_grads_mask(self, list_of_params):
        if len(list_of_params) == 0:
            return None

        list_of_grads_masks = []
        for param in list_of_params:
            if param.requires_grad:
                grad_size = param.grad.size()
                grad_format = torch_npu.get_npu_format(param)
                list_of_grads_masks.append(torch_npu.npu_format_cast(torch.ones(grad_size).npu(), grad_format))
        grad_mask_combined = npu_combine_tensors(list_of_grads_masks)

        return grad_mask_combined

    def _maybe_init_combined_grads_masks(self):
        # Create a mask to ensure the padded data to be zero in case of combining tensors with NPU-private format.
        if not self.is_params_grads_combined:
            raise ValueError("Value of param 'is_params_grads_combined' must be True")

        combined_grads_masks = []
        for params_group_one_dtype in self.params_all_group:
            combined_grads_mask = self._combine_grads_mask(params_group_one_dtype)
            combined_grads_masks.append(combined_grads_mask)

        self.combined_grads_masks = combined_grads_masks
        self.is_grads_masks_combined = True

    def _get_combined_grad_masks(self):
        return self.combined_grads_masks

    def clip_grad_norm_fused_(self, max_norm, norm_type=2):
        if not self.is_params_grads_combined:
            with torch.no_grad():
                self._maybe_init_combined_params_and_grads()

        if not self.is_grads_masks_combined:
            self._maybe_init_combined_grads_masks()

        combined_grads = self.get_combined_grads()
        combined_grads_masks = self._get_combined_grad_masks()

        return self._clip_grad_norm_fused_(combined_grads, combined_grads_masks, max_norm, norm_type)
