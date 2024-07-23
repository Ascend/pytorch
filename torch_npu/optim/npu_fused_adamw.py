import math
from collections import defaultdict

import torch
from torch_npu.utils import npu_combine_tensors
from torch_npu.utils._error_code import ErrCode, pta_error
from .npu_fused_optim_base import NpuFusedOptimizerBase

__all__ = ["NpuFusedAdamW"]


class NpuFusedAdamW(NpuFusedOptimizerBase):
    r"""Implements AdamW algorithm.

    For further details regarding the algorithm we refer to `Decoupled Weight Decay Regularization`_.
    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay coefficient (default: 1e-2)
        amsgrad (boolean, optional): whether to use the AMSGrad variant of this
            algorithm from the paper `On the Convergence of Adam and Beyond`_
            (default: False)
        maximize (bool, optional): maximize the params based on the objective, instead of
            minimizing (default: False)
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=1e-2, amsgrad=False):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr) + pta_error(ErrCode.VALUE))
        if eps < 0.0:
            raise ValueError("Invalid epsilon value: {}".format(eps) + pta_error(ErrCode.VALUE))
        if betas[0] < 0.0 or betas[0] >= 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]) + pta_error(ErrCode.VALUE))
        if betas[1] < 0.0 or betas[1] >= 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]) + pta_error(ErrCode.VALUE))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay) + pta_error(ErrCode.VALUE))
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad)
        super(NpuFusedAdamW, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(NpuFusedAdamW, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)

    def _init_param_state(self, p, amsgrad):
        state = self.state[p]
        # State initialization
        if len(state) == 0:
            state['step'] = 0
            # Exponential moving average of gradient values
            state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
            # Exponential moving average of squared gradient values
            state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
            if amsgrad:
                # Maintains max of all exp. moving avg. of sq. grad. values
                state['max_exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
        else:
            exp_avg_tmp = torch.zeros_like(p, memory_format=torch.preserve_format)
            exp_avg_tmp.copy_(state['exp_avg'])
            state['exp_avg'] = exp_avg_tmp

            exp_avg_sq_tmp = torch.zeros_like(p, memory_format=torch.preserve_format)
            exp_avg_sq_tmp.copy_(state['exp_avg_sq'])
            state['exp_avg_sq'] = exp_avg_sq_tmp

            if amsgrad:
                max_exp_avg_sq_tmp = torch.zeros_like(p, memory_format=torch.preserve_format)
                max_exp_avg_sq_tmp.copy_(state['max_exp_avg_sq'])
                state['max_exp_avg_sq'] = max_exp_avg_sq_tmp

    def _combine_group_param_states(self, group_index):
        group = self.param_groups[group_index]
        group_params_list = self.params_lists_indexed_by_group[group_index]

        amsgrad = group['amsgrad']

        combined_param_states = []
        for params in group_params_list:
            step_list = []
            exp_avg_list = []
            exp_avg_sq_list = []
            max_exp_avg_sq_list = []

            for p in params:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError('NpuFusedAdamW does not support sparse gradients, '
                                       'please consider SparseAdam instead' + pta_error(ErrCode.NOT_SUPPORT))

                self._init_param_state(p, amsgrad)
                state = self.state[p]
                step_list.append(state['step'])
                exp_avg_list.append(state['exp_avg'])
                exp_avg_sq_list.append(state['exp_avg_sq'])
                if amsgrad:
                    max_exp_avg_sq_list.append(state['max_exp_avg_sq'])

            combined_step = 0
            combined_exp_avg = None
            combined_exp_avg_sq = None
            combined_max_exp_avg_sq = None

            if len(exp_avg_list) > 0:
                combined_step = step_list[0]
                combined_exp_avg = npu_combine_tensors(exp_avg_list)
                combined_exp_avg_sq = npu_combine_tensors(exp_avg_sq_list)
                combined_max_exp_avg_sq = npu_combine_tensors(max_exp_avg_sq_list)

            combined_state = defaultdict(dict)
            combined_state['step'] = combined_step
            combined_state['exp_avg'] = combined_exp_avg
            combined_state['exp_avg_sq'] = combined_exp_avg_sq
            combined_state['max_exp_avg_sq'] = combined_max_exp_avg_sq
            combined_param_states.append(combined_state)
        self.combined_param_states_indexed_by_group[group_index] = combined_param_states

    def _maybe_init_combined_states(self):
        if self.is_states_combined:
            return
        self.combined_param_states_indexed_by_group = len(self.param_groups) * [None]

        for i, _ in enumerate(self.param_groups):
            self._combine_group_param_states(i)

        if not all(value is None for value in self.combined_param_states_indexed_by_group):
            self.is_states_combined = True

    def _group_step(self, group_index):
        group = self.param_groups[group_index]
        for p in group['params']:
            if p.grad is None:
                continue

            grad = p.grad
            if grad.is_sparse:
                raise RuntimeError('NpuFusedAdamW does not support sparse gradients, '
                                   'please consider SparseAdam instead' + pta_error(ErrCode.NOT_SUPPORT))
            state_p = self.state[p]
            state_p['step'] += 1

        amsgrad = group['amsgrad']
        beta1, beta2 = group['betas']

        combined_group_params = self.combined_params_indexed_by_group[group_index]
        combined_group_grads = self.combined_grads_indexed_by_group[group_index]
        combined_group_param_states = self.combined_param_states_indexed_by_group[group_index]

        for combined_param, combined_grad, combined_param_state in zip(combined_group_params,
                                                                       combined_group_grads,
                                                                       combined_group_param_states):
            if combined_param is None or combined_grad is None:
                continue

            # Perform stepweight decay. The fused method is used here to speed up the calculation
            combined_param.mul_(1 - group['lr'] * group['weight_decay'])

            exp_avg, exp_avg_sq = combined_param_state['exp_avg'], combined_param_state['exp_avg_sq']
            if amsgrad:
                max_exp_avg_sq = combined_param_state['max_exp_avg_sq']

            combined_param_state['step'] += 1
            bias_correction1 = 1 - beta1 ** combined_param_state['step']
            bias_correction2 = 1 - beta2 ** combined_param_state['step']

            # Decay the first and second moment running average coefficient
            exp_avg.mul_(beta1).add_(combined_grad, alpha=1 - beta1)
            exp_avg_sq.mul_(beta2).addcmul_(combined_grad, combined_grad, value=1 - beta2)
            if amsgrad:
                # Maintains the maximum of all 2nd moment running avg. till now
                torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                # Use the max. for normalizing running avg. of gradient
                denom = (max_exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])
            else:
                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])

            step_size = group['lr'] / bias_correction1

            combined_param.addcdiv_(exp_avg, denom, value=-step_size)
