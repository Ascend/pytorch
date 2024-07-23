from collections import defaultdict

import torch
from torch_npu.utils import npu_combine_tensors
from torch_npu.utils._error_code import ErrCode, pta_error
from .npu_fused_optim_base import NpuFusedOptimizerBase

__all__ = ["NpuFusedAdadelta"]


class NpuFusedAdadelta(NpuFusedOptimizerBase):
    """Implements NpuFusedAdadelta algorithm.

    It has been proposed in `ADADELTA: An Adaptive Learning Rate Method`__.

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional, default=1.0): coefficient that scale delta before it is applied
            to the parameters
        rho (float, optional, default=0.9): coefficient used for computing a running average
            of squared gradients
        eps (float, optional, default=1e-6): term added to the denominator to improve
            numerical stability
        weight_decay (float, optional, default=0): weight decay (L2 penalty)

    """

    def __init__(self, params, lr=1.0, rho=0.9, eps=1e-6, weight_decay=0):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr) + pta_error(ErrCode.VALUE))
        if rho < 0.0 or rho > 1.0:
            raise ValueError("Invalid rho value: {}".format(rho) + pta_error(ErrCode.VALUE))
        if eps < 0.0:
            raise ValueError("Invalid epsilon value: {}".format(eps) + pta_error(ErrCode.VALUE))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay) + pta_error(ErrCode.VALUE))

        defaults = dict(lr=lr, rho=rho, eps=eps, weight_decay=weight_decay)
        super(NpuFusedAdadelta, self).__init__(params, defaults)

    def _init_param_state(self, p):
        state = self.state[p]
        # State initialization
        if len(state) == 0:
            state['step'] = 0
            state['square_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
            state['acc_delta'] = torch.zeros_like(p, memory_format=torch.preserve_format)
        else:
            square_avg_tmp = torch.zeros_like(p, memory_format=torch.preserve_format)
            square_avg_tmp.copy_(state['square_avg'])
            state['square_avg'] = square_avg_tmp

            acc_delta_tmp = torch.zeros_like(p, memory_format=torch.preserve_format)
            acc_delta_tmp.copy_(state['acc_delta'])
            state['acc_delta'] = acc_delta_tmp

    def _combine_group_param_states(self, group_index):
        group_params_list = self.params_lists_indexed_by_group[group_index]

        combined_param_states = []
        for params in group_params_list:
            step_list = []
            square_avg_list = []
            acc_delta_list = []

            for p in params:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError('NpuFusedAdadelta does not support sparse gradients' +
                                       pta_error(ErrCode.NOT_SUPPORT))
                
                self._init_param_state(p)
                state = self.state[p]
                step_list.append(state['step'])
                square_avg_list.append(state['square_avg'])
                acc_delta_list.append(state['acc_delta'])
            
            combined_step = 0
            combined_square_avg = None
            combined_acc_delta = None

            if len(square_avg_list) > 0:
                combined_step = step_list[0]
                combined_square_avg = npu_combine_tensors(square_avg_list)
                combined_acc_delta = npu_combine_tensors(acc_delta_list)
            
            combined_state = defaultdict(dict)
            combined_state['step'] = combined_step
            combined_state['square_avg'] = combined_square_avg
            combined_state['acc_delta'] = combined_acc_delta
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
                raise RuntimeError('NpuFusedAdadelta does not support sparse gradients' +
                                   pta_error(ErrCode.NOT_SUPPORT))
            state_p = self.state[p]
            state_p['step'] += 1

        rho, eps = group['rho'], group['eps']

        combined_group_params = self.combined_params_indexed_by_group[group_index]
        combined_group_grads = self.combined_grads_indexed_by_group[group_index]
        combined_group_param_states = self.combined_param_states_indexed_by_group[group_index]

        for combined_param, combined_grad, combined_param_state in zip(combined_group_params, 
                                                                       combined_group_grads, 
                                                                       combined_group_param_states):
            if combined_param is None or combined_grad is None:
                continue

            square_avg, acc_delta = combined_param_state['square_avg'], combined_param_state['acc_delta']
            combined_param_state['step'] += 1

            if group['weight_decay'] != 0:
                combined_grad = combined_grad.add(combined_param, alpha=group['weight_decay'])

            square_avg.mul_(rho).addcmul_(combined_grad, combined_grad, value=1 - rho)
            std = square_avg.add(eps).sqrt_()
            delta = acc_delta.add(eps).sqrt_().div_(std).mul_(combined_grad)
            combined_param.add_(delta, alpha=-group['lr'])
            acc_delta.mul_(rho).addcmul_(delta, delta, value=1 - rho)
