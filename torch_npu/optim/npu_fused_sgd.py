from collections import defaultdict
import math

import torch
from torch.optim.optimizer import required

from torch_npu.utils import npu_combine_tensors
from .npu_fused_optim_base import NpuFusedOptimizerBase

LR_MIN = 0.0
MOMENTUM_MIN = 0.0
DAMPENING_DEFAULT = 0.0
WEIGHT_DECAY_MIN = 0.0


class NpuFusedSGD(NpuFusedOptimizerBase):
    r"""Implements stochastic gradient descent (optionally with momentum).

    Nesterov momentum is based on the formula from
    `On the importance of initialization and momentum in deep learning`__.

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float): learning rate
        momentum (float, optional, default=0): momentum factor
        weight_decay (float, optional, default=0): weight decay (L2 penalty)
        dampening (float, optional, default=0): dampening for momentum
        nesterov (bool, optional, default=False): enables Nesterov momentum
    """

    def __init__(self,
                 params,
                 lr=required,
                 momentum=MOMENTUM_MIN,
                 dampening=DAMPENING_DEFAULT,
                 weight_decay=WEIGHT_DECAY_MIN,
                 nesterov=False):
        if lr is not required and lr < LR_MIN:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < MOMENTUM_MIN:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < WEIGHT_DECAY_MIN:
            raise ValueError(
                "Invalid weight_decay value: {}".format(weight_decay))
        if nesterov and (momentum <= MOMENTUM_MIN
                         or not math.isclose(dampening, DAMPENING_DEFAULT, abs_tol=1e-15)):
            raise ValueError(
                "Nesterov momentum requires a momentum and zero dampening")
        defaults = dict(lr=lr,
                        momentum=momentum,
                        dampening=dampening,
                        weight_decay=weight_decay,
                        nesterov=nesterov)
        self._momentum_buffer_already_in_state = False
        super(NpuFusedSGD, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(NpuFusedSGD, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    def _init_param_state(self, p, weight_decay):
        d_p = p.grad
        state = self.state[p]
        if 'momentum_buffer' not in state:
            # the first update of sgd only use the weight decayed grad as buf
            self._momentum_buffer_already_in_state = False
            if weight_decay != 0:
                d_p = d_p.add(p, alpha=weight_decay)
            state['momentum_buffer'] = torch.clone(d_p).detach()
        else:
            self._momentum_buffer_already_in_state = True
            temp = torch.clone(d_p).detach()
            temp.copy_(state['momentum_buffer'])
            state['momentum_buffer'] = temp

    def _combine_group_param_states(self, group_index):
        group = self.param_groups[group_index]
        group_params = self.params_lists_indexed_by_group[group_index]

        weight_decay = group['weight_decay']
        momentum = group['momentum']

        group_combined_states = []
        for group_params_one_dtype in group_params:
            if momentum == 0:
                group_params_one_dtype = defaultdict(dict)
                group_params_one_dtype['momentum_buffer'] = None
                group_combined_states.append(group_params_one_dtype)
            else:
                momentum_buffer_list = []
                for p in group_params_one_dtype:
                    if p.grad is None:
                        continue

                    self._init_param_state(p, weight_decay)
                    state = self.state[p]
                    momentum_buffer_list.append(state['momentum_buffer'])

                combined_momentum_buffer = None
                if len(momentum_buffer_list) > 0:
                    combined_momentum_buffer = npu_combine_tensors(
                        momentum_buffer_list)

                combined_states_one_dtype = defaultdict(dict)
                combined_states_one_dtype[
                    'momentum_buffer'] = combined_momentum_buffer
                group_combined_states.append(combined_states_one_dtype)

        self.combined_param_states_indexed_by_group[
            group_index] = group_combined_states

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
        weight_decay = group['weight_decay']
        momentum = group['momentum']
        dampening = group['dampening']
        nesterov = group['nesterov']

        combined_group_params = self.combined_params_indexed_by_group[
            group_index]
        combined_group_grads = self.combined_grads_indexed_by_group[
            group_index]
        combined_group_param_states = self.combined_param_states_indexed_by_group[
            group_index]

        for combined_param_one_dtype, combined_grad_one_dtype, combined_param_state_one_dtype in zip(
                combined_group_params, combined_group_grads,
                combined_group_param_states):
            if combined_param_one_dtype is None or combined_grad_one_dtype is None:
                continue

            if weight_decay != 0:
                combined_grad_one_dtype = combined_grad_one_dtype.add(
                    combined_param_one_dtype, alpha=weight_decay)
            if momentum != 0:
                buf = combined_param_state_one_dtype['momentum_buffer']
                if self._momentum_buffer_already_in_state:
                    buf.mul_(momentum).add_(combined_grad_one_dtype,
                                            alpha=1 - dampening)

                if nesterov:
                    combined_grad_one_dtype = combined_grad_one_dtype.add(
                        buf, alpha=momentum)
                else:
                    combined_grad_one_dtype = buf

            combined_param_one_dtype.add_(combined_grad_one_dtype,
                                          alpha=-group['lr'])
    
    def step(self, closure=None):
        ret = super().step(closure)
        self._momentum_buffer_already_in_state = True
        return ret
