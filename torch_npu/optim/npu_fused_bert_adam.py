import math
from collections import defaultdict

import torch
from torch.optim.optimizer import required
from torch_npu.utils import npu_combine_tensors
from .npu_fused_optim_base import NpuFusedOptimizerBase


WARMUP_DEFAULT = 0.002
DEGREE_DEFAULT = 0.5


def _clip_grad_norm_(combined_param, combined_grad, max_norm):
    if combined_param is None or combined_grad is None:
        return
    norm_type = 2.0
    total_norm = combined_grad.float().abs().pow(norm_type).sum().pow(1. / norm_type)
    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1.:
        combined_grad.mul_(clip_coef)


def warmup_cosine(x, warmup=WARMUP_DEFAULT):
    if x < warmup:
        return x / warmup
    return 0.5 * (1.0 + torch.cos(math.pi * x))


def warmup_constant(x, warmup=WARMUP_DEFAULT):
    if x < warmup:
        return x / warmup
    return 1.0


def warmup_linear(x, warmup=WARMUP_DEFAULT):
    if x < warmup:
        return x / warmup
    return max((x - 1.) / (warmup - 1.), 0.)


def warmup_poly(x, warmup=WARMUP_DEFAULT, degree=DEGREE_DEFAULT):
    if x < warmup:
        return x / warmup
    return (1.0 - x)**degree


SCHEDULES = {
    'warmup_cosine':warmup_cosine,
    'warmup_constant':warmup_constant,
    'warmup_linear':warmup_linear,
    'warmup_poly':warmup_poly,
}


class NpuFusedBertAdam(NpuFusedOptimizerBase):
    """Implements BERT version of Adam algorithm with weight decay fix. This is the fused version on NPU
    Params:
        lr: learning rate
        warmup: portion of t_total for the warmup, -1  means no warmup. Default: -1
        t_total: total number of training steps for the learning
            rate schedule, -1  means constant learning rate. Default: -1
        schedule: schedule to use for the warmup (see above). Default: 'warmup_linear'
        b1: Adams b1. Default: 0.9
        b2: Adams b2. Default: 0.999
        e: Adams epsilon. Default: 1e-6
        weight_decay: Weight decay. Default: 0.01
        max_grad_norm: Maximum norm for the gradients (-1 means no clipping). Default: 1.0
    """

    def __init__(self, params, lr=required, warmup=-1, t_total=-1, schedule='warmup_linear',
                 b1=0.9, b2=0.999, e=1e-6, weight_decay=0.01,
                 max_grad_norm=1.0):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if (warmup < 0.0 and warmup != -1) or warmup >= 1.0:
            raise ValueError("Invalid warmup: {}".format(warmup))
        if schedule not in SCHEDULES:
            raise ValueError("Invalid schedule parameter: {}".format(schedule))
        if b1 < 0.0 or b1 >= 1.0:
            raise ValueError("Invalid b1 parameter: {}".format(b1))
        if b2 < 0.0 or b2 >= 1.0:
            raise ValueError("Invalid b2 parameter: {}".format(b2))
        if e < 0.0:
            raise ValueError("Invalid epsilon value: {}".format(e))
        defaults = dict(lr=lr, schedule=schedule, warmup=warmup, t_total=t_total,
                        b1=b1, b2=b2, e=e, weight_decay=weight_decay,
                        max_grad_norm=max_grad_norm)
        self.max_grad_norm = max_grad_norm
        super(NpuFusedBertAdam, self).__init__(params, defaults)

    def _init_param_state(self, p):
        state = self.state[p]
        # state initialization
        if len(state) == 0:
            state['step'] = 0
            # Exponential moving average of gradient values
            state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
            # Exponential moving average of squared gradient values
            state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
        else:
            exp_avg_tmp = torch.zeros_like(p, memory_format=torch.preserve_format)
            exp_avg_tmp.copy_(state['exp_avg'])
            state['exp_avg'] = exp_avg_tmp

            exp_avg_sq_tmp = torch.zeros_like(p, memory_format=torch.preserve_format)
            exp_avg_sq_tmp.copy_(state['exp_avg_sq'])
            state['exp_avg_sq'] = exp_avg_sq_tmp

    def _combine_group_param_states(self, group_index):
        group_params_list = self.params_lists_indexed_by_group[group_index]

        combined_param_states = []
        for params in group_params_list:
            step_list = []
            exp_avg_list = []
            exp_avg_sq_list = []

            for p in params:
                if p.grad is None:
                    continue

                self._init_param_state(p)
                state = self.state[p]
                step_list.append(state['step'])
                exp_avg_list.append(state['exp_avg'])
                exp_avg_sq_list.append(state['exp_avg_sq'])

            combined_step = 0
            combined_exp_avg = None
            combined_exp_avg_sq = None

            if len(exp_avg_list) > 0:
                combined_step = step_list[0]
                combined_exp_avg = npu_combine_tensors(exp_avg_list)
                combined_exp_avg_sq = npu_combine_tensors(exp_avg_sq_list)

            combined_state = defaultdict(dict)
            combined_state['step'] = combined_step
            combined_state['exp_avg'] = combined_exp_avg
            combined_state['exp_avg_sq'] = combined_exp_avg_sq

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

        beta1, beta2 = group['b1'], group['b2']

        combined_group_params = self.combined_params_indexed_by_group[group_index]
        combined_group_grads = self.combined_grads_indexed_by_group[group_index]
        combined_group_param_states = self.combined_param_states_indexed_by_group[group_index]

        # loop for dtypes
        for combined_param, combined_grad, combined_param_state in zip(
            combined_group_params,
            combined_group_grads,
            combined_group_param_states):

            if combined_param is None or combined_grad is None:
                continue

            exp_avg, exp_avg_sq = combined_param_state['exp_avg'], combined_param_state['exp_avg_sq']

            if self.max_grad_norm > 0:
                _clip_grad_norm_(combined_param, combined_grad, self.max_grad_norm)

            exp_avg.mul_(beta1).add_(1 - beta1, combined_grad)
            exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, combined_grad, combined_grad)
            update = exp_avg / (exp_avg_sq.sqrt() + group['e'])

            if group['weight_decay'] > 0.0:
                update += group['weight_decay'] * combined_param.data

            if group['t_total'] != -1:
                schedule_fct = SCHEDULES[group['schedule']]
                lr_scheduled = group['lr'] * schedule_fct(
                    combined_param_state['step'] / group['t_total'], group['warmup'])
            else:
                lr_scheduled = group['lr']

            update_with_lr = lr_scheduled * update
            combined_param.data.add_(-update_with_lr)
            combined_param_state['step'] += 1
