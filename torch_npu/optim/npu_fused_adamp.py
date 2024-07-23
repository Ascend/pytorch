import math
from collections import defaultdict

import torch
from torch_npu.utils import npu_combine_tensors
from torch_npu.utils._error_code import ErrCode, pta_error
from .npu_fused_optim_base import NpuFusedOptimizerBase

__all__ = ["NpuFusedAdamP"]


class NpuFusedAdamP(NpuFusedOptimizerBase):
    """Implements AdamP algorithm.

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional, default: 1e-3): learning rate
        betas (Tuple[float, float], optional, default: (0.9, 0.999)): coefficients used
            for computing running averages of gradient and its square
        eps (float, optional, default: 1e-8): term added to the denominator to improve
            numerical stability
        weight_decay (float, optional, default: 0): weight decay coefficient
        delta (float, optional, default: 0.1): threshold of cosine similarity
        wd_ratio (float, optional, default: 0.1): weight decay ratio for dynamic tuning
        nesterov (bool, optional, default: False): enables Nesterov momentum
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, delta=0.1, wd_ratio=0.1, nesterov=False):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
                        delta=delta, wd_ratio=wd_ratio, nesterov=nesterov)
        self.middle_vars_are_combined_by_group = False
        super(NpuFusedAdamP, self).__init__(params, defaults)

    def _channel_view(self, x):
        return x.view(x.size(0), -1)

    def _layer_view(self, x):
        return x.view(1, -1)

    def _cosine_similarity(self, x, y, eps, view_func):
        x = view_func(x)
        y = view_func(y)

        x_norm = x.norm(dim=1).add_(eps)
        y_norm = y.norm(dim=1).add_(eps)
        dot = (x * y).sum(dim=1)

        return dot.abs() / x_norm / y_norm

    def _projection(self, p, grad, perturb, delta, wd_ratio, eps):
        wd = 1
        expand_size = [-1] + [1] * (len(p.shape) - 1)
        for view_func in [self._channel_view, self._layer_view]:

            cosine_sim = self._cosine_similarity(grad, p, eps, view_func)

            if cosine_sim.max() < delta / math.sqrt(view_func(p).size(1)):
                p_n = p / view_func(p).norm(dim=1).view(expand_size).add_(eps)
                perturb -= p_n * view_func(p_n * perturb).sum(dim=1).view(expand_size)
                wd = wd_ratio

                return perturb, wd

        return perturb, wd

    def _init_param_state(self, p):
        state = self.state[p]
        # State initialization
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

    def _combine_middle_vars(self, group_index):
        group_params_list = self.params_lists_indexed_by_group[group_index]

        self.perturb_lists_indexed_by_group[group_index] = []
        self.combined_perturb_lists_indexed_by_group[group_index] = []

        self.wd_ratio_lists_indexed_by_group[group_index] = []
        self.combined_wd_ratio_lists_indexed_by_group[group_index] = []

        for params in group_params_list:
            perturb_list = []
            wd_ratio_list = []

            for p in params:
                perturb_list.append(torch.zeros_like(p))
                wd_ratio_list.append(torch.zeros_like(p))

            combined_perturb = npu_combine_tensors(perturb_list)
            combined_wd_ratio = npu_combine_tensors(wd_ratio_list)

            self.perturb_lists_indexed_by_group[group_index].append(perturb_list)
            self.combined_perturb_lists_indexed_by_group[group_index].append(combined_perturb)

            self.wd_ratio_lists_indexed_by_group[group_index].append(wd_ratio_list)
            self.combined_wd_ratio_lists_indexed_by_group[group_index].append(combined_wd_ratio)

    def _combine_middle_vars_by_group(self):
        if self.middle_vars_are_combined_by_group:
            return

        self.perturb_lists_indexed_by_group = []
        self.combined_perturb_lists_indexed_by_group = []

        self.wd_ratio_lists_indexed_by_group = []
        self.combined_wd_ratio_lists_indexed_by_group = []

        for _ in self.param_groups:
            self.perturb_lists_indexed_by_group.append([])
            self.combined_perturb_lists_indexed_by_group.append([])

            self.wd_ratio_lists_indexed_by_group.append([])
            self.combined_wd_ratio_lists_indexed_by_group.append([])

        for i, _ in enumerate(self.param_groups):
            self._combine_middle_vars(i)
        self.middle_vars_are_combined_by_group = True

    def _combine_group_param_states(self, group_index):
        group_params_list = self.params_lists_indexed_by_group[group_index]

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
                    raise RuntimeError('NpuFusedAdamP does not support sparse gradients, '
                                       'please consider SparseAdam instead' + pta_error(ErrCode.NOT_SUPPORT))

                self._init_param_state(p)
                state = self.state[p]
                step_list.append(state['step'])
                exp_avg_list.append(state['exp_avg'])
                exp_avg_sq_list.append(state['exp_avg_sq'])

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

            state = self.state[p]
            state['step'] += 1

        beta1, beta2 = group['betas']
        nesterov = group['nesterov']

        combined_group_params = self.combined_params_indexed_by_group[group_index]
        combined_group_grads = self.combined_grads_indexed_by_group[group_index]
        combined_group_param_states = self.combined_param_states_indexed_by_group[group_index]

        params_lists_indexed = self.params_lists_indexed_by_group[group_index]
        perturb_lists_indexed = self.perturb_lists_indexed_by_group[group_index]
        combined_perturb_lists_indexed = self.combined_perturb_lists_indexed_by_group[group_index]
        wd_ratio_lists_indexed = self.wd_ratio_lists_indexed_by_group[group_index]
        combined_wd_ratio_lists_indexed = self.combined_wd_ratio_lists_indexed_by_group[group_index]

        for combined_param, combined_grad, combined_param_state, params_list, perturb_list, \
            combined_perturb, wd_ratio_list, combined_wd_ratio in zip(
                combined_group_params, combined_group_grads,
                combined_group_param_states, params_lists_indexed,
                perturb_lists_indexed, combined_perturb_lists_indexed,
                wd_ratio_lists_indexed, combined_wd_ratio_lists_indexed
        ):

            if combined_param is None or combined_grad is None:
                continue

            exp_avg, exp_avg_sq = combined_param_state['exp_avg'], combined_param_state['exp_avg_sq']

            combined_param_state['step'] += 1
            bias_correction1 = 1 - beta1 ** combined_param_state['step']
            bias_correction2 = 1 - beta2 ** combined_param_state['step']

            # Decay the first and second moment running average coefficient
            exp_avg.mul_(beta1).add_(combined_grad, alpha=1 - beta1)
            exp_avg_sq.mul_(beta2).addcmul_(combined_grad, combined_grad, value=1 - beta2)
            denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])
            step_size = group['lr'] / bias_correction1

            if nesterov:
                perturb = (beta1 * exp_avg + (1 - beta1) * combined_grad) / denom
            else:
                perturb = exp_avg / denom

            combined_perturb.copy_(perturb)

            # Projection
            for param, perturb_in_list, wd_ratio_in_list in zip(params_list, perturb_list, wd_ratio_list):
                wd_ratio = 1

                if len(param.shape) > 1:
                    perturb_i, wd_ratio = self._projection(param, param.grad.data, perturb_in_list, group['delta'],
                                                           group['wd_ratio'], group['eps'])
                    perturb_in_list.copy_(perturb_i)

                if group['weight_decay'] > 0:
                    wd_ratio_in_list.fill_(wd_ratio)

            # Weight decay
            if group['weight_decay'] > 0:
                combined_param.mul_(1 - group['lr'] * group['weight_decay'] * combined_wd_ratio)

            # Step
            combined_param.add_(-step_size, combined_perturb)

    @torch.no_grad()
    def step(self, closure=None):
        if not self.is_params_grads_combined:
            self._maybe_init_combined_params_and_grads()
        
        if not self.is_states_combined:
            self._maybe_init_combined_states()

        # combine middle vars
        self._combine_middle_vars_by_group()

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for i, _ in enumerate(self.param_groups):
            self._group_step(i)

        return loss
