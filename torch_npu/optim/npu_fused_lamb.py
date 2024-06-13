from collections import defaultdict

import torch
from torch_npu.utils import npu_combine_tensors
from torch_npu.utils._error_code import ErrCode, pta_error
from .npu_fused_optim_base import NpuFusedOptimizerBase

__all__ = ["NpuFusedLamb"]


class NpuFusedLamb(NpuFusedOptimizerBase):
    r"""Implements NpuFusedLamb algorithm.

    It has been proposed in `Large Batch Optimization for Deep Learning: Training BERT in 76 minutes`_.

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional, default=1e-3): learning rate
        betas (Tuple[float, float], optional, default=(0.9, 0.999)): coefficients used for computing
            running averages of gradient and its square
        eps (float, optional, default=1e-8): term added to the denominator to improve
            numerical stability
        weight_decay (float, optional, default=0): weight decay (L2 penalty)
        adam (bool, optional): always use trust ratio = 1, which turns this into
            Adam. Useful for comparison purposes.
        use_global_grad_norm(bool, optional, default=False): use global grad norm
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-6,
                 weight_decay=0, adam=False, use_global_grad_norm=False):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr) + pta_error(ErrCode.VALUE))
        if betas[0] < 0.0 or betas[0] >= 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]) + pta_error(ErrCode.VALUE))
        if betas[1] < 0.0 or betas[1] >= 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]) + pta_error(ErrCode.VALUE))
        if eps < 0.0:
            raise ValueError("Invalid epsilon value: {}".format(eps) + pta_error(ErrCode.VALUE))
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay)
        self.adam = adam
        self.use_global_grad_norm = use_global_grad_norm
        self.global_grad_norm = torch.Tensor([1]).to('npu')
        self.middle_vars_are_combined_by_group = False
        super(NpuFusedLamb, self).__init__(params, defaults)

    def _init_param_state(self, p):
        state = self.state[p]
        # State initialization
        if len(state) == 0:
            state['step'] = 0
            # Exponential moving average of gradient values
            state['exp_avg'] = torch.zeros_like(p)
            # Exponential moving average of squared gradient values
            state['exp_avg_sq'] = torch.zeros_like(p)
        else:
            exp_avg_tmp = torch.zeros_like(p)
            exp_avg_tmp.copy_(state['exp_avg'])
            state['exp_avg'] = exp_avg_tmp

            exp_avg_sq_tmp = torch.zeros_like(p)
            exp_avg_sq_tmp.copy_(state['exp_avg_sq'])
            state['exp_avg_sq'] = exp_avg_sq_tmp

    def _combine_middle_vars(self, group_index):
        group_params_list = self.params_lists_indexed_by_group[group_index]

        self.trust_ratio_lists_indexed_by_group[group_index] = []
        self.param_pow_lists_indexed_by_group[group_index] = []
        self.adam_step_pow_lists_indexed_by_group[group_index] = []

        self.combined_trust_ratios_indexed_by_group[group_index] = []
        self.combined_param_pows_indexed_by_group[group_index] = []
        self.combined_adam_step_pows_indexed_by_group[group_index] = []

        for params in group_params_list:
            trust_ratio_list = []
            param_pow_list = []
            adam_step_pow_list = []

            for p in params:
                trust_ratio_list.append(torch.zeros_like(p))
                param_pow_list.append(torch.zeros_like(p))
                adam_step_pow_list.append(torch.zeros_like(p))

            combined_trust_ratio = npu_combine_tensors(trust_ratio_list)
            combined_param_pow = npu_combine_tensors(param_pow_list)
            combined_adam_step_pow = npu_combine_tensors(adam_step_pow_list)

            self.trust_ratio_lists_indexed_by_group[group_index].append(trust_ratio_list)
            self.param_pow_lists_indexed_by_group[group_index].append(param_pow_list)
            self.adam_step_pow_lists_indexed_by_group[group_index].append(adam_step_pow_list)

            self.combined_trust_ratios_indexed_by_group[group_index].append(combined_trust_ratio)
            self.combined_param_pows_indexed_by_group[group_index].append(combined_param_pow)
            self.combined_adam_step_pows_indexed_by_group[group_index].append(combined_adam_step_pow)

    def _combine_middle_vars_by_group(self):
        if self.middle_vars_are_combined_by_group:
            return

        self.trust_ratio_lists_indexed_by_group = []
        self.param_pow_lists_indexed_by_group = []
        self.adam_step_pow_lists_indexed_by_group = []

        self.combined_trust_ratios_indexed_by_group = []
        self.combined_param_pows_indexed_by_group = []
        self.combined_adam_step_pows_indexed_by_group = []

        for _ in self.param_groups:
            self.trust_ratio_lists_indexed_by_group.append([])
            self.param_pow_lists_indexed_by_group.append([])
            self.adam_step_pow_lists_indexed_by_group.append([])

            self.combined_trust_ratios_indexed_by_group.append([])
            self.combined_param_pows_indexed_by_group.append([])
            self.combined_adam_step_pows_indexed_by_group.append([])

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

            for p in params:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError('NpuFusedLamb does not support sparse gradients, '
                                       'please consider SparseAdam instead.' + pta_error(ErrCode.NOT_SUPPORT))
                
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

    def _get_global_grad_norm(self):
        global_norm = 0
        combined_grads = self.get_optimizer_combined_grads()
        combined_grad_masks = self.get_optimizer_combined_grad_masks()
        for combined_grad, combined_grad_mask in zip(combined_grads, combined_grad_masks):
            if combined_grad is not None:
                global_norm += combined_grad.pow(2).mul_(combined_grad_mask).sum()
        global_norm.sqrt_()
        return global_norm

    def _group_step(self, group_index):
        group = self.param_groups[group_index]
        for p in group['params']:
            if p.grad is None:
                continue
            state_p = self.state[p]
            state_p['step'] += 1
        beta1, beta2 = group['betas']
        step_size = group['lr']

        combined_group_params = self.combined_params_indexed_by_group[group_index]
        combined_group_grads = self.combined_grads_indexed_by_group[group_index]
        combined_group_param_states = self.combined_param_states_indexed_by_group[group_index]
        trust_ratio_lists = self.trust_ratio_lists_indexed_by_group[group_index]
        param_pow_lists = self.param_pow_lists_indexed_by_group[group_index]
        adam_step_pow_lists = self.adam_step_pow_lists_indexed_by_group[group_index]
        combined_trust_ratios = self.combined_trust_ratios_indexed_by_group[group_index]
        combined_param_pows = self.combined_param_pows_indexed_by_group[group_index]
        combined_adam_step_pows = self.combined_adam_step_pows_indexed_by_group[group_index]

        for combined_param, combined_grad, combined_param_state, \
            trust_ratio_list, param_pow_list, adam_step_pow_list, \
            combined_trust_ratio, combined_param_pow, \
            combined_adam_step_pow in zip(combined_group_params,
                                          combined_group_grads,
                                          combined_group_param_states,
                                          trust_ratio_lists,
                                          param_pow_lists,
                                          adam_step_pow_lists,
                                          combined_trust_ratios,
                                          combined_param_pows,
                                          combined_adam_step_pows):
            if combined_param is None or combined_grad is None:
                continue

            if self.global_grad_norm.item() > 1:
                combined_grad = combined_grad / self.global_grad_norm

            exp_avg, exp_avg_sq = combined_param_state['exp_avg'], combined_param_state['exp_avg_sq']
            combined_param_state['step'] += 1

            # Decay the first and second moment running average coefficient
            exp_avg.mul_(beta1).add_(combined_grad, alpha=1 - beta1)
            exp_avg_sq.mul_(beta2).addcmul_(combined_grad, combined_grad, value=1 - beta2)

            adam_step = exp_avg / exp_avg_sq.sqrt().add(group['eps'])
            if group['weight_decay'] != 0:
                adam_step.add_(combined_param, alpha=group['weight_decay'])

            if self.adam:
                combined_trust_ratio.fill_(1)
            else:
                combined_param_pow.copy_(combined_param.pow(2))
                combined_adam_step_pow.copy_(adam_step.pow(2))

                for param_pow, adam_step_pow, trust_ratio in zip(param_pow_list, 
                                                                 adam_step_pow_list, 
                                                                 trust_ratio_list):
                    weight_norm = param_pow.sum().sqrt().clamp(0, 10)
                    adam_norm = adam_step_pow.sum().sqrt()
                    if weight_norm == 0 or adam_norm == 0:
                        trust_ratio.fill_(1)
                    else:
                        trust_ratio.fill_(weight_norm / adam_norm)

            combined_param.addcmul_(adam_step, combined_trust_ratio, value=-step_size)

    @torch.no_grad()
    def step(self, closure=None):
        if not self.is_params_grads_combined:
            self._maybe_init_combined_params_and_grads()
        
        if not self.is_states_combined:
            self._maybe_init_combined_states()

        self._combine_middle_vars_by_group()

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        if self.use_global_grad_norm:
            self.global_grad_norm = self._get_global_grad_norm()
        for i, _ in enumerate(self.param_groups):
            self._group_step(i)

        return loss
