# Copyright (c) 2020 Huawei Technologies Co., Ltd
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

from contextlib import contextmanager
import torch
import torch_npu

class ReplayGraph(torch_npu._C._NPUReplayGraphBase):
    def __new__(cls, **kwargs):
        return super(ReplayGraph, cls).__new__(cls, **kwargs)

    def __generate_replay_graph(self, inputs: list, assigned_outputs: list,
                              returnable_outputs: list, retain_inner_outputs: bool=False):
        super(ReplayGraph, self).generate_replay_graph(inputs, assigned_outputs,
                                                       returnable_outputs, retain_inner_outputs)

    def __replay(self, inputs: list, assigned_outputs: list) -> tuple:
        return super(ReplayGraph, self).replay(inputs, assigned_outputs)

    def __get_inner_outputs(self, inputs: list) -> tuple:
        return super(ReplayGraph, self).get_inner_outputs(inputs)

    def __is_replay_cache_hit(self, inputs: list) -> bool:
        return super(ReplayGraph, self).is_replay_cache_hit(inputs)


class WrapModule(object):
    def __init__(self, module, fwd_func, requires_grad=False, warm_up_step=3, verbose=False):
        self.module = module
        self.fwd_func = fwd_func
        self.cur_step = 0
        self.fwd_graph = None
        self.bwd_graph = None
        self.param_grad = []
        self.verbose = verbose
        self.requires_grad = requires_grad
        self.warm_up_step = warm_up_step if self.requires_grad else 1

    def __wrap_forward(self, *args, **kwargs):
        for arg in args:
            if not isinstance(arg, torch.Tensor):
                raise TypeError("All args should be tensor in replay graph mode")

        for arg in kwargs.values():
            if not isinstance(arg, torch.Tensor):
                raise TypeError("All args should be tensor in replay graph mode")

        replay_cache = False
        if self.fwd_graph is not None:
            origin_inputs = [arg for arg in args if isinstance(arg, torch.Tensor)]
            replay_cache = self.fwd_graph._ReplayGraph__is_replay_cache_hit(origin_inputs)
            origin_inputs = []

        if self.cur_step < self.warm_up_step or not replay_cache:
            self.cur_step = self.cur_step + 1
            for p in self.module.parameters():
                p.grad = torch.zeros_like(p)
            shallow_args = ()
            fwd_inputs = []
            for arg in args:
                if isinstance(arg, torch.Tensor):
                    shallow_input = torch.empty_like(arg)
                    if self.requires_grad:
                        shallow_input.requires_grad_(True)
                    tu = (shallow_input,)
                    fwd_inputs.append(shallow_input)
                    shallow_args = shallow_args + tu
                else:
                    tu = (arg,)
                    shallow_args = shallow_args + tu

            with enable_replay_graph_mode(self.verbose):
                shallow_fwd_output = self.fwd_func(*shallow_args, **kwargs)
                if self.requires_grad and not isinstance(shallow_fwd_output, torch.Tensor):
                    raise TypeError("shallow_fwd_output shoule be one tensor.")

                # shallow_fwd_output must be a tensor list
                if isinstance(shallow_fwd_output, torch.Tensor):
                    shallow_fwd_output = [shallow_fwd_output]
                if isinstance(shallow_fwd_output, tuple):
                    shallow_fwd_output = list(shallow_fwd_output)

                fwd_graph_info = [fwd_inputs, self.module.parameters(), self.module.buffers()]
                fwd_graph_inputs = []
                for fwd_info in fwd_graph_info:
                    fwd_graph_inputs.extend(fwd_info)
                fwd_assigned_outputs = []
                self.fwd_graph = generate_replay_graph(replay_graph=self.fwd_graph,
                                                       inputs=fwd_graph_inputs,
                                                       assigned_outputs=fwd_assigned_outputs,
                                                       returnable_outputs=shallow_fwd_output,
                                                       retain_inner_outputs=True)

                # Inner_outputs is only required in the requires_grad case.
                if self.requires_grad:
                    saved_var = self.fwd_graph._ReplayGraph__get_inner_outputs(inputs=fwd_inputs)
                    grad_input = torch.empty_like(shallow_fwd_output)
                    torch.autograd.backward(shallow_fwd_output, grad_input)

                    self.param_grad = [p.grad for p in self.module.parameters() if p.grad is not None]
                    grad_output = [fwd_input.grad for fwd_input in fwd_inputs]
                    bwd_graph_info = [fwd_graph_inputs, saved_var, [grad_input], self.param_grad, [shallow_fwd_output]]
                    bwd_graph_inputs = []
                    for bwd_info in bwd_graph_info:
                        bwd_graph_inputs.extend(bwd_info)
                    self.bwd_graph = generate_replay_graph(replay_graph=self.bwd_graph,
                                                           inputs=bwd_graph_inputs,
                                                           assigned_outputs=self.param_grad,
                                                           returnable_outputs=grad_output)

            saved_var, fwd_graph_inputs, bwd_graph_inputs, grad_input = [], [], [], []
            grad_output, shallow_fwd_output, fwd_inputs, shallow_input = [], [], [], []
            fwd_graph_info, bwd_graph_info = [], []


        class ReplayFunction(torch.autograd.Function):
            @staticmethod
            def forward(ctx, *args, **kwargs):
                fwd_inputs = [arg for arg in args if isinstance(arg, torch.Tensor)]
                fwd_inputs_full_info = [fwd_inputs, self.module.parameters(), self.module.buffers()]
                fwd_inputs_full = []
                for info in fwd_inputs_full_info:
                    fwd_inputs_full.extend(info)
                fwd_assigned_outputs = []

                fwd_output = self.fwd_graph._ReplayGraph__replay(inputs=fwd_inputs_full,
                                                                 assigned_outputs=fwd_assigned_outputs)

                if self.requires_grad:
                    save_var = self.fwd_graph._ReplayGraph__get_inner_outputs(inputs=fwd_inputs)
                    ctx.fwd_input = fwd_inputs
                    ctx.saved_var = save_var
                    if fwd_output is not None and fwd_output[0] is not None:
                        ctx.output = fwd_output[0]
                        fwd_output[0].requires_grad_(True)
                    else:
                        raise ValueError("Forward output has no value")
                return fwd_output

            @staticmethod
            @torch.autograd.function.once_differentiable
            def backward(ctx, *grad_outputs):
                if not self.requires_grad:
                    raise ValueError("Backward should be used when input tensor set requires_grad True.")

                need_init_grad = False
                for p in self.module.parameters():
                    if p.grad is None:
                        need_init_grad = True
                        break

                if need_init_grad:
                    self.param_grad = []
                    for p in self.module.parameters():
                        if p.grad is None:
                            p.grad = torch.zeros_like(p)
                        self.param_grad.append(p.grad)

                bwd_inputs_full_info = [ctx.fwd_input, self.module.parameters(), self.module.buffers(),
                                        ctx.saved_var, grad_outputs, self.param_grad, [ctx.output]]
                bwd_inputs_full = []
                for info in bwd_inputs_full_info:
                    bwd_inputs_full.extend(info)

                bwd_output = self.bwd_graph._ReplayGraph__replay(inputs=bwd_inputs_full,
                                                                 assigned_outputs=self.param_grad)
                if bwd_output is None:
                    raise ValueError("Backward output has no value")
                ctx.saved_var, ctx.output = [], []
                return bwd_output

        ret = ReplayFunction.apply(*args, **kwargs)
        if ret[0] is None:
            raise ValueError("ReplayFunction return has no value")
        return ret


def make_replay_graph(module: torch.nn.Module, requires_grad: bool=True, verbose_: bool=False) -> torch.nn.Module:
    wrap_module = WrapModule(module, module.forward, requires_grad=requires_grad, verbose=verbose_)
    module.forward = wrap_module._WrapModule__wrap_forward
    module.is_replay_graph = True
    return module


def generate_replay_graph(replay_graph: ReplayGraph, inputs: list, assigned_outputs: list,
                          returnable_outputs: list, retain_inner_outputs: bool=False) -> ReplayGraph:
    if replay_graph is None:
        replay_graph = ReplayGraph()
    replay_graph._ReplayGraph__generate_replay_graph(inputs, assigned_outputs, returnable_outputs, retain_inner_outputs)
    return replay_graph

@contextmanager
def enable_replay_graph_mode(verbose: bool=False):
    torch_npu._C._npu_enable_replay_graph_mode(verbose)
    yield 1
    torch_npu._C._npu_disable_replay_graph_mode()
