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

from statistics import mode
import warnings
import logging
import torch
import torch_npu


def npu(self, device=None):
    r"""Moves all model parameters and buffers to the npu.

    This also makes associated parameters and buffers different objects. So
    it should be called before constructing optimizer if the module will
    live on npu while being optimized.

    Arguments:
        device (int, optional): if specified, all parameters will be
            copied to that device

    Returns:
        Module: self
    """
    if device is None:
        device = torch.device("npu")
    if torch_npu.npu.is_available():
        # Ref [cast weight in single op mode]
        is_graph_mode = torch_npu.npu.is_graph_mode()
        if is_graph_mode:
            torch_npu.npu.disable_graph_mode()
        with torch.no_grad():
            self.cast_weight(device)
        if is_graph_mode:
            torch_npu.npu.enable_graph_mode()
    return self._apply(lambda t: t.npu(device))


def to(self, *args, **kwargs):
    device, dtype, non_blocking, convert_to_format = torch._C._nn._parse_to(*args, **kwargs)

    if dtype is not None:
        if not (dtype.is_floating_point or dtype.is_complex):
            raise TypeError('nn.Module.to only accepts floating point or complex '
                            'dtypes, but got desired dtype={}'.format(dtype))
        if dtype.is_complex:
            warnings.warn(
                "Complex modules are a new feature under active development whose design may change, "
                "and some modules might not work as expected when using complex tensors as parameters or buffers. "
                "Please file an issue at https://github.com/pytorch/pytorch/issues/new?template=bug-report.md "
                "if a complex module does not work as expected.")
    if torch_npu.npu.is_available():
        with torch.no_grad():
            # Ref [cast weight in single op mode]
            is_graph_mode = torch_npu.npu.is_graph_mode()
            if is_graph_mode:
                torch_npu.npu.disable_graph_mode()
            with torch.no_grad():
                self.cast_weight(device)
            if is_graph_mode:
                torch_npu.npu.enable_graph_mode();

    def convert(t):
        if convert_to_format is not None and t.dim() == 4:
            return t.to(device, dtype if t.is_floating_point() or t.is_complex() else None,
                        non_blocking, memory_format=convert_to_format)
        return t.to(device, dtype if t.is_floating_point() or t.is_complex() else None, non_blocking)

    return self._apply(convert)


def cast_weight(self, device):

    def _format_cast(module, class_name):
        if issubclass(class_name, torch.nn.Linear):
            module.weight.data = module.weight.data.to(device)
            module.weight.data = torch_npu.npu_format_cast(module.weight.data, 29) # ACL_FORMAT_FRACTAL_NZ
        if issubclass(class_name, (torch.nn.BatchNorm3d, torch.nn.BatchNorm2d, torch.nn.BatchNorm1d)):
            if module.affine:
                module.weight.data = module.weight.data.to(device)
                module.weight.data = torch_npu.npu_format_cast(module.weight.data, 3)  # ACL_FORMAT_NC1HWC0
                module.bias.data = module.bias.data.to(device)
                module.bias.data = torch_npu.npu_format_cast(module.bias.data, 3)
            if module.track_running_stats:
                module.running_mean.data = module.running_mean.data.to(device)
                module.running_mean.data = torch_npu.npu_format_cast(module.running_mean.data, 3)
                module.running_var.data = module.running_var.data.to(device)
                module.running_var.data = torch_npu.npu_format_cast(module.running_var.data, 3)
        if issubclass(class_name, torch.nn.Conv2d):
            if module.groups > 1:
                return
            if hasattr(module, "weight") and module.weight is not None:
                module.weight.data = module.weight.data.to(device)
                module.weight.data = torch_npu.npu_format_cast(module.weight.data, 4)  # ACL_FORMAT_FRACTAL_Z
        if issubclass(class_name, torch.nn.Conv3d):
            module.weight.data = module.weight.data.to(device)
            module.weight.data = torch_npu.npu_format_cast(module.weight.data.half(), 33).float()  # ACL_FRACTAL_Z_3D
        if "MultiheadAttention" in str(class_name) and \
            hasattr(module,"q_proj_weight") and module.q_proj_weight is not None and \
            hasattr(module,"k_proj_weight") and module.k_proj_weight is not None and \
            hasattr(module,"v_proj_weight") and module.v_proj_weight is not None:
            module.q_proj_weight.data = module.q_proj_weight.data.to(device)
            module.q_proj_weight.data = torch_npu.npu_format_cast(module.q_proj_weight.data, 29)
            module.k_proj_weight.data = module.k_proj_weight.data.to(device)
            module.k_proj_weight.data = torch_npu.npu_format_cast(module.k_proj_weight.data, 29)
            module.v_proj_weight.data = module.v_proj_weight.data.to(device)
            module.v_proj_weight.data = torch_npu.npu_format_cast(module.v_proj_weight.data, 29)

    if device is None or "npu" not in str(device):
        return

    current_class = self.__class__
    _format_cast(self, current_class)

    if not self.children:
        return 

    for sub_module in self.children():
        if isinstance(sub_module, torch.nn.Module):
            sub_module.cast_weight(device)


def layernorm_forward(self, input: torch.Tensor) -> torch.Tensor:
    if self.training or (not input.is_npu):
        return torch.nn.functional.layer_norm(
            input, self.normalized_shape, self.weight, self.bias, self.eps)
    else:
        return torch_npu.npu_layer_norm_eval(input, self.normalized_shape, self.weight, self.bias, self.eps)


def ddp_forward(self, *inputs, **kwargs):
    if self.ddp_uneven_inputs_config.ddp_join_enabled:
        ones = torch.ones(
            1, device=self.device
        )
        work = torch_npu.distributed.all_reduce(ones, group=self.process_group, async_op=True)
        self.reducer._set_forward_pass_work_handle(
            work, self.ddp_uneven_inputs_config.ddp_join_divide_by_initial_world_size
        )

    # Calling _rebuild_buckets before forward compuation,
    # It may allocate new buckets before deallocating old buckets
    # inside _rebuild_buckets. To save peak memory usage,
    # call _rebuild_buckets before the peak memory usage increases
    # during forward computation.
    # This should be called only once during whole training period.
    if self.reducer._rebuild_buckets():
        logging.info("Reducer buckets have been rebuilt in this iteration.")

    if self.require_forward_param_sync:
        self._sync_params()

    if self.ddp_uneven_inputs_config.ddp_join_enabled:
        # Notify joined ranks whether they should sync in backwards pass or not.
        self._check_global_requires_backward_grad_sync(is_joined_rank=False)

    if self.device_ids and self.device_type != "npu":
        if len(self.device_ids) == 1:
            inputs, kwargs = self.to_kwargs(inputs, kwargs, self.device_ids[0])
            output = self.module(*inputs[0], **kwargs[0])
        else:
            inputs, kwargs = self.scatter(inputs, kwargs, self.device_ids)
            outputs = self.parallel_apply(self._module_copies[:len(inputs)], inputs, kwargs)
            output = self.gather(outputs, self.output_device)
    else:
        output = self.module(*inputs, **kwargs)

    if torch.is_grad_enabled() and self.require_backward_grad_sync:
        self.require_forward_param_sync = True
        # We'll return the output object verbatim since it is a freeform
        # object. We need to find any tensors in this object, though,
        # because we need to figure out which parameters were used during
        # this forward pass, to ensure we short circuit reduction for any
        # unused parameters. Only if `find_unused_parameters` is set.
        if self.find_unused_parameters:
            self.reducer.prepare_for_backward(list(torch.nn.parallel.distributed._find_tensors(output)))
        else:
            self.reducer.prepare_for_backward([])
    else:
        self.require_forward_param_sync = False

    return output


def apply_module_patch():
    torch.nn.Module.npu = npu
    torch.nn.Module.to = to
    torch.nn.Module.cast_weight = cast_weight
    torch.nn.LayerNorm.forward = layernorm_forward
    torch.nn.parallel.DistributedDataParallel.forward = ddp_forward
