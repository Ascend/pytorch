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

import warnings
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
        with torch.no_grad():
            self.cast_weight(device)
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
            self.cast_weight(device)

    def convert(t):
        if convert_to_format is not None and t.dim() == 4:
            return t.to(device, dtype if t.is_floating_point() or t.is_complex() else None,
                        non_blocking, memory_format=convert_to_format)
        return t.to(device, dtype if t.is_floating_point() or t.is_complex() else None, non_blocking)

    return self._apply(convert)


def cast_weight(self, device):
    if device is None:
        return

    if "npu" not in str(device):
        return

    current_class = self.__class__
    if issubclass(current_class, torch.nn.Linear):
        self.weight.data = self.weight.data.to(device)
        self.weight.data = torch_npu.npu_format_cast(self.weight.data, 29) #ACL_FORMAT_FRACTAL_NZ
    elif issubclass(current_class, (torch.nn.BatchNorm2d, torch.nn.BatchNorm1d)):
        if self.affine == True:
            self.weight.data = self.weight.data.to(device)
            self.weight.data = torch_npu.npu_format_cast(self.weight.data, 3)  #ACL_FORMAT_NC1HWC0
            self.bias.data = self.bias.data.to(device)
            self.bias.data = torch_npu.npu_format_cast(self.bias.data, 3)
        self.running_mean.data = self.running_mean.data.to(device)
        self.running_mean.data = torch_npu.npu_format_cast(self.running_mean.data, 3)
        self.running_var.data = self.running_var.data.to(device)
        self.running_var.data = torch_npu.npu_format_cast(self.running_var.data, 3)
    elif issubclass(current_class, torch.nn.Conv2d):
        if (self.in_channels == self.groups and self.groups > 1 and self.weight.size(0) % self.in_channels == 0):
            return
        self.weight.data = self.weight.data.to(device)
        self.weight.data = torch_npu.npu_format_cast(self.weight.data, 4)  #ACL_FORMAT_FRACTAL_Z
    elif issubclass(current_class, torch.nn.Conv3d):
        self.weight.data = self.weight.data.to(device)
        self.weight.data = torch_npu.npu_format_cast(self.weight.data.half(), 33).float()  #ACL_FRACTAL_Z_3D
    elif ("MultiheadAttention" in str(current_class)):
        if hasattr(self,"q_proj_weight") and self.q_proj_weight is not None and \
            hasattr(self,"k_proj_weight") and self.k_proj_weight is not None and \
            hasattr(self,"v_proj_weight") and self.v_proj_weight is not None:
            self.q_proj_weight.data = self.q_proj_weight.data.to(device)
            self.q_proj_weight.data = torch_npu.npu_format_cast(self.q_proj_weight.data, 29)
            self.k_proj_weight.data = self.k_proj_weight.data.to(device)
            self.k_proj_weight.data = torch_npu.npu_format_cast(self.k_proj_weight.data, 29)
            self.v_proj_weight.data = self.v_proj_weight.data.to(device)
            self.v_proj_weight.data = torch_npu.npu_format_cast(self.v_proj_weight.data, 29)

    if self.children() is not None:
        for sub_module in self.children():
            if isinstance(sub_module, torch.nn.Module):
                sub_module.cast_weight(device)


def apply_module_patch():
    torch.nn.Module.npu = npu
    torch.nn.Module.to = to
    torch.nn.Module.cast_weight = cast_weight


class LayerNorm(torch.nn.LayerNorm):

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.training:
            return torch.nn.functional.layer_norm(
                input, self.normalized_shape, self.weight, self.bias, self.eps)
        else:
            return torch_npu.npu_layer_norm_eval(input, self.normalized_shape, self.weight, self.bias, self.eps)
