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

import torch

from torch._tensor_str import _Formatter as SrcFormatter
from torch._tensor_str import PRINT_OPTS, _tensor_str_with_formatter, _add_suffixes, get_summarized_data
from torch.overrides import has_torch_function_unary, handle_torch_function

import torch_npu


class _Formatter(SrcFormatter):
    def __init__(self, tensor):
        self.floating_dtype = tensor.dtype.is_floating_point
        self.int_mode = True
        self.sci_mode = False
        self.max_width = 1
        super(_Formatter, self).__init__(tensor)


def _tensor_str(self, indent):
    if self.numel() == 0:
        return '[]'

    if self.has_names():
        # There are two main codepaths (possibly more) that tensor printing goes through:
        # - tensor data can fit comfortably on screen
        # - tensor data needs to be summarized
        # Some of the codepaths don't fully support named tensors, so we send in
        # an unnamed tensor to the formatting code as a workaround.
        self = self.rename(None)

    # step 1:
    # Put 'to-cpu' here is to avoid the long compile time of 'ConcatD','Pack' on npu.
    # Previous version put this operation in _Formatter class.
    device = self.device
    is_npu = self.is_npu
    if is_npu:
        self = self.cpu()

    summarize = self.numel() > PRINT_OPTS.threshold
    if self.dtype is torch.float16 or self.dtype is torch.bfloat16:
        self = self.float()

    if self.dtype.is_complex:
        real_formatter = _Formatter(get_summarized_data(self.real) if summarize else self.real)
        imag_formatter = _Formatter(get_summarized_data(self.imag) if summarize else self.imag)
        rst = _tensor_str_with_formatter(self, indent, summarize, real_formatter, imag_formatter)
    else:
        formatter = _Formatter(get_summarized_data(self) if summarize else self)
        rst = _tensor_str_with_formatter(self, indent, summarize, formatter)

    # step 2:
    # When above operations finished, we need to do 'to-npu' with self for following operations.
    if is_npu:
        self = self.to(device)
    return rst


def _str_intern(inp):
    prefix = 'tensor('
    indent = len(prefix)
    suffixes = []
    self, tangent = torch.autograd.forward_ad.unpack_dual(inp)

    if self.device.type == 'meta':
        suffixes.append('device=\'' + str(self.device.type) + '\'')
    elif self.device.type != torch._C._get_default_device()\
            or (self.device.type == torch_npu.npu.native_device and torch.npu.current_device() != self.device.index):
        suffixes.append('device=\'' + str(torch_npu.npu.npu_device) + ':' + str(torch.npu.current_device()) + '\'')

    _default_complex_dtype = torch.cdouble if torch.get_default_dtype() == torch.double else torch.cfloat
    has_default_dtype = self.dtype in (torch.get_default_dtype(), _default_complex_dtype, torch.int64, torch.bool)
    if self.is_sparse:
        suffixes.append('size=' + str(tuple(self.shape)))
        suffixes.append('nnz=' + str(self._nnz()))
        if not has_default_dtype:
            suffixes.append('dtype=' + str(self.dtype))
        indices_prefix = 'indices=tensor('
        indices = self._indices().detach()
        indices_str = _tensor_str(indices, indent + len(indices_prefix))
        if indices.numel() == 0:
            indices_str += ', size=' + str(tuple(indices.shape))
        values_prefix = 'values=tensor('
        values = self._values().detach()
        values_str = _tensor_str(values, indent + len(values_prefix))
        if values.numel() == 0:
            values_str += ', size=' + str(tuple(values.shape))
        tensor_str = indices_prefix + indices_str + '),\n' + ' ' * indent + values_prefix + values_str + ')'
    elif self.is_quantized:
        suffixes.append('size=' + str(tuple(self.shape)))
        if not has_default_dtype:
            suffixes.append('dtype=' + str(self.dtype))
        suffixes.append('quantization_scheme=' + str(self.qscheme()))
        if self.qscheme() == torch.per_tensor_affine or self.qscheme() == torch.per_tensor_symmetric:
            suffixes.append('scale=' + str(self.q_scale()))
            suffixes.append('zero_point=' + str(self.q_zero_point()))
        elif self.qscheme() == torch.per_channel_affine or self.qscheme() == torch.per_channel_symmetric \
                or self.qscheme() == torch.per_channel_affine_float_qparams:
            suffixes.append('scale=' + str(self.q_per_channel_scales()))
            suffixes.append('zero_point=' + str(self.q_per_channel_zero_points()))
            suffixes.append('axis=' + str(self.q_per_channel_axis()))
        tensor_str = _tensor_str(self.dequantize(), indent)
    else:
        if self.is_meta:
            suffixes.append('size=' + str(tuple(self.shape)))
            if self.dtype != torch.get_default_dtype():
                suffixes.append('dtype=' + str(self.dtype))
            tensor_str = '...'
        else:
            if self.numel() == 0 and not self.is_sparse:
                if self.dim() != 1:
                    suffixes.append('size=' + str(tuple(self.shape)))

                if self.dtype != torch.get_default_dtype():
                    suffixes.append('dtype=' + str(self.dtype))
                tensor_str = '[]'
            else:
                if not has_default_dtype:
                    suffixes.append('dtype=' + str(self.dtype))

                if self.layout != torch.strided:
                    tensor_str = _tensor_str(self.to_dense(), indent)
                else:
                    tensor_str = _tensor_str(self, indent)

    if self.layout != torch.strided:
        suffixes.append('layout=' + str(self.layout))

    if inp.grad_fn is not None:
        name = type(inp.grad_fn).__name__
        if name == 'CppFunction':
            name = inp.grad_fn.name().rsplit('::', 1)[-1]
        suffixes.append('grad_fn=<{}>'.format(name))
    elif inp.requires_grad:
        suffixes.append('requires_grad=True')

    if self.has_names():
        suffixes.append('names={}'.format(self.names))

    if tangent is not None:
        suffixes.append('tangent={}'.format(tangent))

    return _add_suffixes(prefix + tensor_str, suffixes, indent, force_newline=self.is_sparse)


def _str(self):
    with torch.no_grad():
        return _str_intern(self)


class Tensor(torch.Tensor):
    def __deepcopy__(self, memo):
        if has_torch_function_unary(self):
            return handle_torch_function(Tensor.__deepcopy__, (self,), self, memo)
        if not self.is_leaf:
            raise RuntimeError("Only Tensors created explicitly by the user "
                               "(graph leaves) support the deepcopy protocol at the moment")
        if id(self) in memo:
            return memo[id(self)]
        with torch.no_grad():
            if self.is_sparse or self.device.type == 'xla':
                new_tensor = self.clone()
            elif self.device.type == 'npu':
                new_tensor = self.clone().detach().requires_grad_(self.requires_grad)
            else:
                new_storage = self.storage().__deepcopy__(memo)
                if self.is_quantized:
                    # quantizer_params can be different type based on torch attribute
                    quantizer_params: Union[Tuple[torch.qscheme, float, int], Tuple[torch.qscheme, Tensor, Tensor, int]]
                    if self.qscheme() == torch.per_tensor_affine:
                        quantizer_params = self.qscheme(), self.q_scale(), self.q_zero_point()
                    elif self.qscheme() in (torch.per_channel_affine, torch.per_channel_affine_float_qparams):
                        quantizer_params = self.qscheme(), \
                            self.q_per_channel_scales(), \
                            self.q_per_channel_zero_points(), \
                            self.q_per_channel_axis()
                    else:
                        raise RuntimeError(f"Unsupported qscheme {self.qscheme()} in deepcopy")
                    new_tensor = torch._utils._rebuild_qtensor(
                        new_storage,
                        self.storage_offset(),
                        self.size(),
                        self.stride(),
                        quantizer_params,
                        self.requires_grad,
                        self._backward_hooks)
                else:
                    new_tensor = self.new()
                    new_tensor.set_(new_storage, self.storage_offset(), self.size(), self.stride())
                    new_tensor.requires_grad = self.requires_grad
            if self.grad is not None:
                new_tensor.grad = self.grad.__deepcopy__(memo)
            memo[id(self)] = new_tensor
            return new_tensor

    def __repr__(self):
        if has_torch_function_unary(self):
            return handle_torch_function(Tensor.__repr__, (self,), self)
        return _str(self)

    def share_memory_(self):
        r"""Moves the underlying storage to shared memory.

        This is a no-op if the underlying storage is already in shared memory
        and for CUDA tensors. Tensors in shared memory cannot be resized.
        """
        if has_torch_function_unary(self):
            return handle_torch_function(Tensor.share_memory_, (self,), self)

        if self.device.type == 'npu':
            self.storage()
        else:
            self.storage().share_memory_()
        return self


def add_str_methods():
    torch._tensor_str._str = _str
    torch.Tensor.__deepcopy__ = Tensor.__deepcopy__
    torch.Tensor.__repr__ = Tensor.__repr__
    torch.Tensor.share_memory_ = Tensor.share_memory_
