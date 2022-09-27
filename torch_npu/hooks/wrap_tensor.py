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

from .module import HOOKModule


WrapTensorOps = [
    '__add__', '__and__', '__bool__', '__div__', '__ge__', '__gt__', '__iadd__', '__iand__',
    '__idiv__', '__ifloordiv__', '__ilshift__', '__imod__', '__imul__', '__ior__', '__irshift__',
    '__isub__', '__ixor__', '__lshift__', '__matmul__', '__mod__', '__mul__', '__nonzero__',
    '__or__', '__radd__', '__rmul__', '__rshift__', '__sub__', '__truediv__', '__xor__',
    'abs', 'abs_', 'absolute', 'absolute_', 'acos', 'acos_', 'acosh', 'acosh_', 'add', 'add_',
    'addbmm', 'addbmm_', 'addcdiv', 'addcdiv_', 'addcmul', 'addcmul_', 'addmm', 'addmm_',
    'addmv', 'addmv_', 'addr', 'addr_', 'align_as', 'align_to', 'all', 'allclose', 'amax',
    'amin', 'angle', 'any', 'arccos', 'arccos_', 'arccosh', 'arccosh_', 'arcsin', 'arcsin_',
    'arcsinh', 'arcsinh_', 'arctan', 'arctan_', 'arctanh', 'arctanh_', 'argmax', 'argmin',
    'argsort', 'asin', 'asin_', 'asinh', 'asinh_', 'atan', 'atan2', 'atan2_', 'atan_', 'atanh',
    'atanh_', 'baddbmm', 'baddbmm_', 'bernoulli', 'bernoulli_', 'bincount', 'bitwise_and',
    'bitwise_and_', 'bitwise_not', 'bitwise_not_', 'bitwise_or', 'bitwise_or_', 'bitwise_xor',
    'bitwise_xor_', 'bmm', 'broadcast_to', 'cauchy_', 'ceil', 'ceil_', 'cholesky', 'chunk',
    'clamp', 'cholesky_solve', 'cholesky_inverse', 'clamp_', 'clamp_max', 'clamp_max_', 'clip',
    'clamp_min', 'clamp_min_', 'clip_', 'copy_', 'copysign', 'copysign_', 'cos', 'cos_', 'cosh',
    'cosh_', 'count_nonzero', 'cummax', 'cummin', 'cumprod', 'cumprod_', 'cumsum', 'cumsum_',
    'deg2rad', 'deg2rad_', 'det', 'diag', 'diag_embed', 'diagflat', 'diagonal', 'diff', 'dist',
    'digamma', 'digamma_', 'div', 'div_', 'divide', 'divide_', 'dot', 'eig', 'eq', 'eq_', 'erf',
    'equal', 'erf_', 'erfc', 'erfc_', 'erfinv', 'erfinv_', 'exp', 'exp2', 'exp2_', 'expm1',
    'exp_', 'expm1_', 'exponential_', 'fill_', 'fix', 'fill_diagonal_', 'fix_', 'flip', 'fliplr',
    'flatten', 'flipud', 'float_power', 'float_power_', 'floor', 'floor_', 'floor_divide',
    'floor_divide_', 'fmax', 'fmin', 'fmod', 'fmod_', 'frac', 'frac_', 'gather', 'gcd', 'gcd_',
    'ge', 'ge_', 'geometric_', 'geqrf', 'ger', 'get_device', 'greater', 'greater_', 'gt', 'gt_',
    'greater_equal', 'greater_equal_', 'hardshrink', 'heaviside', 'heaviside_', 'histc', 'hypot',
    'hypot_', 'igamma', 'igamma_', 'igammac', 'igammac_', 'index_add', 'index_add_', 'inverse',
    'index_copy', 'index_copy_', 'index_fill', 'index_fill_', 'index_put', 'index_put_', 'inner',
    'index_select', 'isclose', 'isfinite', 'isinf', 'isnan', 'isneginf', 'isposinf', 'isreal',
    'kron', 'kthvalue', 'lcm', 'lcm_', 'ldexp', 'ldexp_', 'le', 'le_', 'lerp', 'lerp_', 'where',
    'less', 'less_', 'less_equal', 'less_equal_', 'lgamma', 'lgamma_', 'log', 'log10', 'log10_',
    'log1p', 'log1p_', 'log2', 'log2_', 'log_', 'log_normal_', 'log_softmax', 'logcumsumexp',
    'logdet', 'logaddexp', 'logaddexp2', 'logical_and', 'logical_and_', 'logical_not', 'logit',
    'logical_not_', 'logical_or', 'logical_or_', 'logical_xor', 'logical_xor_', 'logit_',
    'logsumexp', 'lstsq', 'lt', 'lt_', 'lu_solve', 'map2_', 'map_', 'masked_fill', 'matmul',
    'masked_fill_', 'masked_scatter', 'masked_scatter_', 'masked_select', 'matrix_exp', 'max',
    'maximum', 'mean', 'matrix_power', 'median', 'min', 'minimum', 'mm', 'mode', 'msort', 'mul',
    'mul_', 'multinomial', 'multiply', 'multiply_', 'mv', 'mvlgamma', 'mvlgamma_', 'nansum',
    'narrow', 'narrow_copy', 'ne', 'ne_', 'neg', 'neg_', 'negative', 'negative_', 'nonzero',
    'normal_', 'not_equal', 'not_equal_', 'permute', 'pinverse', 'polygamma', 'pow', 'pow_',
    'polygamma_', 'prelu', 'prod', 'put_', 'rad2deg', 'rad2deg_', 'random_', 'ravel', 'real',
    'reciprocal', 'reciprocal_', 'relu', 'relu_', 'remainder', 'repeat_interleave', 'reshape',
    'remainder_', 'renorm', 'renorm_', 'repeat', 'reshape_as', 'resize_', 'resize_as_', 'roll',
    'rot90', 'round', 'round_', 'rsqrt', 'rsqrt_', 'scatter', 'scatter_', 'scatter_add', 'set_',
    'scatter_add_', 'select', 'sgn', 'sgn_', 'sigmoid', 'sigmoid_', 'sign', 'sign_', 'signbit',
    'sin', 'sin_', 'sinc', 'sinc_', 'sinh', 'sinh_', 'slogdet', 'smm', 'softmax', 'solve',
    'sort', 'split_with_sizes', 'sqrt', 'sqrt_', 'square', 'square_', 'squeeze', 'squeeze_',
    'sspaddmm', 'std', 'sub', 'sub_', 'sum', 'sum_to_size', 'svd', 'symeig', 't', 't_', 'take',
    'tan', 'tan_', 'tanh', 'tanh_', 'tensor_split', 'tile', 'topk', 'transpose', 'transpose_',
    'triangular_solve', 'tril', 'tril_', 'triu', 'true_divide', 'triu_', 'true_divide_',
    'trunc', 'trunc_', 'type_as', 'unbind', 'unflatten', 'unfold', 'unsafe_chunk', 'unsqueeze',
    'unsafe_split', 'unsafe_split_with_sizes', 'var', 'vdot', 'unsqueeze_', 'view_as', 'xlogy',
    'xlogy_'
]


def get_tensor_ops():
    global WrapTensorOps
    _tensor_ops = dir(torch._C._TensorBase)
    assert set(WrapTensorOps) <= set(_tensor_ops)
    return WrapTensorOps


class HOOKTensor(object):
    pass


class TensorOPTemplate(HOOKModule):
    
    def __init__(self, op_name):
        self.op_name_ = op_name
        self.prefix_op_name_ = "Tensor_" + str(op_name) + "_"
        super().__init__()

    def forward(self, *args, **kwargs):
        return getattr(torch._C._TensorBase, str(self.op_name_))(*args, **kwargs)


def wrap_tensor_op(op_name):
    def tensor_op_template(*args, **kwargs):
        return TensorOPTemplate(op_name)(*args, **kwargs)
    return tensor_op_template


def wrap_tensor_ops_and_bind():
    _tensor_ops = get_tensor_ops()
    for op_name in _tensor_ops:
        setattr(HOOKTensor, "wrap_" + str(op_name), wrap_tensor_op(op_name))
