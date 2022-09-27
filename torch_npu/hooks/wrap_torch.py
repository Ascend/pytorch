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


WrapTorchOps = [
    '_adaptive_avg_pool2d', '_add_relu', '_add_relu_', '_aminmax',
    '_batch_norm_impl_index', '_convolution', 'abs', 'abs_', 'absolute', 'acos',
    'acos_', 'acosh', 'acosh_', 'adaptive_avg_pool1d', 'adaptive_max_pool1d', 'add',
    'addbmm', 'addcdiv', 'addcmul', 'addmm', 'addmv', 'addmv_', 'addr', 'amax',
    'affine_grid_generator', 'align_tensors', 'all', 'alpha_dropout', 'amin',
    'alpha_dropout_', 'angle', 'any', 'arange', 'arccos', 'arccos_', 'arccosh',
    'arccosh_', 'arcsin', 'arcsin_', 'arcsinh', 'arcsinh_', 'arctan', 'arctan_',
    'arctanh', 'arctanh_', 'argmax', 'argmin', 'argsort', 'asin', 'asin_', 'asinh',
    'asinh_', 'atan', 'atan2', 'atan_', 'atanh', 'atanh_', 'atleast_1d', 'atleast_2d',
    'atleast_3d', 'avg_pool1d', 'baddbmm', 'bartlett_window', 'batch_norm',
    'batch_norm_backward_elemt', 'batch_norm_backward_reduce', 'batch_norm_elemt',
    'batch_norm_gather_stats', 'batch_norm_gather_stats_with_counts', 'bernoulli',
    'batch_norm_stats', 'batch_norm_update_stats', 'bilinear', 'bincount', 'binomial',
    'binary_cross_entropy_with_logits', 'bitwise_and', 'bitwise_not', 'bitwise_or',
    'bitwise_xor', 'blackman_window', 'block_diag', 'bmm', 'broadcast_tensors',
    'broadcast_to', 'cartesian_prod', 'cat', 'cdist', 'ceil', 'ceil_', 'celu',
    'celu_', 'chain_matmul', 'channel_shuffle', 'cholesky', 'cholesky_inverse',
    'cholesky_solve', 'choose_qparams_optimized', 'chunk', 'clamp', 'clamp_',
    'clamp_max', 'clamp_max_', 'clamp_min', 'clamp_min_', 'clip', 'clip_', 'clone',
    'column_stack', 'combinations', 'constant_pad_nd', 'conv1d', 'conv2d', 'conv3d',
    'conv_tbc', 'conv_transpose1d', 'conv_transpose2d', 'conv_transpose3d', 'cos',
    'convolution', 'copysign', 'cos_', 'cosh', 'cosh_', 'cosine_embedding_loss',
    'cosine_similarity', 'count_nonzero', 'cross', 'ctc_loss', 'cummax', 'cummin',
    'cumprod', 'cumsum', 'deg2rad', 'deg2rad_', 'det', 'diag', 'diag_embed', 'diff',
    'diagflat', 'diagonal', 'digamma', 'dist', 'div', 'divide', 'dot', 'dropout',
    'dropout_', 'dsmm', 'dstack', 'eig', 'einsum', 'embedding', 'embedding_bag',
    'embedding_renorm_', 'eq', 'equal', 'erf', 'erf_', 'erfc', 'erfc_', 'erfinv',
    'exp', 'exp2', 'exp2_', 'exp_', 'expm1', 'expm1_', 'eye', 'feature_dropout',
    'feature_alpha_dropout', 'feature_alpha_dropout_', 'feature_dropout_', 'fix',
    'fill_', 'fix_', 'flatten', 'flip', 'fliplr', 'flipud', 'float_power', 'floor',
    'floor_', 'floor_divide', 'fmax', 'fmin', 'fmod', 'frac', 'frac_', 'full',
    'frobenius_norm', 'full_like', 'gather', 'gcd', 'gcd_', 'ge', 'geqrf', 'ger',
    'greater', 'greater_equal', 'grid_sampler', 'grid_sampler_2d', 'group_norm',
    'grid_sampler_3d', 'gru', 'gru_cell', 'gt', 'hamming_window', 'hann_window',
    'hardshrink', 'heaviside', 'hinge_embedding_loss', 'histc', 'hsmm', 'hspmm',
    'hstack', 'hypot', 'igamma', 'igammac', 'index_add', 'index_copy', 'inner',
    'index_fill', 'index_put', 'index_put_', 'index_select', 'instance_norm',
    'isclose', 'isfinite', 'isinf', 'isnan', 'isneginf', 'isposinf', 'istft',
    'kaiser_window', 'kl_div', 'kron', 'kthvalue', 'layer_norm', 'lcm', 'lcm_',
    'ldexp', 'ldexp_', 'le', 'lerp', 'less', 'less_equal', 'lgamma', 'linspace',
    'log', 'log10', 'log10_', 'log1p', 'log1p_', 'log2', 'log2_', 'log_softmax',
    'log_', 'logaddexp', 'logaddexp2', 'logcumsumexp', 'logdet', 'logical_and',
    'logical_not', 'logical_or', 'logical_xor', 'logit', 'logit_', 'logspace',
    'logsumexp', 'lstm', 'lstm_cell', 'lstsq', 'lt', 'lu_solve', 'masked_fill',
    'margin_ranking_loss', 'masked_scatter', 'masked_select', 'matrix_exp',
    'matmul', 'matrix_power', 'matrix_rank', 'max', 'max_pool1d', 'max_pool2d',
    'max_pool1d_with_indices', 'max_pool3d', 'maximum', 'mean', 'median', 'min',
    'meshgrid', 'minimum', 'mm', 'mode', 'moveaxis', 'movedim', 'msort', 'mul',
    'multinomial', 'multiply', 'mv', 'mvlgamma', 'nan_to_num', 'nan_to_num_',
    'nanmedian', 'nansum', 'narrow', 'native_batch_norm', 'native_group_norm',
    'narrow_copy', 'native_layer_norm', 'native_norm', 'ne', 'neg', 'negative',
    'neg_', 'negative_', 'nextafter', 'nonzero', 'norm', 'norm_except_dim',
    'normal', 'not_equal', 'nuclear_norm', 'ones_like', 'pairwise_distance',
    'ones', 'pdist', 'pinverse', 'pixel_shuffle', 'pixel_unshuffle', 'poisson',
    'poisson_nll_loss', 'polar', 'polygamma', 'pow', 'prelu', 'prod', 'rad2deg',
    'promote_types', 'rad2deg_', 'range', 'ravel', 'real', 'reciprocal', 'relu',
    'reciprocal_', 'relu_', 'remainder', 'renorm', 'repeat_interleave', 'reshape',
    'resize_as_', 'roll', 'rot90', 'round', 'round_', 'rrelu', 'rrelu_', 'rsqrt',
    'row_stack', 'rsqrt_', 'rsub', 'saddmm', 'scalar_tensor', 'scatter', 'select',
    'scatter_add', 'searchsorted', 'selu', 'selu_', 'sgn', 'sigmoid', 'sigmoid_',
    'sign', 'signbit', 'sin', 'sin_', 'sinc', 'sinc_', 'sinh', 'sinh_', 'slogdet',
    'smm', 'softmax', 'solve', 'sort', 'sparse_coo_tensor', 'split', 'square',
    'split_with_sizes', 'spmm', 'sqrt', 'sqrt_', 'square_', 'squeeze', 'sspaddmm',
    'stack', 'std', 'std_mean', 'stft', 'sub', 'subtract', 'sum', 'svd', 'swapaxes',
    'swapdims', 'symeig', 't', 'take', 'tan', 'tan_', 'tanh', 'tanh_', 'tensordot',
    'tensor_split', 'threshold', 'threshold_', 'tile', 'topk', 'transpose', 'trapz',
    'triangular_solve', 'tril', 'tril_indices', 'triplet_margin_loss', 'triu',
    'triu_indices', 'true_divide', 'trunc', 'trunc_', 'unique_consecutive', 'xlogy',
    'unbind', 'unique_dim', 'unsafe_chunk', 'unsafe_split', 'vander', 'var', 'vdot',
    'unsafe_split_with_sizes', 'unsqueeze', 'var_mean', 'vstack', 'where', 'xlogy_'
]


def get_torch_ops():
    global WrapTorchOps
    _torch_ops = dir(torch._C._VariableFunctionsClass)
    assert set(WrapTorchOps) <= set(_torch_ops)
    return WrapTorchOps


class HOOKTorchOP(object):
    pass


class TorchOPTemplate(HOOKModule):
    
    def __init__(self, op_name):
        self.op_name_ = op_name
        self.prefix_op_name_ = "Torch_" + str(op_name) + "_"
        super().__init__()

    def forward(self, *args, **kwargs):
        return getattr(torch._C._VariableFunctionsClass, str(self.op_name_))(*args, **kwargs)


def wrap_torch_op(op_name):
    def torch_op_template(*args, **kwargs):
        return TorchOPTemplate(op_name)(*args, **kwargs)
    return torch_op_template


def wrap_torch_ops_and_bind():
    _torch_ops = get_torch_ops()
    for op_name in _torch_ops:
        setattr(HOOKTorchOP, "wrap_" + op_name, wrap_torch_op(op_name))
