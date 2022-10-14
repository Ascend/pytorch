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

from torch.nn.functional import (
    conv1d, conv2d, conv3d, conv_transpose1d, conv_transpose2d, conv_transpose3d,
    conv_tbc, avg_pool1d, avg_pool2d, avg_pool3d, fractional_max_pool2d_with_indices,
    fractional_max_pool2d, fractional_max_pool3d_with_indices, fractional_max_pool3d,
    max_pool1d_with_indices, max_pool1d, max_pool2d_with_indices, max_pool2d,
    max_pool3d_with_indices, max_pool3d, max_unpool1d, max_unpool2d, max_unpool3d,
    lp_pool2d, lp_pool1d, adaptive_max_pool1d_with_indices, adaptive_max_pool1d,
    adaptive_max_pool2d_with_indices, adaptive_max_pool2d, adaptive_max_pool3d_with_indices,
    adaptive_max_pool3d, adaptive_avg_pool1d, adaptive_avg_pool2d, adaptive_avg_pool3d,
    dropout, alpha_dropout, dropout2d, dropout3d, feature_alpha_dropout, threshold, threshold_,
    relu, relu_, glu, hardtanh, hardtanh_, relu6, elu, elu_, selu, selu_, celu, celu_,
    leaky_relu, leaky_relu_, prelu, rrelu, rrelu_, logsigmoid, gelu, hardshrink, tanhshrink,
    softsign, softplus, softmin, softmax, gumbel_softmax, log_softmax, softshrink, tanh,
    sigmoid, hardsigmoid, linear, bilinear, silu, hardswish, embedding, embedding_bag,
    batch_norm, instance_norm, layer_norm, group_norm, local_response_norm, ctc_loss,
    nll_loss, poisson_nll_loss, gaussian_nll_loss, kl_div, cross_entropy, binary_cross_entropy,
    binary_cross_entropy_with_logits, smooth_l1_loss, l1_loss, mse_loss, margin_ranking_loss,
    hinge_embedding_loss, multilabel_margin_loss, soft_margin_loss, multilabel_soft_margin_loss,
    cosine_embedding_loss, multi_margin_loss, pixel_shuffle, pixel_unshuffle, channel_shuffle,
    upsample, interpolate, upsample_nearest, upsample_bilinear, grid_sample, affine_grid,
    pad, pairwise_distance, pdist, cosine_similarity, one_hot, triplet_margin_loss,
    triplet_margin_with_distance_loss, normalize, unfold, fold, multi_head_attention_forward
)

from .module import HOOKModule


WrapFunctionalOps = [
    'conv1d', 'conv2d', 'conv3d', 'conv_transpose1d', 'conv_transpose2d', 'conv_transpose3d',
    'conv_tbc', 'avg_pool1d', 'avg_pool2d', 'avg_pool3d', 'fractional_max_pool2d_with_indices',
    'fractional_max_pool2d', 'fractional_max_pool3d_with_indices', 'fractional_max_pool3d',
    'max_pool1d_with_indices', 'max_pool1d', 'max_pool2d_with_indices', 'max_pool2d',
    'max_pool3d_with_indices', 'max_pool3d', 'max_unpool1d', 'max_unpool2d', 'max_unpool3d',
    'lp_pool2d', 'lp_pool1d', 'adaptive_max_pool1d_with_indices', 'adaptive_max_pool1d',
    'adaptive_max_pool2d_with_indices', 'adaptive_max_pool2d', 'adaptive_max_pool3d_with_indices',
    'adaptive_max_pool3d', 'adaptive_avg_pool1d', 'adaptive_avg_pool2d', 'adaptive_avg_pool3d',
    'dropout', 'alpha_dropout', 'dropout2d', 'dropout3d', 'feature_alpha_dropout', 'threshold', 'threshold_',
    'relu', 'relu_', 'glu', 'hardtanh', 'hardtanh_', 'relu6', 'elu', 'elu_', 'selu', 'selu_', 'celu', 'celu_',
    'leaky_relu', 'leaky_relu_', 'prelu', 'rrelu', 'rrelu_', 'logsigmoid', 'gelu', 'hardshrink', 'tanhshrink',
    'softsign', 'softplus', 'softmin', 'softmax', 'gumbel_softmax', 'log_softmax', 'softshrink', 'tanh',
    'sigmoid', 'hardsigmoid', 'linear', 'bilinear', 'silu', 'hardswish', 'embedding', 'embedding_bag',
    'batch_norm', 'instance_norm', 'layer_norm', 'group_norm', 'local_response_norm', 'ctc_loss',
    'nll_loss', 'poisson_nll_loss', 'gaussian_nll_loss', 'kl_div', 'cross_entropy', 'binary_cross_entropy',
    'binary_cross_entropy_with_logits', 'smooth_l1_loss', 'l1_loss', 'mse_loss', 'margin_ranking_loss',
    'hinge_embedding_loss', 'multilabel_margin_loss', 'soft_margin_loss', 'multilabel_soft_margin_loss',
    'cosine_embedding_loss', 'multi_margin_loss', 'pixel_shuffle', 'pixel_unshuffle', 'channel_shuffle',
    'upsample', 'interpolate', 'upsample_nearest', 'upsample_bilinear', 'grid_sample', 'affine_grid',
    'pad', 'pairwise_distance', 'pdist', 'cosine_similarity', 'one_hot', 'triplet_margin_loss',
    'triplet_margin_with_distance_loss', 'normalize', 'unfold', 'fold', 'multi_head_attention_forward'
]


def get_functional_ops():
    global WrapFunctionalOps
    _all_functional_ops = dir(torch.nn.functional)
    assert set(WrapFunctionalOps) <= set(_all_functional_ops)
    return WrapFunctionalOps


class HOOKFunctionalOP(object):
    pass


class FunctionalOPTemplate(HOOKModule):
    
    def __init__(self, op_name):
        self.op_name_ = op_name
        self.prefix_op_name_ = "Functional_" + str(op_name) + "_"
        super().__init__()

    def forward(self, *args, **kwargs):
        return eval(self.op_name_)(*args, **kwargs)


def wrap_functional_op(op_name):
    def functional_op_template(*args, **kwargs):
        return FunctionalOPTemplate(op_name)(*args, **kwargs)
    return functional_op_template


def wrap_functional_ops_and_bind():
    _functional_ops = get_functional_ops()
    for op_name in _functional_ops:
        setattr(HOOKFunctionalOP, "wrap_" + op_name, wrap_functional_op(op_name))
