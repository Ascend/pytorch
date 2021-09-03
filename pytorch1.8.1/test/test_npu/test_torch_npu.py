# Copyright (c) 2020 Huawei Technologies Co., Ltd
# Copyright (c) 2019, Facebook CORPORATION. 
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


import sys
sys.path.append('./common/')
import io
import inspect
import math
import random
import re
import copy
import torch
import torch.cuda
import torch.backends.cuda
import tempfile
import unittest
import warnings
import types
import pickle
import textwrap
import torch.backends.quantized
import torch.testing._internal.data
import numpy as np
import contextlib
import torch.backends.quantized
import torch.testing._internal.data
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch.nn.utils.rnn as rnn_utils
import torch.nn.utils.prune as prune
from torch.utils.dlpack import from_dlpack, to_dlpack
from torch._six import inf, nan, string_classes, istuple
from itertools import product, combinations, combinations_with_replacement, permutations
from functools import reduce
from random import randrange
from common.util_test_new import create_common_tensor
from common.util_test_new import compare_res_new
from torch.nn.utils import clip_grad_norm_, clip_grad_value_
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from torch import multiprocessing as mp
from torch.testing._internal.common_methods_invocations import tri_tests_args, run_additional_tri_tests, \
    _compare_trilu_indices
from common.common_nn import *
from torch.nn import Parameter
from common.common_utils_new import TestCase, iter_indices, TEST_NUMPY, TEST_SCIPY, TEST_MKL, \
    TEST_LIBROSA, TEST_WITH_ROCM, run_tests, skipIfNoLapack, suppress_warnings, \
    IS_WINDOWS, PY3, NO_MULTIPROCESSING_SPAWN, do_test_dtypes, do_test_empty_full, \
    IS_SANDCASTLE, load_tests, slowTest, skipCUDANonDefaultStreamIf, skipCUDAMemoryLeakCheckIf, \
    BytesIOContext, skipIfRocm, freeze_rng_state, download_file, get_function_arglist, \
    repeat_test_for_types, ALL_TENSORTYPES, ALL_TENSORTYPES2, TemporaryFileName, TEST_WITH_UBSAN, IS_PPC
from multiprocessing.reduction import ForkingPickler
from common.common_device_type_new import instantiate_device_type_tests, \
    skipCPUIfNoLapack, skipCUDAIfNoMagma, skipCUDAIfRocm, skipCUDAIfNotRocm, onlyCUDA, onlyCPU, \
    dtypes, dtypesIfCUDA, deviceCountAtLeast, skipCUDAIf, precisionOverride, \
    PYTORCH_CUDA_MEMCHECK, largeCUDATensorTest, onlyOnCPUAndCUDA, skipCUDAIfNoCudnn, skipCUDAIfCudnnVersionLessThan
from itertools import repeat, product
from torch.autograd import gradcheck, Variable
from torch.autograd.gradcheck import gradgradcheck


torch.set_default_dtype(torch.float32)

_types = [
    torch.half, torch.float, torch.double,
    torch.int8, torch.short, torch.int, torch.long,
    torch.uint8
]

# _types2 adds bfloat16 type to  _types only on ROCm. Should eventually be unified
# with _types when bfloat16 bringup is complete on all platforms.
_types2 = _types + [torch.bfloat16] if TEST_WITH_ROCM else _types

_float_types = [torch.half, torch.float, torch.double]

_float_types_no_half = [torch.float, torch.double]

# _float_types2 adds bfloat16 type to _float_types only on ROCm. Should eventually be unified
# with _float_types when bfloat16 bringup is complete on all platforms
_float_types2 = _float_types + [torch.bfloat16] if TEST_WITH_ROCM else _float_types

_signed_types = [
    torch.half, torch.float, torch.double,
    torch.int8, torch.short, torch.int, torch.long
]

_signed_types_no_half = [
    torch.float, torch.double,
    torch.int8, torch.short, torch.int, torch.long
]

_unsigned_types = [torch.uint8]

# Helper values and functions for producing tensors and scalars to use in tensor op tests.
# Tensor dimension sizes (Small, Medium, Large, Giant)
_S = 5
_M = 50
_L = 1000
_G = 275000000

# Value to clamp divisors to since dividing by small numbers can be unstable
# on devices.
_div_min = 2**-8

LOWER = 0
UPPER = 2


# Returns floating or integral scalar corresponding to dtype
def _number(floating, integer, dtype):
    if dtype in [torch.half, torch.float, torch.double, torch.bfloat16]:
        return floating
    return integer


# Converts half/bfloat16 dtype to float when device is cpu
def _convert_t(dtype, device):
    if device == 'cpu' and dtype in {torch.half, torch.bfloat16}:
        return torch.float
    return dtype


# Returns a tensor of the requested shape, dtype, and device
# Requesting a half CPU tensor returns a float CPU tensor with
# values representable by a half.
# Initialization uses randint for non-float types and randn for float types.
def _make_tensor(shape, dtype, device, fill_ones=False):
    # Returns a tensor filled with ones
    if fill_ones:
        return torch.ones(*shape, dtype=_convert_t(dtype, device), device=device)

    # Returns a tensor with random integer values
    if dtype not in _float_types2:
        t = torch.randint(0, 10, shape, device=device)
        return t.to(_convert_t(dtype, device))

    # Populates the CPU tensor with floats representable as half/bfloat16
    if dtype == torch.half and device == 'cpu':
        return torch.randn(*shape, dtype=torch.float, device=device).half().float()
    if dtype == torch.bfloat16 and device == 'cpu':
        return torch.randn(*shape, dtype=torch.float, device=device).bfloat16().float()

    # Default: returns a tensor with random float values
    return torch.randn(shape, dtype=dtype, device=device).to(dtype=dtype)


def _small_0d(dtype, device):
    return _make_tensor((1,), dtype, device).squeeze()


def _small_2d(dtype, device, has_zeros=True, fill_ones=False, oneish=False):
    t = _make_tensor((_S, _S), dtype, device, fill_ones=fill_ones)
    if oneish:
        return t.clamp(min=_number(.99, 1, dtype), max=1.01)
    if not has_zeros:
        return t.clamp(min=(_number(_div_min, 1, dtype)))
    return t


def _small_3d(dtype, device, has_zeros=True, fill_ones=False, oneish=False):
    t = _make_tensor((_S, _S, _S), dtype, device, fill_ones=fill_ones)
    if oneish:
        return t.clamp(min=_number(.99, 1, dtype), max=1.01)
    if not has_zeros:
        return t.clamp(min=(_number(_div_min, 1, dtype)))
    return t


def _small_3d_ones(dtype, device):
    return _small_3d(dtype, device, fill_ones=True)


def _small_3d_unique(dtype, device):
    return (torch.randperm(_S * _S * _S,
            dtype=_convert_t(dtype, device), device=device) + 1).view(_S, _S, _S)


def _medium_1d(dtype, device):
    return _make_tensor((_M,), dtype, device)


def _medium_2d(dtype, device):
    return _make_tensor((_M, _M), dtype, device)


def _large_2d(dtype, device):
    t = _make_tensor((_L, _L), dtype, device)
    return t.normal_()


def _giant_1d(dtype, device):
    return _make_tensor((_G), dtype, device)


# Helper method that returns a function which takes dtype and device and
# instantiates tensors of the given shape.
# Useful for tensor op tests with custom shapes.
def _new_t(shape):
    def tmp(dtype, device):
        return _make_tensor(shape, dtype, device)
    return tmp


def _wrap_maybe_warns(regex):
    def decorator(fn):
        def inner(self, device, dtype):
            with self.maybeWarnsRegex(UserWarning, regex):
                fn(self, device, dtype)
        return inner
    return decorator


tensor_sum_tests = [
    ('sum', '', _small_2d, lambda t, d: [], 1e-2, 1e-2, 1e-5, _types2, False),
    ('sum', 'dim', _small_3d, lambda t, d: [1], 1e-2, 1e-2, 1e-5, _types2, False),
    ('sum', 'neg_dim', _small_3d, lambda t, d: [-1], 1e-2, 1e-5, 1e-5, _types, False),
]

tensor_sub_tests = [
    ('sub', '', _small_3d, lambda t, d: [_number(3.14, 3, t)], 1e-2),
    ('sub', 'tensor', _small_3d, lambda t, d: [_small_3d(t, d)], 1e-2),
]

tensor_sqrt_tests = [
    ('sqrt', '', _small_3d, lambda t, d: [], 1e-3, 1e-5, 1e-5, _float_types),
]

tensor_split_tests = [
    ('split', '', _small_3d, lambda t, d: [2], 1e-5, 1e-5, 1e-5, _types, False),
    ('split', 'dim', _small_3d, lambda t, d: [2, 1], 1e-5, 1e-5, 1e-5, _types, False),
    ('split', 'neg_dim', _small_3d, lambda t, d: [2, -3], 1e-5, 1e-5, 1e-5, _types, False),
]

tensor_sort_tests = [
    ('sort', '', _small_3d_unique, lambda t, d: [], 1e-5, 1e-5, 1e-5, _types, False),
    ('sort', 'dim', _small_3d_unique, lambda t, d: [1], 1e-5, 1e-5, 1e-5, _types, False),
    ('sort', 'neg_dim', _small_3d_unique, lambda t, d: [-1], 1e-5, 1e-5, 1e-5, _types, False),
    ('sort', 'dim_descending', _small_3d_unique, lambda t, d: [1, True], 1e-5, 1e-5, 1e-5, _types, False),
    ('sort', 'neg_dim_descending', _small_3d_unique, lambda t, d: [-1, True], 1e-5, 1e-5, 1e-5, _types, False),
]

tensor_sigmoid_tests = [
    ('sigmoid', '', _small_3d, lambda t, d: [], 1e-3, 1e-2, 1e-5, _float_types2),
]

tensor_rsqrt_tests = [
    ('rsqrt', '', lambda t, d: _small_3d(t, d) + 1, lambda t, d: [], 1e-2, 1e-5, 1e-4, _float_types_no_half),
]

tensor_add_tests = [
    ('add', '', _small_3d, lambda t, d: [_number(3.14, 3, t)], 1e-2),
    ('add', 'tensor', _small_3d, lambda t, d: [_small_3d(t, d)], 1e-2),
]

tensor_addmm_tests = [
    ('addmm', '', _medium_2d, lambda t, d: [_medium_2d(t, d), _medium_2d(t, d)],
        1e-1, 1e-1, 1e-4, _float_types2),
    ('addmm', 'scalar', _medium_2d,
        lambda t, d: [_number(0.4, 2, t), _medium_2d(t, d), _medium_2d(t, d)],
        1e-1, 1e-1, 1e-4, _float_types2, True,
        [_wrap_maybe_warns("This overload of addmm_? is deprecated")]),
    ('addmm', 'two_scalars', _medium_2d,
        lambda t, d: [_number(0.5, 3, t), _number(0.4, 2, t), _medium_2d(t, d), _medium_2d(t, d)],
        1e-1, 1e-1, 1e-4, _float_types2, True,
        [_wrap_maybe_warns("This overload of addmm_? is deprecated")]),
]

tensor_abs_tests = [
    ('abs', '', _small_3d, lambda t, d: []),
]

tensor_bmm_tests = [
    ('bmm', '', _small_3d, lambda t, d: [_small_3d(t, d)],
     1e-5, 1e-5, 1e-5, _float_types_no_half, False),
]
tensor_clamp_tests = [
    ('clamp', 'neg', _medium_2d, lambda t, d: [-1, 5], 1e-5, 1e-5, 1e-5, _signed_types),
    ('clamp', 'pos', _medium_2d, lambda t, d: [1, 5], 1e-5, 1e-5, 1e-5, _unsigned_types),
]

tensor_div_tests = [
    ('div', '', _small_3d, lambda t, d: [_number(3.14, 3, t)], 1e-1),
    ('div', 'tensor', _small_3d,
     lambda t, d: [_small_3d(t, d, has_zeros=False)], 1e-1),
]

tensor_equ_tests = [
    ('eq', '', _small_3d_ones, lambda t, d: [_small_3d(t, d)], 1e-5, 1e-5, 1e-5, _types2),
    ('eq', 'equal', _small_3d_ones, lambda t, d: [_small_3d_ones(t, d)], 1e-5, 1e-5, 1e-5, _types2),
]

tensor_exp_tests = [
    ('exp', '', _small_3d, lambda t, d: [], 1e-2, 1e-5, 1e-5, _float_types),
]

tensor_fill_tests = [
    ('fill_', '', _medium_2d, lambda t, d: [_number(3.14, 3, t)], 1e-3, 1e-5, 1e-5, _types, False),
]

tensor_floor_tests = [
    ('floor', '', _small_3d, lambda t, d: [], 1e-5, 1e-5, 1e-5, _float_types),
    ('floor_divide', '', _small_3d, lambda t, d: [_number(3.14, 3, t)], 1, 1e-5, 1e-5, _types),
    ('floor_divide', 'tensor', _small_3d,
        lambda t, d: [_small_3d(t, d, has_zeros=False)], 1, 1e-5, 1e-5, _types),
]

tensor_fmod_tests = [
    ('fmod', 'value', _small_3d, lambda t, d: [3], 1e-3),
    ('fmod', 'tensor', _small_3d, lambda t, d: [_small_3d(t, d, has_zeros=False)], 1e-3),
]

tensor_ge_tests = [
    ('ge', '', _medium_2d, lambda t, d: [_medium_2d(t, d)], 1e-5, 1e-5, 1e-5, _types2),
]

tensor_log_tests = [
    ('log', '', _small_3d, lambda t, d: [], 1e-2, 1e-1, 1e-5, _float_types2),
]

tensor_log2_tests = [
    ('log2', '', _small_3d, lambda t, d: [], 1e-2, 1e-1, 1e-5, _float_types2),
]

tensor_neg_tests = [
    ('neg', '', _small_3d, lambda t, d: [], 1e-5, 1e-5, 1e-5, _float_types2),
]

tensor_nonzero_tests = [
    ('nonzero', '', _small_3d, lambda t, d: [], 1e-5, 1e-5, 1e-5, _types, False),
]

tensor_gt_tests = [
    ('gt', '', _medium_2d, lambda t, d: [_medium_2d(t, d)], 1e-5, 1e-5, 1e-5, _types2),
]

tensor_le_tests = [
    ('le', '', _medium_2d, lambda t, d: [_medium_2d(t, d)], 1e-5, 1e-5, 1e-5, _types2),
]

tensor_lt_tests = [
    ('lt', '', _medium_2d, lambda t, d: [_medium_2d(t, d)], 1e-5, 1e-5, 1e-5, _types2),
]

tensor_max_tests = [
    ('max', '', _small_3d, lambda t, d: [], 1e-5, 1e-5, 1e-5, _types, False),
    ('max', 'dim', _small_3d_unique, lambda t, d: [1], 1e-5, 1e-5, 1e-5, _types, False),
    ('max', 'neg_dim', _small_3d_unique, lambda t, d: [-1], 1e-5, 1e-5, 1e-5, _types, False),
    ('max', 'elementwise', _medium_2d, lambda t, d: [_medium_2d(t, d)], 1e-5, 1e-5, 1e-5, _types, False),
]

tensor_min_tests = [
    ('min', '', _small_3d, lambda t, d: [], 1e-5, 1e-5, 1e-5, _types, False),
    ('min', 'dim', _small_3d_unique, lambda t, d: [1], 1e-5, 1e-5, 1e-5, _types, False),
    ('min', 'neg_dim', _small_3d_unique, lambda t, d: [-1], 1e-5, 1e-5, 1e-5, _types, False),
    ('min', 'elementwise', _medium_2d, lambda t, d: [_medium_2d(t, d)], 1e-5, 1e-5, 1e-5, _types, False),
]

tensor_ne_tests = [
    ('ne', '', _small_3d_ones, lambda t, d: [_small_3d(t, d)], 1e-5, 1e-5, 1e-5, _types2),
    ('ne', 'equal', _small_3d_ones, lambda t, d: [_small_3d_ones(t, d)], 1e-5, 1e-5, 1e-5, _types2),
]

tensor_pow_tests = [
    ('pow', '', _small_3d, lambda t, d: [_number(3.14, 3, t)], 1e-1, 1e-4, 1e-4, _float_types),
    ('pow', '1', _small_3d, lambda t, d: [_number(1., 1, t)], 1e-1),
    ('pow', '2', _small_3d, lambda t, d: [_number(2., 2, t)], 1e-1),
    ('pow', '3', _small_3d, lambda t, d: [_number(3., 3, t)], 1e-1),
    ('pow', '-1', _small_3d, lambda t, d: [_number(-1., -1, t)], 1e-1, 1e-4, 1e-4, _float_types),
    ('pow', '-2', _small_3d, lambda t, d: [_number(-2., -2, t)],
     1e-1, 1e-4, 1e-4, _float_types_no_half, False, [skipCUDAIfRocm]),
    ('pow', 'tensor', _small_3d, lambda t, d: [_small_3d(t, d).abs()],
     1e-1, 1e-4, 1e-4, _float_types),
]

tensor_mul_tests = [
    ('mul', '', _small_3d, lambda t, d: [_number(3.14, 3, t)], 1e-2),
    ('mul', 'tensor', _small_3d, lambda t, d: [_small_3d(t, d)], 1e-2),
    ('mul', 'scalar', _small_0d, lambda t, d: [_small_0d(torch.int32, d)], 1e-2),
]

tensor_prod_tests = [
    ('prod', '', lambda t, d: _small_2d(t, d, oneish=True), lambda t, d: [], 1e-2, 1e-1, 1e-5, _types2, False),
    ('prod', 'dim', _small_3d, lambda t, d: [1], 1e-3, 1e-1, 1e-5, _types2, False),
    ('prod', 'neg_dim', _small_3d, lambda t, d: [-1], 1e-3, 1e-1, 1e-5, _types2, False),
]

tensor_remainder_tests = [
    ('remainder', 'value', _small_3d, lambda t, d: [3], 1e-1, 1e-5, 1e-5, _signed_types),
    ('remainder', 'negative_value', _small_3d, lambda t, d: [-3], 1e-1, 1e-5, 1e-5, _signed_types),
    ('remainder', 'tensor', _small_3d,
     lambda t, d: [_small_3d(t, d, has_zeros=False)],
     1e-1, 1e-5, 1e-5, _signed_types),
    ('remainder', 'negative_tensor', _small_3d,
     lambda t, d: [0 - _small_3d(t, d, has_zeros=False)],
     1e-1, 1e-5, 1e-5, _signed_types),
]

tensor_topk_tests = [
    ('topk', 'dim_sort', _small_3d_unique, lambda t, d: [2, 1, False, True],
        1e-5, 1e-5, 1e-5, _types, False),
    ('topk', 'neg_dim_sort', _small_3d_unique, lambda t, d: [2, -1, False, True],
        1e-5, 1e-5, 1e-5, _types, False),
    ('topk', 'dim_desc_sort', _small_3d_unique, lambda t, d: [2, 1, True, True],
        1e-5, 1e-5, 1e-5, _types, False),
    ]

tensor_zero_tests = [
    ('zero_', '', _small_3d, lambda t, d: [], 1e-5, 1e-5, 1e-5, _types, False),
    ]


def _test_math(self, torchfn, mathfn, input=None, test_expand=False, rtol=None, atol=None):
    if input is None:
        input = []
        input.append(list(range(-5, 5)))
        input.append([0 for x in range(-5, 5)])
        input.append([x + 1e-6 for x in range(-5, 5)])
        # Some vectorized implementations don't support large ranges
        input.append([x + 1e10 for x in range(-5, 5)])
        input.append([x - 1e10 for x in range(-5, 5)])
        input.append(torch.randn(10).tolist())
        input.append((torch.randn(10) + 1e6).tolist())
        input.append([math.pi * (x / 2) for x in range(-5, 5)])

    def compare_reference(input, dtype):
        input = torch.tensor(input, dtype=dtype)
        res1 = torchfn(input.clone())
        res2 = input.clone().apply_(mathfn)
        torch.testing.assert_allclose(res1, res2, rtol=rtol, atol=atol)

    # compare against the reference math function
    compare_reference(input, torch.double)
    compare_reference(input, torch.float)

    def check_non_contiguous(shape, dtype):
        contig = torch.randn(shape, dtype=dtype)
        non_contig = torch.empty(shape + (2,), dtype=dtype)[..., 0]
        non_contig.copy_(contig)
        self.assertFalse(non_contig.is_contiguous())
        self.assertEqual(torchfn(contig), torchfn(non_contig), 'non-contiguous')

    # compare application against contiguous vs. non-contiguous
    check_non_contiguous((5, 7), torch.double)
    check_non_contiguous((1024,), torch.double)
    check_non_contiguous((5, 7), torch.float)
    check_non_contiguous((1024,), torch.float)

    def check_non_contiguous_index(dtype):
        contig = torch.randn((2, 2, 1, 2), dtype=dtype)
        non_contig = contig[:, 1, ...]
        contig = non_contig.clone()
        self.assertFalse(non_contig.is_contiguous())
        self.assertEqual(torchfn(contig), torchfn(non_contig), 'non-contiguous index')

    check_non_contiguous_index(torch.float)
    check_non_contiguous_index(torch.double)

    def check_non_contiguous_expand(shape, dtype):
        contig = torch.randn(shape, dtype=dtype)
        non_contig = contig.clone().expand(3, -1, -1)
        self.assertFalse(non_contig.is_contiguous())
        contig = torchfn(contig)
        non_contig = torchfn(non_contig)
        for i in range(3):
            self.assertEqual(contig, non_contig[i], 'non-contiguous expand[' + str(i) + ']')

    # Expand is not defined for in-place operations
    if test_expand:
        # The size 1 case is special as it leads to 0 stride and needs to persists
        check_non_contiguous_expand((1, 3), torch.double)
        check_non_contiguous_expand((1, 7), torch.double)
        check_non_contiguous_expand((5, 7), torch.float)

    # If size(dim) == 1, stride(dim) is not defined.
    # The code needs to be able to handle this
    def check_contiguous_size1(dtype):
        contig = torch.randn((5, 100), dtype=dtype)
        contig = contig[:1, :50]
        contig2 = torch.empty(contig.size(), dtype=dtype)
        contig2.copy_(contig)
        self.assertTrue(contig.is_contiguous())
        self.assertTrue(contig2.is_contiguous())
        self.assertEqual(torchfn(contig), torchfn(contig2), 'contiguous size1')

    check_contiguous_size1(torch.double)
    check_contiguous_size1(torch.float)

    def check_contiguous_size1_largedim(dtype):
        contig = torch.randn((5, 2, 3, 1, 4, 5, 3, 2, 1, 2, 3, 4), dtype=dtype)
        contig = contig[:1, :, :, :, :, :, :, :, :, :, :, :]
        contig2 = torch.empty(contig.size(), dtype=dtype)
        contig2.copy_(contig)
        self.assertTrue(contig.is_contiguous())
        self.assertTrue(contig2.is_contiguous())
        self.assertEqual(torchfn(contig), torchfn(contig2), 'contiguous size1')

    check_contiguous_size1_largedim(torch.double)
    check_contiguous_size1_largedim(torch.float)

    def check_large(dtype):
        input = torch.randn(1024, 512, dtype=dtype)
        actual = torchfn(input)
        expected = torch.stack([torchfn(slice) for slice in input])
        self.assertEqual(actual, expected, 'large')

    # compare large tensor vs. repeated small applications to expose
    # possible parallelism bugs.
    check_large(torch.double)
    check_large(torch.float)


# Creates and decorates a generic test and adds it to the class.
def generate_test_function(cls,
                           op_str,
                           subtest_str,
                           tensor_ctor,
                           arg_ctor,
                           half_precision,
                           bfloat16_precision,
                           float_precision,
                           dtype_list,
                           decorators):
    def fn(self, device, dtype):
        # Generates the CPU inputs
        # Note: CPU tensors are never torch.half
        cpu_tensor = tensor_ctor(dtype, 'cpu')
        cpu_args = arg_ctor(dtype, 'cpu')
        # Converts CPU tensors to device tensors
        device_tensor = cpu_tensor.to(dtype=dtype, device=device)

        device_args = [arg.to(device=device) if torch.is_tensor(arg) else arg for arg in cpu_args]

        # Converts float device tensors to half/bfloat16 when the dtype is half/bfloat16
        # Note: CPU half tensors don't support many operations.
        if dtype in {torch.half, torch.bfloat16}:
            device_args = [arg.to(dtype=dtype) if
                           (torch.is_tensor(arg) and arg.dtype == torch.float) else arg
                           for arg in device_args]

        # Runs the tensor op on CPU and device
        cpu_result = getattr(cpu_tensor, op_str)(*cpu_args)
        device_result = getattr(device_tensor, op_str)(*device_args)

        dtype2precision = {torch.half: half_precision,
                           torch.bfloat16: bfloat16_precision}

        # Compares CPU and device inputs and outputs
        precision = dtype2precision.get(dtype, float_precision)
        self.assertEqual(cpu_tensor, device_tensor, prec=precision, exact_dtype=False)
        self.assertEqual(cpu_args, device_args, prec=precision, exact_dtype=False)
        self.assertEqual(cpu_result, device_result, prec=precision, exact_dtype=False)

    test_name = "test_" + op_str + subtest_str
    assert not hasattr(cls, test_name), "{0} already in TestDevicePrecision".format(test_name)

    # Constructs decorator list and applies decorators
    if decorators is None:
        decorators = [dtypes(*dtype_list)]
    else:
        decorators = decorators + [dtypes(*dtype_list)]

    for dec in decorators:
        fn = dec(fn)

    setattr(cls, test_name, fn)


def caller(cls,
           op_str,
           subtest_str,
           tensor_ctor,
           arg_ctor,
           half_precision=1e-5,
           bfloat16_precision=1e-5,
           float_precision=1e-5,
           dtype_list=_types,
           make_inplace_variant=True,
           decorators=None):
    if subtest_str:
        subtest_str = '_' + subtest_str

    generate_test_function(cls, op_str, subtest_str, tensor_ctor, arg_ctor, half_precision,
                           bfloat16_precision, float_precision, dtype_list, decorators)

    if make_inplace_variant:
        op_str = op_str + '_'
        subtest_str = 'inplace' + subtest_str
        generate_test_function(cls, op_str, subtest_str, tensor_ctor, arg_ctor, half_precision,
                               bfloat16_precision, float_precision, dtype_list, decorators)


def generate_tensor_op_tests(cls, op_list):
    for test in op_list:
        caller(cls, *test)


# Functions to test negative dimension wrapping
METHOD = 1
INPLACE_METHOD = 2
FUNCTIONAL = 4
DIM_ARG = None


def make_neg_dim_test(name, tensor_arg, arg_constr, types, extra_dim=0):
    def neg_dim_test(self):
        if isinstance(tensor_arg, list):
            assert METHOD not in types and INPLACE_METHOD not in types
            x = [torch.randn(arg) for arg in tensor_arg]
            ndim = len(tensor_arg[-1])
        else:
            x = torch.randn(*tensor_arg)
            ndim = len(tensor_arg)
        ndim += extra_dim

        n_dim_to_test = sum(map(lambda e: e is DIM_ARG, arg_constr()))

        for dims_val in combinations(range(ndim), n_dim_to_test):
            arg = arg_constr()
            arg_neg = copy.deepcopy(arg)
            idx = 0
            for i, v in enumerate(arg):
                if v is DIM_ARG:
                    arg[i] = dims_val[idx]
                    arg_neg[i] = dims_val[idx] - ndim
                    idx += 1

            if METHOD in types:
                a = getattr(x, name)(*arg)
                b = getattr(x, name)(*arg_neg)
                self.assertEqual(a, b)

            if INPLACE_METHOD in types:
                a = x.clone()
                getattr(a, name + '_')(*arg)
                b = x.clone()
                getattr(b, name + '_')(*arg_neg)
                self.assertEqual(a, b)

            if FUNCTIONAL in types:
                a = getattr(torch, name)(x, *arg)
                b = getattr(torch, name)(x, *arg_neg)
                self.assertEqual(a, b)

    return neg_dim_test


def add_neg_dim_tests():
    neg_dim_tests = [

        ('topk', (10, 20), lambda: [5, DIM_ARG], [METHOD, FUNCTIONAL]),
    ]

    for decl in neg_dim_tests:
        if len(decl) == 4:
            name, tensor_arg, arg_constr, types = decl
            extra_dim = 0
        elif len(decl) == 5:
            name, tensor_arg, arg_constr, types, extra_dim = decl

        test_name = 'test_' + name + '_neg_dim'

        assert not hasattr(TestTopk, test_name), "Duplicated test name: " + test_name
        setattr(TestTopk, test_name, make_neg_dim_test(name, tensor_arg, arg_constr, types, extra_dim))


class TestAbs(TestCase):
    exact_dtype = True

    def _make_tensors(self, shape, val_range=(-100, 100), use_floating=True, use_integral=True):
        float_types = [torch.double,
                       torch.float]
        int_types = [torch.int64,
                     torch.int32,
                     torch.int16]

        def make_contiguous(shape, dtype):
            if dtype in float_types:
                val = torch.randn(shape, dtype=dtype)
                val = val * ((val_range[1] - val_range[0]) / (math.pi * 2.0))
                val = val + ((val_range[1] - val_range[0]) / 2.0)
                val = torch.clamp(val, min=val_range[0], max=val_range[1])
                return val
            result = torch.zeros(shape, dtype=dtype)
            result.apply_(lambda x: random.randint(val_range[0], val_range[1]))
            return result

        def make_non_contiguous(shape, dtype):
            contig = make_contiguous(shape, dtype)
            non_contig = torch.empty(shape + (2, 2), dtype=dtype)[..., 0]
            non_contig = non_contig.select(-1, -1)
            non_contig.copy_(contig)
            self.assertFalse(non_contig.is_contiguous())
            return non_contig

        def make_contiguous_slice(size, dtype):
            contig = make_contiguous((1, size), dtype)
            non_contig = contig[:1, 1:size - 1]
            self.assertTrue(non_contig.is_contiguous())
            return contig

        types = []
        if use_floating:
            types += float_types
        if use_integral:
            types += int_types
        tensors = {"cont": [], "noncont": [], "slice": []}
        for dtype in types:
            tensors["cont"].append(make_contiguous(shape, dtype))
            tensors["noncont"].append(make_non_contiguous(shape, dtype))
            tensors["slice"].append(make_contiguous_slice(sum(list(shape)), dtype))

        return tensors

    # @dtypes(torch.float, torch.double)
    @dtypes(torch.float)
    def test_abs_zero(self, device, dtype):
        abs_zeros = torch.tensor([0.0, -0.0], device=device, dtype=dtype).abs().tolist()
        for num in abs_zeros:
            self.assertGreater(math.copysign(1.0, num), 0.0)

    @slowTest
    def test_abs_slow(self, device):
        def _test_abs(tensors_dict):
            for _category, tensors in tensors_dict.items():
                for data in tensors:
                    _test_abs_single(data)

        def _test_abs_single(data):
            switch = torch.rand(data.size()).mul(2).floor().mul(2).add(-1).type(data.dtype)
            res = torch.mul(data, switch)
            self.assertTensorsSlowEqual(res.abs(), data, 1e-16)

        shapes = [(3, 4), (3, 5, 7), (2, 2, 5, 8, 2, 3), (1000,), (10, 10, 10)]

        for shape in shapes:
            # Test all except char/byte
            _test_abs(self._make_tensors(shape, val_range=(0, 1000)))

            # Test char
            _test_abs_single(torch.CharTensor(*shape).random_(0, 100))

            # Test byte
            byte_tensor = torch.ByteTensor(*shape).random_(0, 100)
            self.assertTensorsSlowEqual(byte_tensor, byte_tensor.abs(), 1e-16)

        # Checking that the right abs function is called for LongTensor
        bignumber = 2 ** 31 + 1
        res = torch.LongTensor((-bignumber,))
        self.assertGreater(res.abs()[0], 0)

        # One of
        rec = torch.randn(2, 2, 3, 7, 6, 2).type(torch.float64).clamp(0, 1)
        val1 = rec.select(-1, -1)[0][0][0].sum()
        val2 = rec.select(-1, -1).abs()[0][0][0].sum()
        self.assertEqual(val1, val2, 1e-8, 'absolute value')

        # Both abs(0.0) and abs(-0.0) should result in 0.0
        for dtype in (torch.float, torch.double):
            for abs_zeros in (torch.tensor([0.0, -0.0], dtype=dtype).abs().tolist(),
                              # test a large tensor so that the vectorized version is tested
                              torch.abs(-torch.zeros(10000, dtype=dtype)).tolist()):
                for num in abs_zeros:
                    self.assertGreater(math.copysign(1.0, num), 0.0)


class TestAdd(TestCase):
    exact_dtype = True

    @unittest.skip("unittest is unsupported")
    def test_add_(self, device):
        # [res] torch.add([res,] tensor1, tensor2)
        m1 = torch.randn(100, 100, device=device)
        v1 = torch.randn(100, device=device)

        # contiguous
        res1 = torch.add(m1[4], v1)
        res2 = res1.clone().zero_()
        for i in range(m1.size(1)):
            res2[i] = m1[4, i] + v1[i]
        self.assertEqual(res1.to('cpu'), res2)

        m1 = torch.randn(100, 100, device=device)
        v1 = torch.randn(100, device=device)

        # non-contiguous
        res1 = torch.add(m1[:, 4], v1)
        res2 = res1.clone().zero_()
        for i in range(m1.size(0)):
            res2[i] = m1[i, 4] + v1[i]
        self.assertEqual(res1.to('cpu'), res2)

        # [res] torch.add([res,] tensor, value)
        m1 = torch.randn(10, 10, device=device)

        # contiguous
        res1 = m1.clone()
        res1[3].add_(2)
        res2 = m1.clone()
        for i in range(m1.size(1)):
            res2[3, i] = res2[3, i] + 2
        self.assertEqual(res1.to('cpu'), res2)

        # non-contiguous
        m1 = torch.randn(10, 10, device=device)
        res1 = m1.clone()
        res1[:, 3].add_(2)
        res2 = m1.clone()
        for i in range(m1.size(0)):
            res2[i, 3] = res2[i, 3] + 2
        self.assertEqual(res1.to('cpu'), res2)

        # inter-type
        m1 = torch.randn(10, 10, device=device)
        self.assertEqual((m1 + 3).to('cpu'), m1 + torch.tensor(3))
        # self.assertEqual(3 + m1, torch.tensor(3) + m1)
        self.assertEqual((3 + m1).to('cpu'), (3 + m1).to('cpu'))
        one = torch.tensor(1, dtype=torch.uint8, device=device)
        self.assertEqual(torch.add(one, 1).to('cpu'), 2)
        self.assertEqual(torch.add(one, 1).to('cpu').dtype, torch.uint8)

        # contiguous + non-contiguous
        m1 = torch.randn(10, 10, device=device)
        m2 = torch.randn(10, 10, device=device).t()
        res = m1 + m2
        self.assertTrue(res.is_contiguous())
        self.assertEqual(res.to('cpu'), m1 + m2.contiguous())

        # 1d + empty
        m1 = torch.tensor([1.0], dtype=torch.float, device=device)
        m2 = torch.tensor([], dtype=torch.float, device=device)
        self.assertEqual((m1 + m2).to('cpu'), [])

        # bool
        m1 = torch.tensor([True, False, False, True, False, False], dtype=torch.bool, device=device)
        m2 = torch.tensor([True, True, False, False, False, True], dtype=torch.bool, device=device)
        expected = torch.tensor([True, True, False, True, False, True], dtype=torch.bool, device=device)
        self.assertEqual((m1 + m2).to('cpu'), expected)

        # fused multiply add
        a = torch.zeros(2, 3, dtype=torch.bool, device=device)
        res = torch.add(a, a, alpha=0)
        expected = torch.zeros(2, 3, device=device).bool()
        self.assertEqual(res.to('cpu'), expected)

        # bfloat16
        m1 = torch.tensor([1., 2.], dtype=torch.bfloat16)
        m2 = torch.tensor([3., 4.], dtype=torch.bfloat16)
        self.assertEqual(m1 + m2, torch.tensor([4., 6.], dtype=torch.bfloat16))

        # mismatched alpha
        m1 = torch.tensor([1], dtype=torch.int8, device=device)
        m2 = torch.tensor([2], dtype=torch.int8, device=device)
        self.assertRaisesRegex(RuntimeError,
                               r"Boolean alpha only supported for Boolean results\.",
                               lambda: torch.add(m1, m2, alpha=True))
        self.assertRaisesRegex(RuntimeError,
                               r"For integral input tensors, argument alpha must not be a floating point number\.",
                               lambda: torch.add(m1, m2, alpha=1.0))

    @dtypes(torch.float)
    def test_add_with_tail(self, device, dtype):
        # test tensor where there is a tail which is not a multiple
        # of GPU warp size
        for tail_size in [1, 63, 67, 130]:
            size = 4096 + tail_size
            a = torch.randn(size, device=device, dtype=dtype)
            b = torch.randn(size, device=device, dtype=dtype)
            c = a + b
            for x, y, z in zip(a.tolist(), b.tolist(), c.tolist()):
                self.assertEqual(x + y, z)


class TestAddmm(TestCase):
    exact_dtype = True

    @dtypes(torch.float, torch.double)
    def test_addmm_sizes(self, device, dtype):
        for m in [1, 25]:
            for n in [1, 10]:
                for k in [1, 8]:
                    M = torch.randn(n, m, device=device, dtype=dtype)
                    m1 = torch.randn(n, k, device=device, dtype=dtype)
                    m2 = torch.randn(k, m, device=device, dtype=dtype)
                    res1 = torch.addmm(M, m1, m2)

                    res2 = torch.zeros(n, m, device=device, dtype=dtype)
                    res2 += M
                    for i in range(n):
                        for j in range(m):
                            for l in range(k):
                                res2[i, j] += m1[i, l] * m2[l, j]
                    self.assertEqual(res1.to('cpu'), res2)


class TestArange(TestCase):

    # @dtypes(torch.half, torch.float, torch.double,
    # torch.int8, torch.short, torch.int, torch.long,
    # torch.uint8)

    @dtypes(torch.half, torch.float, torch.int)
    def test_arange(self, device, dtype):
        shape_format = [
            [0, 100, 2, dtype], [1, 100, 1, dtype], ]

        for item in shape_format:
            # some dtype is not support by cpu
            # cpu_output = torch.arange(item[0], item[1], item[2], dtype=item[3], device="cpu").numpy()
            cpu_output = torch.arange(item[0], item[1], item[2], dtype=torch.float32, device="cpu").to(item[3]).numpy()
            npu_output = torch.arange(item[0], item[1], item[2], dtype=item[3], device=device).cpu().numpy()
            self.assertRtolEqual(cpu_output, npu_output)

    def test_arange_bfloat16(self, device):
        ref_tensor = torch.tensor([0, 1, 2, 3], dtype=torch.bfloat16, device=device)
        bfloat16_tensor = torch.arange(0, 4, dtype=torch.bfloat16, device=device)
        self.assertEqual(ref_tensor, bfloat16_tensor)

        # step=2
        ref_tensor = torch.tensor([0, 2, 4], dtype=torch.bfloat16, device=device)
        bfloat16_tensor = torch.arange(0, 6, step=2, dtype=torch.bfloat16, device=device)
        self.assertEqual(ref_tensor, bfloat16_tensor)


class TestBmm(TestCase):
    exact_dtype = True

    @dtypes(torch.float)
    def test_bmm_(self, device, dtype):
        num_batches = 10
        M, N, O = 23, 8, 12
        b1 = torch.randn(num_batches, M, N, dtype=dtype, device=device)
        b2 = torch.randn(num_batches, N, O, dtype=dtype, device=device)
        res = torch.bmm(b1, b2)
        for i in range(num_batches):
            r = torch.mm(b1[i], b2[i])
            self.assertEqual(r.to('cpu'), res[i])
        if torch.cuda.is_available():
            # check that mixed arguments are rejected
            self.assertRaises(RuntimeError, lambda: torch.bmm(b1, b2.cuda()))
            self.assertRaises(RuntimeError, lambda: torch.bmm(b1.cuda(), b2))


class TestCat(TestCase):
    exact_dtype = True

    @dtypes(torch.half, torch.float, torch.int8, torch.int)
    def test_cat_all_dtypes_and_devices(self, device, dtype):
        x = torch.tensor([[1, 2], [3, 4]], dtype=dtype, device=device)

        expected1 = torch.tensor([[1, 2], [3, 4], [1, 2], [3, 4]], dtype=dtype, device=device)
        # self.assertEqual(torch.cat((x, x), 0), expected1)
        torch.equal(torch.cat((x, x), 0), expected1)

        expected2 = torch.tensor([[1, 2, 1, 2], [3, 4, 3, 4]], dtype=dtype, device=device)
        # self.assertEqual(torch.cat((x, x), 1), expected2)
        torch.equal(torch.cat((x, x), 1), expected2)

    @unittest.skip("unittest is unsupported")
    def test_cat_empty(self, device):
        dtype = torch.float32

        x = torch.randn((4, 3, 32, 32), dtype=dtype, device=device)
        empty = torch.randn((4, 0, 32, 32), dtype=dtype, device=device)

        res1 = torch.cat([x, empty], dim=1)
        res2 = torch.cat([empty, x], dim=1)
        # self.assertEqual(res1, res2)
        torch.equal(res1, res2)

        res1 = torch.cat([empty, empty], dim=1)
        self.assertEqual(res1, empty)

        # check non-legacy-behavior (sizes don't match)
        empty = torch.randn((4, 0, 31, 32), dtype=dtype, device=device)
        self.assertRaises(RuntimeError, lambda: torch.cat([x, empty], dim=1))
        self.assertRaises(RuntimeError, lambda: torch.cat([empty, x], dim=1))

        # check non-legacy-behavior (dimensions don't match)
        empty = torch.randn((4, 0), dtype=dtype, device=device)
        self.assertRaises(RuntimeError, lambda: torch.cat([x, empty], dim=1))
        self.assertRaises(RuntimeError, lambda: torch.cat([empty, x], dim=1))

    #@unittest.skip("unittest is unsupported")
    def test_cat_out(self, device):
        x = torch.zeros((0), device=device)
        y = torch.randn((4, 6), device=device)

        with self.assertRaisesRegex(
                RuntimeError, r"unsupported operation:.* input tensor 0"):
            torch.cat([x, y], dim=0, out=x)

        with self.assertRaisesRegex(
                RuntimeError, r"unsupported operation:.* input tensor 1"):
            torch.cat([x, y], dim=0, out=y)

        z = torch.zeros((4, 6), device=device)
        with self.assertRaisesRegex(
                RuntimeError, r"unsupported operation:.* input tensor 1"):
            torch.cat([y, z], out=z[:2, :])

        w = y.view(-1).clone()
        a = torch.cat([w[:2], w[4:6]])
        b = torch.cat([w[:2], w[4:6]], out=w[6:10])
        self.assertEqual(a, b)
        self.assertEqual(w[:6], y.view(-1)[:6])

    def test_cat_out_channels_last(self, device):
        x = torch.randn((4, 3, 8, 8))
        y = torch.randn(x.shape)
        res1 = torch.cat((x, y))
        z = res1.clone().contiguous(memory_format=torch.channels_last)
        res2 = torch.cat((x, y), out=z)
        self.assertEqual(res1, res2)

    def test_cat_bad_dtypes(self, device):
        def cross_product(a, b, skip_same=True):
            result = []
            for dtype_a in a:
                for dtype_b in b:
                    if skip_same and (dtype_a == dtype_b):
                        continue
                    result.append((dtype_a, dtype_b))
            return result
            
    # @unittest.skip("unittest is unsupported")
    def test_cat_big(self, device):
        SIZE1 = 6500
        SIZE2 = 4500
        concat_list = []
        concat_list.append(torch.ones((SIZE1, 1024 * 512), dtype=torch.uint8, device=device))
        concat_list.append(torch.ones((SIZE2, 1024 * 512), dtype=torch.uint8, device=device))
        result = torch.cat(concat_list)
        self.assertEqual(result.size(0), SIZE1 + SIZE2)

    def test_cat_bad_input_sizes(self, device):
        x = torch.randn(2, 1, device=device)
        y = torch.randn(2, 1, 1, device=device)
        z = torch.randn(2, 1, 1, device=device)
        self.assertRaises(RuntimeError, lambda: torch.cat([x, y, z]))

        x = torch.randn(2, 1, 2, device=device)
        y = torch.randn(2, 1, 1, device=device)
        z = torch.randn(2, 2, 1, device=device)
        self.assertRaises(RuntimeError, lambda: torch.cat([x, y, z], dim=1))

    # @unittest.skip("unittest is unsupported")
    def test_cat_scalars(self, device):
        x = torch.tensor(0, device=device)
        y = torch.tensor(1, device=device)
        # with self.assertRaisesRegex(RuntimeError, 'zero-dimensional.*cannot be concatenated'):
        with self.assertRaisesRegex(IndexError, 'dimension specified as 0 but tensor has no dimensions'):
            torch.cat([x, y])

    @deviceCountAtLeast(2)
    def test_cat_different_devices(self, devices):
        cuda0 = torch.randn((3, 3), device=devices[0])
        cuda1 = torch.randn((3, 3), device=devices[1])
        with self.assertRaisesRegex(RuntimeError,
                                    "input tensors must be on the same device"):
            torch.cat((cuda0, cuda1))
        cpu = torch.randn(3, 3)
        with self.assertRaisesRegex(RuntimeError,
                                    "input tensors must be on the same device"):
            torch.cat((cuda0, cpu))
        with self.assertRaisesRegex(RuntimeError,
                                    "input tensors must be on the same device"):
            torch.cat((cpu, cuda0))

    def test_cat_preserve_channels_last(self, device):
        x = torch.randn((4, 3, 8, 8), device=device)
        y = torch.randn(x.shape, device=device)
        res1 = torch.cat((x, y))
        res2 = torch.cat(
            (x.contiguous(memory_format=torch.channels_last), y.contiguous(memory_format=torch.channels_last)))
        self.assertEqual(res1.to('cpu'), res2)
        self.assertTrue(res2.is_contiguous(memory_format=torch.channels_last))

    def test_cat(self, device):
        SIZE = 2
        for dim in range(-3, 3):
            pos_dim = dim if dim >= 0 else 3 + dim
            x = torch.rand(13, SIZE, SIZE, device=device).transpose(0, pos_dim)
            y = torch.rand(17, SIZE, SIZE, device=device).transpose(0, pos_dim)
            z = torch.rand(19, SIZE, SIZE, device=device).transpose(0, pos_dim)

            res1 = torch.cat((x, y, z), dim)
            temp = res1.narrow(pos_dim, 0, 13)
            self.assertEqual(res1.narrow(pos_dim, 0, 13).to('cpu'), x, 0)
            self.assertEqual(res1.narrow(pos_dim, 13, 17).to('cpu'), y, 0)
            self.assertEqual(res1.narrow(pos_dim, 30, 19).to('cpu'), z, 0)

        x = torch.randn(20, SIZE, SIZE, device=device)
        self.assertEqual(torch.cat(torch.split(x, 7)).to('cpu'), x)
        self.assertEqual(torch.cat(torch.chunk(x, 7)).to('cpu'), x)

        y = torch.randn(1, SIZE, SIZE, device=device)
        z = torch.cat([x, y])
        self.assertEqual(z.size(), (21, SIZE, SIZE))


class TestClamp(TestCase):
    exact_dtype = True

    def test_clamp(self, device):
        m1 = torch.rand(100, device=device).mul(5).add(-2.5)  # uniform in [-2.5, 2.5]
        # just in case we're extremely lucky.
        min_val = -1
        max_val = 1
        m1[1] = min_val
        m1[2] = max_val

        res1 = m1.clone()
        res1.clamp_(min_val, max_val)
        res2 = m1.clone()
        for i in iter_indices(res2):
            res2[i] = max(min_val, min(max_val, res2[i]))
        self.assertEqual(res1.to('cpu'), res2)

        out = m1.clone()
        torch.clamp(m1, min=min_val, max=max_val, out=out)
        self.assertEqual(out.to('cpu'), res1)

        res1 = torch.clamp(m1, min=min_val)
        res2 = m1.clone()
        for i in iter_indices(res2):
            res2[i] = max(min_val, res2[i])
        self.assertEqual(res1.to('cpu'), res2)

        torch.clamp(m1, min=min_val, out=out)
        self.assertEqual(out.to('cpu'), res1)

        res1 = torch.clamp(m1, max=max_val)
        res2 = m1.clone()
        for i in iter_indices(res2):
            res2[i] = min(max_val, res2[i])
        self.assertEqual(res1.to('cpu'), res2)

        torch.clamp(m1, max=max_val, out=out)
        self.assertEqual(out.to('cpu'), res1)

        # if the tensor contains nan case
        test_tens = torch.tensor([nan], device=device)

        res1 = test_tens.clone()
        res1.clamp_(min_val, max_val)
        res2 = test_tens.clone()
        for i in iter_indices(res2):
            res2[i] = max(min(res2[i], max_val), min_val)
        self.assertEqual(torch.isnan(res1).to('cpu'), torch.isnan(res2))

        out = test_tens.clone()
        torch.clamp(test_tens, min=min_val, max=max_val, out=out)
        self.assertEqual(torch.isnan(out).to('cpu'), torch.isnan(res1))

        res1 = torch.clamp(test_tens, min=min_val)
        res2 = test_tens.clone()
        for i in iter_indices(res2):
            res2[i] = max(res2[i], min_val)
        # self.assertEqual(torch.isnan(res1).to('cpu'), torch.isnan(res2))

        torch.clamp(test_tens, min=min_val, out=out)
        self.assertEqual(torch.isnan(out).to('cpu'), torch.isnan(res1))

        res1 = torch.clamp(test_tens, max=max_val)
        res2 = test_tens.clone()
        for i in iter_indices(res2):
            res2[i] = min(res2[i], max_val)
        self.assertEqual(torch.isnan(res1).to('cpu'), torch.isnan(res2))

        torch.clamp(test_tens, max=max_val, out=out)
        self.assertEqual(torch.isnan(out).to('cpu'), torch.isnan(res1))

        # error_msg = 'At least one of \'min\' or \'max\' must not be None'
        # with self.assertRaisesRegex(RuntimeError, error_msg):
        # m1.clamp()
        # with self.assertRaisesRegex(RuntimeError, error_msg):
        # m1.clamp_()


class TestConv2d(TestCase):
    exact_dtype = True

    def test_Conv2d_inconsistent_types(self, device):
        inputs = torch.randn(4, 1, 7, 7, dtype=torch.float)
        weights = torch.randn(1, 1, 3, 3, dtype=torch.double)
        # inconsistent types should raise an exception
        self.assertRaises(RuntimeError, lambda: nn.functional.conv2d(inputs, weights))
        # but it should work with the same type
        nn.functional.conv2d(inputs.float(), weights.float())

    def test_Conv2d_inconsistent_types_on_GPU_with_cudnn(self, device):
        inputs = torch.randn(4, 1, 7, 7, dtype=torch.float, device="npu")
        weights = torch.randn(1, 1, 3, 3, dtype=torch.float16, device="npu")
        bias = torch.randn(1, dtype=torch.float16, device="npu")
        #with torch.backends.cudnn.flags(enabled=True):
            # inconsistent types should raise an exception
        #    self.assertRaises(RuntimeError, lambda: nn.functional.conv2d(inputs, weights))
        #    self.assertRaises(RuntimeError, lambda: nn.functional.conv2d(inputs, weights.float(), bias))
        
        # but it should work with the same type
        nn.functional.conv2d(inputs.float(), weights.float(), bias.float())

    # @repeat_test_for_types(ALL_TENSORTYPES2)
    def test_Conv2d_deterministic(self, device, dtype=torch.float):
        inputs = torch.randn(2, 3, 5, 5, device="npu", dtype=dtype, requires_grad=True)
        conv1 = torch.nn.Conv2d(3, 3, 3).to("npu", dtype)
        conv2 = torch.nn.Conv2d(3, 3, 3).to("npu", dtype)
        conv2.bias.data.copy_(conv1.bias.data)
        conv2.weight.data.copy_(conv1.weight.data)
        out1 = conv1(inputs)
        out2 = conv2(inputs)
        self.assertEqual(out1.to('cpu'), out2, prec=0.0)
        y = torch.randn(out1.size(), device="npu", dtype=dtype)
        out1.backward(y)
        out2.backward(y)
        self.assertEqual(conv1.bias.grad.data.to('cpu'), conv2.bias.grad.data, prec=0.0)
        self.assertEqual(conv1.weight.grad.data.to('cpu'), conv2.weight.grad.data, prec=0.0)

    def test_Conv2d_missing_argument(self, device):
        c = nn.Conv2d(3, 3, 3).to(device)
        self.assertRaises(TypeError, lambda: c(None))

    def test_Conv2d_backward_twice(self, device):
        input = torch.randn(2, 3, 5, 5).to(device)
        c = nn.Conv2d(3, 3, 3).to(device)
        o1 = c(input)
        o1.sum().backward()
        self.assertRaisesRegex(RuntimeError, 'Specify retain_graph=True',
                               lambda: o1.sum().backward())

    # @repeat_test_for_types(ALL_TENSORTYPES2)
    def test_Conv2d_large_workspace(self, device, dtype=torch.float):
        # These sizes require huge cuDNN workspaces. Make sure we choose a
        # reasonable algorithm that does not run out of memory
        sizes = [
            (1, 256, 109, 175),
            (1, 256, 80, 128),
            (1, 256, 120, 192),
        ]

        def run_test(benchmark):
            # with torch.backends.cudnn.flags(benchmark=benchmark):
            conv = torch.nn.Conv2d(256, 256, kernel_size=3, padding=1).to("npu", dtype)
            for size in sizes:
                x = torch.randn(size, device="npu", dtype=dtype)
                out = conv(x.detach().clone().requires_grad_())
                out.backward(torch.ones_like(out))

        run_test(benchmark=False)
        run_test(benchmark=True)

    # @unittest.skip("unittest is unsupported")
    def test_Conv2d_groups_nobias(self,device):
        # group parameter is not unsupport for npu
        # dev_dtypes = [("cpu", torch.float)]
        dev_dtypes = [("npu", torch.float), ("npu", torch.half)]
        # if TEST_WITH_ROCM:
        #    dev_dtypes += [("npu", torch.bfloat16)]
        for device, dtype in dev_dtypes:
            m = nn.Conv2d(4, 4, kernel_size=3, groups=2, bias=False).to(device, dtype)
            i = torch.randn(2, 4, 6, 6, device=device, dtype=dtype, requires_grad=True)
            output = m(i)
            grad_output = torch.randn(2, 4, 4, 4, device=device, dtype=dtype)
            output.backward(grad_output)

            m1 = nn.Conv2d(2, 2, kernel_size=3, bias=False).to(device, dtype)
            m1.weight.data.copy_(m.weight.data[:2])
            i1 = i.data[:, :2].contiguous().requires_grad_(True)
            output1 = m1(i1)
            output1.backward(grad_output[:, :2].contiguous())

            m2 = nn.Conv2d(2, 2, kernel_size=3, bias=False).to(device, dtype)
            m2.weight.data.copy_(m.weight.data[2:])
            i2 = i.data[:, 2:].contiguous().requires_grad_(True)
            output2 = m2(i2)
            output2.backward(grad_output[:, 2:].contiguous())

            self.assertEqual(output, torch.cat([output1, output2], 1))
            self.assertEqual(i.grad.data,
                             torch.cat([i1.grad.data, i2.grad.data], 1),
                             dtype2prec_DONTUSE[dtype])
            self.assertEqual(m.weight.grad.data,
                             torch.cat([m1.weight.grad.data, m2.weight.grad.data], 0),
                             1e-1 if dtype == torch.half else dtype2prec_DONTUSE[dtype])

    # Almost identical to the above `test_Conv2d_naive_groups`
    # Covering special case when group > 1, input-channel / group < 16 and output-channel is multiple of 16
    # See also https://github.com/pytorch/pytorch/pull/18463#issuecomment-476563686
    # and https://github.com/pytorch/pytorch/pull/18463#issuecomment-477001024
    # @unittest.skip("unittest is unsupported")
    def test_Conv2d_groups_nobias_v2(self, device):
        torch.manual_seed(123)
        dev_dtypes = [("npu", torch.float),("npu", torch.half)]
        # if TEST_WITH_ROCM:
        #     dev_dtypes += [("npu", torch.bfloat16)]
        for device, dtype in dev_dtypes:
            m = nn.Conv2d(4, 16, kernel_size=3, groups=2, bias=False).to(device, dtype)
            i = torch.randn(2, 4, 6, 6, device=device, dtype=dtype, requires_grad=True)
            output = m(i)
            grad_output = torch.randn(2, 16, 4, 4, device=device, dtype=dtype)
            output.backward(grad_output)

            m1 = nn.Conv2d(2, 8, kernel_size=3, bias=False).to(device, dtype)
            m1.weight.data.copy_(m.weight.data[:8])
            i1 = i.data[:, :2].contiguous().requires_grad_(True)
            output1 = m1(i1)
            output1.backward(grad_output[:, :8].contiguous())

            m2 = nn.Conv2d(2, 8, kernel_size=3, bias=False).to(device, dtype)
            m2.weight.data.copy_(m.weight.data[8:])
            i2 = i.data[:, 2:].contiguous().requires_grad_(True)
            output2 = m2(i2)
            output2.backward(grad_output[:, 8:].contiguous())

            self.assertEqual(output, torch.cat([output1, output2], 1))
            self.assertEqual(i.grad.data,
                             torch.cat([i1.grad.data, i2.grad.data], 1),
                             dtype2prec_DONTUSE[dtype])
            self.assertEqual(m.weight.grad.data,
                             torch.cat([m1.weight.grad.data, m2.weight.grad.data], 0),
                             1e-1 if dtype == torch.half else dtype2prec_DONTUSE[dtype])

    # Very similar to test_Conv2d_naive_groups but with special care to handle
    # the number of groups == number of input channels
    # @repeat_test_for_types(ALL_TENSORTYPES)
    # @unittest.skip("unittest is unsupported")
    def test_Conv2d_depthwise_naive_groups(self, device, dtype=torch.float):
        for depth_multiplier in [1, 2]:
            m = nn.Conv2d(2, 2 * depth_multiplier, kernel_size=3, groups=2).to("npu", dtype)
            i = torch.randn(2, 2, 6, 6, device="npu", dtype=dtype).div_(2).requires_grad_()
            output = m(i)
            grad_output = torch.randn(2, 2 * depth_multiplier, 4, 4, device="npu", dtype=dtype) / 2
            output.backward(grad_output)

            offset = 1 * depth_multiplier

            m1 = nn.Conv2d(1, 1 * depth_multiplier, kernel_size=3).to("npu", dtype)
            m1.weight.data = m.weight.data[:offset].clone()
            m1.bias.data = m.bias.data[:offset].clone()
            i1 = i.detach()[:, :1].clone().requires_grad_()
            output1 = m1(i1)
            output1.backward(grad_output[:, :offset].contiguous())

            m2 = nn.Conv2d(1, 1 * depth_multiplier, kernel_size=3).to("npu", dtype)
            m2.weight.data.copy_(m.weight.data[offset:])
            m2.bias.data.copy_(m.bias.data[offset:])
            i2 = i.detach()[:, 1:].clone().requires_grad_()
            output2 = m2(i2)
            output2.backward(grad_output[:, offset:].contiguous())

            self.assertEqual(output.to('cpu'), torch.cat([output1, output2], 1),
                             prec=dtype2prec_DONTUSE[dtype])
            self.assertEqual(i.grad.data.to('cpu'),
                             torch.cat([i1.grad.data, i2.grad.data], 1),
                             prec=dtype2prec_DONTUSE[dtype])
            self.assertEqual(m.bias.grad.data.to('cpu'),
                             torch.cat([m1.bias.grad.data,
                                        m2.bias.grad.data], 0),
                             prec=dtype2prec_DONTUSE[dtype])
            self.assertEqual(m.weight.grad.data.to('cpu'),
                             torch.cat([m1.weight.grad.data,
                                        m2.weight.grad.data], 0),
                             prec=dtype2prec_DONTUSE[dtype])

    def run_grad_conv_test(self, func_forward, func_backward, dim=1, gradient='input'):
        for kern, inp_size in [(3, 6), (3, 7), (4, 9)]:
            for batch, stride, padding, chan_in, chan_out, dilation in \
                    product([1, 2], [1, 2], [0, 1, 2], [2], [3], [1]):

                for has_bias in [True, False]:
                    input_shape = [batch, chan_in]
                    weight_shape = [chan_out, chan_in]
                    for _ in range(dim):
                        input_shape.append(inp_size)
                        weight_shape.append(kern)

                    input = torch.randn(input_shape, requires_grad=True)
                    weight = torch.randn(weight_shape, requires_grad=True)
                    if has_bias:
                        bias = torch.randn([chan_out], requires_grad=True)
                    output = func_forward(input, weight, stride=stride, padding=padding, dilation=dilation, bias=bias)

                    gradient_o = torch.randn(output.shape)
                    gradient_w = torch.autograd.grad(output, input if (gradient == 'input') else weight, gradient_o)

                    self.assertAlmostEqual(gradient_w[0],
                                           func_backward(
                                               input_shape if (gradient == 'input') else input,
                                               weight_shape if (gradient == 'weight') else weight,
                                               gradient_o,
                                               stride=stride,
                                               padding=padding,
                                               dilation=dilation))

    def test_invalid_conv2d(self, device):
        for dtype in [torch.bfloat16, torch.float, torch.double]:
            module = torch.nn.Conv2d(1, 1, kernel_size=3, dilation=2, stride=2).to(dtype)
            input = torch.empty(1, 1, 4, 4).to(dtype)
            self.assertRaises(RuntimeError, lambda: module(input))

            module = nn.Conv2d(in_channels=3, out_channels=33, kernel_size=10, stride=1, bias=True)
            input = torch.randn(1, 3, 1, 1)
            with self.assertRaisesRegex(RuntimeError,
                                        r'Calculated padded input size per channel: \(1 x 1\). ' +
                                        r'Kernel size: \(10 x 10\). Kernel size can\'t be greater than actual input size'):
                module(input)

            # Negative stride check
            module = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=4, stride=-1, bias=True).to(dtype)
            input = torch.randn(1, 3, 4, 4).to(dtype)
            with self.assertRaisesRegex(RuntimeError, 'non-positive stride is not supported'):
                module(input)

            # Zero stride check
            module = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=4, stride=0, bias=True).to(dtype)
            input = torch.randn(1, 3, 4, 4).to(dtype)
            with self.assertRaisesRegex(RuntimeError, 'non-positive stride is not supported'):
                module(input)

    def test_mismatch_shape_conv2d(self, device):
        x = torch.randn(1, 10, 1, 28, 28)
        w = torch.randn(6, 1, 5, 5)

        with self.assertRaisesRegex(RuntimeError,
                                    r'Expected 4-dimensional input for 4-dimensional weight \[6, 1, 5, 5\],' +
                                    r' but got 5-dimensional input of size \[1, 10, 1, 28, 28\] instead'):

            F.conv2d(x, w)

    # @dtypesIfCUDA(*ALL_TENSORTYPES2)
    # @dtypes(torch.float)
    # @unittest.skip("unittest is unsupported")
    def test_Conv2d_naive_groups(self, device, dtype):
        # Check that grouped convolutions matches two half convolutions
        m = nn.Conv2d(4, 4, kernel_size=3, groups=2).to(device, dtype)
        i = torch.randn(2, 4, 6, 6, device=device, dtype=dtype, requires_grad=True)
        output = m(i)
        grad_output = torch.randn(2, 4, 4, 4, device=device, dtype=dtype)
        output.backward(grad_output)

        m1 = nn.Conv2d(2, 2, kernel_size=3).to(device, dtype)
        m1.weight.data.copy_(m.weight.data[:2])
        m1.bias.data.copy_(m.bias.data[:2])
        i1 = i.data[:, :2].contiguous().requires_grad_(True)
        output1 = m1(i1)
        output1.backward(grad_output[:, :2].contiguous())

        m2 = nn.Conv2d(2, 2, kernel_size=3).to(device, dtype)
        m2.weight.data.copy_(m.weight.data[2:])
        m2.bias.data.copy_(m.bias.data[2:])
        i2 = i.data[:, 2:].contiguous().requires_grad_(True)
        output2 = m2(i2)
        output2.backward(grad_output[:, 2:].contiguous())

        self.assertEqual(output, torch.cat([output1, output2], 1))
        self.assertEqual(i.grad.data,
                         torch.cat([i1.grad.data, i2.grad.data], 1),
                         prec=dtype2prec_DONTUSE[dtype])
        self.assertEqual(m.bias.grad.data,
                         torch.cat([m1.bias.grad.data, m2.bias.grad.data], 0),
                         prec=dtype2prec_DONTUSE[dtype])
        self.assertEqual(m.weight.grad.data,
                         torch.cat([m1.weight.grad.data, m2.weight.grad.data], 0),
                         prec=dtype2prec_DONTUSE[dtype])

    # @skipCUDAIfCudnnVersionLessThan(7603)
    # @unittest.skip("unittest is unsupported")
    def test_convert_conv2d_weight_memory_format(self, device='npu'):
        input = torch.randint(1, 10, (2, 8, 4, 4), dtype=torch.float32, device=device)
        model = nn.Sequential(
            nn.Conv2d(8, 4, 3),
            nn.BatchNorm2d(4)).to(device).float()
        for memory_format in [torch.channels_last, torch.contiguous_format]:
            model = nn.utils.convert_conv2d_weight_memory_format(model, memory_format)
            out = model(input)
            self.assertTrue(out.is_contiguous(memory_format=memory_format))

        model = nn.Sequential(
            nn.ConvTranspose2d(8, 4, 3),
            nn.BatchNorm2d(4)).to(device).float()
        for memory_format in [torch.channels_last, torch.contiguous_format]:
            model = nn.utils.convert_conv2d_weight_memory_format(model, memory_format)
            out = model(input)
            self.assertTrue(out.is_contiguous(memory_format=memory_format))

    def test_grad_conv2d_input(self, device):
        self.run_grad_conv_test(F.conv2d, F.grad.conv2d_input, 2, 'input')

    def test_grad_conv2d_weight(self, device):
        self.run_grad_conv_test(F.conv2d, F.grad.conv2d_weight, 2, 'weight')


class TestDiv(TestCase):
    exact_dtype = True

    @dtypes(torch.bfloat16, torch.float)
    def test_div_(self, device, dtype):
        m1 = torch.randn(10, 10, dtype=torch.float).to(dtype=dtype).to(device=device)
        res1 = m1.clone()
        res1[:, 3].div_(2)
        res2 = m1.clone()
        for i in range(m1.size(0)):
            res2[i, 3] = res2[i, 3] / 2
        self.assertEqual(res1.to('cpu'), res2)

        if dtype == torch.bfloat16:
            a1 = torch.tensor([4.2, 6.2], dtype=dtype, device=device)
            a2 = torch.tensor([2., 2.], dtype=dtype, device=device)
            self.assertEqual(a1 / a2,
                             torch.tensor([2.1, 3.1], dtype=dtype, device=device),
                             0.01)
            self.assertEqual(a1.div(a2), a1 / a2)

    # @unittest.skip("unittest is unsupported")
    @dtypes(torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64)
    def test_div_zero(self, device, dtype):
        # a = torch.tensor([0, 1], dtype=dtype, device=device)
        # b = torch.tensor([0, 1], dtype=dtype, device=device)
        a = torch.tensor([0, 1], dtype=torch.float, device=device)
        b = torch.tensor([0, 1], dtype=torch.float, device=device)
        a = a.to(dtype)
        b = b.to(dtype)
        with self.assertRaisesRegex(RuntimeError, 'ZeroDivisionError'):
            a.div(b)


class TestEq(TestCase):
    exact_dtype = True


class TestExp(TestCase):
    exact_dtype = True


class TestFill_(TestCase):
    exact_dtype = True

    def test_fill_all_dtypes_and_devices(self, device):
        for dt in torch.testing.get_all_dtypes():
            for x in [torch.tensor((10, 10), dtype=dt, device=device),
                      torch.empty(10000, dtype=dt, device=device)]:  # large tensor
                numel = x.numel()
                bound = 100 if dt in (torch.uint8, torch.int8) else 2000
                for n in range(-bound, bound, bound // 10):
                    x.fill_(n)
                    self.assertEqual(x, torch.tensor([n] * numel, dtype=dt, device=device))
                    self.assertEqual(dt, x.dtype)


class TestFill_diagonal(TestCase):
    exact_dtype = True

    def test_fill_diagonal(self, device):
        a1 = torch.randn(7, 3).to(device=device)
        a2 = a1.clone()
        v = 1
        for i in range(3):
            a2[i][i] = v
        a1.fill_diagonal_(v)
        self.assertEqual(a1.to('cpu'), a2)

        b1 = torch.randn(7, 3).to(device=device)
        b2 = b1.clone()
        for i in range(3):
            b2[i][i] = v
            b2[i + 4][i] = v
        b1.fill_diagonal_(v, wrap=True)
        self.assertEqual(b1.to('cpu'), b2)

        c1 = torch.rand(3, 3, 3).to(device=device)
        c2 = c1.clone()
        for i in range(3):
            c2[i][i][i] = v
        c1.fill_diagonal_(v)
        self.assertEqual(c1.to('cpu'), c2)

        # non-contiguous tensor
        d1 = torch.rand(3, 3, 3, device=device)[:, 1, ...]
        d2 = d1.clone()
        for i in range(3):
            d2[i][i] = v
        d1.fill_diagonal_(v)
        self.assertEqual(d1.to('cpu'), d2)

        e1 = torch.rand(7, 3, 3, device=device)[:, 1, ...]
        e2 = e1.clone()
        for i in range(3):
            e2[i][i] = v
            e2[i + 4][i] = v
        e1.fill_diagonal_(v, wrap=True)
        self.assertEqual(e1.to('cpu'), e2)


class TestFloor(TestCase):
    exact_dtype = True

    @dtypesIfCUDA(*torch.testing.get_all_math_dtypes('cuda'))
    @dtypes(*torch.testing.get_all_math_dtypes('cpu'))
    def test_floor_divide_tensor_(self, device, dtype):
        x = torch.randn(10, device=device).mul(30).to(dtype)
        y = torch.arange(1, 11, dtype=dtype, device=device)

        z = x // y
        z_alt = torch.trunc(x.double() / y.double()).to(dtype)

        self.assertEqual(z.dtype, x.dtype)
        self.assertEqual(z, z_alt)

    @dtypesIfCUDA(*torch.testing.get_all_math_dtypes('cuda'))
    @dtypes(*torch.testing.get_all_math_dtypes('cpu'))
    def test_floor_divide_scalar(self, device, dtype):
        x = torch.randn(100, device=device).mul(10).to(dtype)

        z = x // 3
        z_alt = torch.tensor([math.trunc(v.item() / 3.) for v in x], dtype=x.dtype, device=device)

        self.assertEqual(z.dtype, x.dtype)
        self.assertEqual(z, z_alt)

    # Note: this tests fails on XLA
    # @onlyOnCPUAndCUDA
    @dtypes(torch.float, torch.long)
    def test_floor_divide_out(self, device, dtype):
        x = torch.randn(10, device=device).mul(10).to(dtype)
        y = torch.arange(1, 11, dtype=dtype, device=device)
        o = torch.empty(10, dtype=dtype, device=device)

        torch.floor_divide(x, y, out=o)
        self.assertEqual(o, x // y)

        # Tests scalar with out
        torch.floor_divide(x, 2, out=o)
        self.assertEqual(o, x // 2)

        if dtype == torch.int:
            o = torch.empty(10, dtype=torch.float, device=device)
            torch.floor_divide(x, y, out=o)
            self.assertEqual(o, torch.floor_divide(x.float(), y.float()))


class TestFmod(TestCase):
    exact_dtype = True


class TestGe(TestCase):
    exact_dtype = True


class TestFull(TestCase):
    def test_full_deprecation_warning(self, device):
        size = (2, 2)
        # Tests bool and integer fill_values deprecated without specific dtype set
        with self.maybeWarnsRegex(UserWarning, 'Deprecation warning: .+'):
            self.assertEqual(torch.full(size, True).dtype, torch.float)
        with self.maybeWarnsRegex(UserWarning, 'Deprecation warning: .+'):
            self.assertEqual(torch.full(size, 1).dtype, torch.float)

        # Explicitly setting the dtype doesn't warn
        with self.maybeWarnsRegex(UserWarning, ''):
            self.assertEqual(torch.full(size, 1, dtype=torch.long).dtype, torch.long)
        with self.maybeWarnsRegex(UserWarning, ''):
            self.assertEqual(torch.full(size, True, dtype=torch.bool).dtype,
                             torch.bool)

        # Performs same tests with named tensor
        with self.maybeWarnsRegex(UserWarning, 'Deprecation warning: .+|Named tensors .+'):
            self.assertEqual(torch.full(size, True, names=('a', 'b')).dtype, torch.float)
        with self.maybeWarnsRegex(UserWarning, 'Deprecation warning: .+|Named tensors .+'):
            self.assertEqual(torch.full(size, 1, names=('a', 'b')).dtype, torch.float)

        with self.maybeWarnsRegex(UserWarning, 'Named tensors .+'):
            dt = torch.full(size, True, names=('a', 'b'), dtype=torch.bool).dtype
            self.assertEqual(dt, torch.bool)
        with self.maybeWarnsRegex(UserWarning, 'Named tensors .+'):
            dt = torch.full(size, 1, names=('a', 'b'), dtype=torch.long).dtype
            self.assertEqual(dt, torch.long)

    @dtypes(torch.half, torch.float, torch.double)
    def test_full_inference(self, device, dtype):
        size = (2, 2)

        prev_default = torch.get_default_dtype()
        torch.set_default_dtype(dtype)

        # Tests bool fill value inference
        # Note: in the future this will return a tensor of torch.bool dtype
        t = torch.full(size, True)
        self.assertEqual(t.dtype, dtype)

        # Tests integer fill value inference
        # Note: in the future this will return a tensor of torch.long dtype
        t = torch.full(size, 1)
        self.assertEqual(t.dtype, dtype)

        # Tests float fill value inference
        t = torch.full(size, 1.)
        self.assertEqual(t.dtype, dtype)

        torch.set_default_dtype(prev_default)

    @dtypes(torch.half, torch.float, torch.int)
    def test_full_out(self, device, dtype):
        # o = torch.empty((5,), device=device, dtype=torch.long)
        o = torch.empty((5,), device=device, dtype=dtype)
        # verifies dtype/out conflict throws a RuntimeError
        with self.assertRaises(RuntimeError):
            torch.full(o.shape, 1., dtype=torch.float, out=o)

        # verifies out dtype overrides inference
        self.assertEqual(torch.full(o.shape, 1., out=o).dtype, o.dtype)


class TestAdaptiveAvgPool2d(TestCase):
    def cpu_op_exec(self, input, output_size):
        m = nn.AdaptiveAvgPool2d(output_size)
        output = m(input)
        return output.numpy()

    def npu_op_exec(self, input, output_size):
        m = nn.AdaptiveAvgPool2d(output_size).npu()
        output = m(input)
        return output.cpu().numpy()

    def test_adaptiveAvgPool2d_shape_format_fp16(self, device):
        format_list = [0, 3]
        shape_list = [(32, 16, 16),
                      (16, 1024, 256),
                      (1024, 464, 11, 9),
                      (1, 2048, 15, 15)]
        shape_format = [
            [np.float16, i, j] for i in format_list for j in shape_list
        ]
        # TODO(Ascend): tbe operator has problem in precision and (x, 1) case and so on.
        output_list = [(4, 4), (3, 5), (1), (1, None), (None, 2)]
        for item in shape_format:
            cpu_input, npu_input = create_common_tensor(item, 0, 100)
            cpu_input = cpu_input.to(torch.float32)
            for output_size in output_list:
                cpu_output = self.cpu_op_exec(cpu_input, output_size)
                npu_output = self.npu_op_exec(npu_input, output_size)
                cpu_output = cpu_output.astype(np.float16)
                self.assertRtolEqual(cpu_output, npu_output)

    def test_adaptiveAvgPool2d_shape_format_fp32(self, device):
        format_list = [0, 3]
        shape_list = [(32, 16, 16),
                      (16, 1024, 256),
                      (1024, 464, 11, 9),
                      (1, 2048, 15, 15)]
        shape_format = [
            [np.float16, i, j] for i in format_list for j in shape_list
        ]
        output_list = [(4, 4), (3, 5), (1), (1, None), (None, 2)]
        for item in shape_format:
            cpu_input, npu_input = create_common_tensor(item, 0, 100)
            for output_size in output_list:
                cpu_output = self.cpu_op_exec(cpu_input, output_size)
                npu_output = self.npu_op_exec(npu_input, output_size)
                self.assertRtolEqual(cpu_output, npu_output)


class TestDropout(TestCase):
    def _test_dropout(self, cls, device, input, memory_format=torch.contiguous_format):
        p = 0.2
        input = input.to(device).fill_(1 - p)

        module = cls(p)
        input_var = input.clone(memory_format=memory_format).requires_grad_()
        output = module(input_var)
        self.assertTrue(output.is_contiguous(memory_format=memory_format))
        self.assertLess(abs(output.data.mean() - (1 - p)), 0.05)
        output.backward(input)
        self.assertTrue(input_var.grad.is_contiguous(memory_format=memory_format))
        self.assertLess(abs(input_var.grad.data.mean() - (1 - p)), 0.05)

        module = cls(p, True)
        input_var = input.clone(memory_format=memory_format).requires_grad_()
        output = module(input_var + 0)
        self.assertTrue(output.is_contiguous(memory_format=memory_format))
        self.assertLess(abs(output.data.mean() - (1 - p)), 0.05)
        output.backward(input)
        self.assertTrue(input_var.grad.is_contiguous(memory_format=memory_format))
        self.assertLess(abs(input_var.grad.data.mean() - (1 - p)), 0.05)

        # check eval mode doesn't change anything
        for inplace in [True, False]:
            module = cls(p, inplace).eval()
            self.assertEqual(input.to('cpu'), module(input))

        # Check that these don't raise errors
        module.__repr__()
        str(module)

    def test_invalid_dropout_p(self, device):
        v = torch.ones(1)
        self.assertRaises(ValueError, lambda: F.dropout(v, -0.1))
        self.assertRaises(ValueError, lambda: F.dropout(v, 1.1))

    def test_Dropout(self, device):
        input = torch.Tensor(1000)
        self._test_dropout(nn.Dropout, device, input)

    def test_empty_dropout(self, device):
        x = torch.Tensor([]).to(device)
        out = torch.nn.functional.dropout(x)
        self.assertEqual(out.size(), x.size())


class TestGt(TestCase):
    exact_dtype = True


class TestLe(TestCase):
    exact_dtype = True


class TestLt(TestCase):
    exact_dtype = True


class TestMax(TestCase):
    exact_dtype = True

    def _testCSelection(self, torchfn, mathfn):
        # Two tensors
        size = (100, 100)
        a = torch.rand(*size)
        b = torch.rand(*size)
        c = torchfn(a, b)
        expected_c = torch.zeros(*size)
        expected_c.map2_(a, b, lambda _, a, b: mathfn(a, b))
        self.assertEqual(expected_c, c, 0)

    def _testSelection(self, torchfn, mathfn):
        # contiguous
        m1 = torch.randn(100, 100)
        m1 = m1.npu()
        res1 = torchfn(m1)
        res2 = m1[0, 0]
        for i, j in iter_indices(m1):
            res2 = mathfn(res2, m1[i, j])
        self.assertEqual(res1.cpu(), res2.cpu())

        # non-contiguous
        m1 = torch.randn(10, 10, 10)
        m1 = m1.npu()
        m2 = m1[:, 4]
        res1 = torchfn(m2)
        res2 = m2[0, 0]
        for i, j in iter_indices(m2):
            res2 = mathfn(res2, m2[i][j])
        self.assertEqual(res1.cpu(), res2.cpu())

        # with indices
        m1 = torch.randn(100, 100)
        m1 = m1.npu()
        res1val, res1ind = torchfn(m1, 1, False)
        res2val = m1[:, 0:1].clone().squeeze()
        res2ind = res1ind.clone().fill_(0)
        for i, j in iter_indices(m1):
            if mathfn(res2val[i], m1[i, j]) != res2val[i]:
                res2val[i] = m1[i, j]
                res2ind[i] = j

        maxerr = 0
        for i in range(res1val.size(0)):
            maxerr = max(maxerr, abs(res1val[i] - res2val[i]))
            self.assertEqual(res1ind[i].cpu(), res2ind[i].cpu())
        self.assertLessEqual(abs(maxerr), 1e-5)

        # NaNs
        for index in (0, 4, 99):
            m1 = torch.randn(100)
            m1 = m1.npu()
            m1[index] = nan
            res1val, res1ind = torch.max(m1, 0)
            self.assertTrue(math.isnan(res1val))
            self.assertEqual(res1ind, index)
            res1val = torchfn(m1)
            self.assertTrue(math.isnan(res1val))

        # Bool
        m1 = torch.tensor([True, False, True], dtype=torch.bool)
        m1 = m1.npu()
        res1 = torchfn(m1)
        res2 = m1[0]
        for i in iter_indices(m1):
            res2 = mathfn(res2.cpu(), m1[i].cpu())
        self.assertEqual(res1.cpu(), res2.cpu())

    def test_max_1(self, device):
        self._testSelection(torch.max, max)

    @onlyCPU
    def test_max_mixed_devices(self, device):
        a = torch.randn(10, device=device)
        if torch.cuda.is_available():
            values = torch.randn(10).cuda()
            indices = torch.cuda.LongTensor()
            self.assertRaises(RuntimeError,
                              lambda: torch.max(a, 0, out=(values, indices)))

    @dtypesIfCUDA(torch.half, torch.float, torch.double)
    @dtypes(torch.float, torch.double)
    def test_max_with_inf(self, device, dtype):
        a = torch.tensor([[-inf, -inf, inf, 3], [inf, inf, -inf, -1]], dtype=dtype, device=device)
        self.assertTrue(torch.all(torch.max(a, dim=1)[0] == inf).item())
        self.assertTrue(torch.max(a).item() == inf)

    def test_max_elementwise_1(self, device):
        self._testCSelection(torch.max, max)


class TestMin(TestCase):
    def _testCSelection(self, torchfn, mathfn):
        # Two tensors
        size = (100, 100)
        a = torch.rand(*size)
        b = torch.rand(*size)
        c = torchfn(a, b)
        expected_c = torch.zeros(*size)
        expected_c.map2_(a, b, lambda _, a, b: mathfn(a, b))
        self.assertEqual(expected_c, c, 0)

    def _testSelection(self, torchfn, mathfn):
        # contiguous
        m1 = torch.randn(100, 100)
        m1 = m1.npu()
        res1 = torchfn(m1)
        res2 = m1[0, 0]
        for i, j in iter_indices(m1):
            res2 = mathfn(res2, m1[i, j])
        self.assertEqual(res1.cpu(), res2.cpu())

        # non-contiguous
        m1 = torch.randn(10, 10, 10)
        m1 = m1.npu()
        m2 = m1[:, 4]
        res1 = torchfn(m2)
        res2 = m2[0, 0]
        for i, j in iter_indices(m2):
            res2 = mathfn(res2, m2[i][j])
        self.assertEqual(res1.cpu(), res2.cpu())

        # with indices
        m1 = torch.randn(100, 100)
        m1 = m1.npu()
        res1val, res1ind = torchfn(m1, 1, False)
        res2val = m1[:, 0:1].clone().squeeze()
        res2ind = res1ind.clone().fill_(0)
        for i, j in iter_indices(m1):
            if mathfn(res2val[i], m1[i, j]) != res2val[i]:
                res2val[i] = m1[i, j]
                res2ind[i] = j

        maxerr = 0
        for i in range(res1val.size(0)):
            maxerr = max(maxerr, abs(res1val[i] - res2val[i]))
            self.assertEqual(res1ind[i].cpu(), res2ind[i].cpu())
        self.assertLessEqual(abs(maxerr), 1e-5)

        # NaNs
        for index in (0, 4, 99):
            m1 = torch.randn(100)
            m1 = m1.npu()
            m1[index] = nan
            res1val, res1ind = torch.max(m1, 0)
            self.assertTrue(math.isnan(res1val))
            self.assertEqual(res1ind, index)
            res1val = torchfn(m1)
            self.assertTrue(math.isnan(res1val))

        # Bool
        m1 = torch.tensor([True, False, True], dtype=torch.bool)
        m1 = m1.npu()
        res1 = torchfn(m1)
        res2 = m1[0]
        for i in iter_indices(m1):
            res2 = mathfn(res2.cpu(), m1[i].cpu())
        self.assertEqual(res1.cpu(), res2.cpu())

    exact_dtype = True

    @dtypesIfCUDA(torch.half, torch.float, torch.double)
    @dtypes(torch.float, torch.double)
    def test_min_with_inf(self, device, dtype):
        a = torch.tensor([[-inf, -inf, inf, 3], [inf, inf, -inf, -1]], dtype=dtype, device=device)
        self.assertTrue(torch.all(torch.min(a, dim=1)[0] == (-inf)).item())
        self.assertTrue(torch.min(a).item() == -inf)

    @dtypesIfCUDA(torch.half, torch.float, torch.double)
    @dtypes(torch.float, torch.double)
    def _test_min_max_binary_op_nan(self, device, dtype):
        a = torch.rand(1000, dtype=dtype, device=device)
        b = torch.rand(1000, dtype=dtype, device=device)

        # 0:250: a -- nan, b -- not nan
        a[:250] = float('nan')
        # 250:500: a -- not nan, b -- nan
        b[250:500] = float('nan')
        # 500:750: a and b both nan
        a[500:750] = float('nan')
        b[500:750] = float('nan')
        # 750:1000: neither nan

        ma = torch.max(a, b)
        mi = torch.min(a, b)

        for i in range(750):
            self.assertTrue(torch.isnan(ma[i]), "max(a, b): {}, a: {}, b: {}".format(ma[i], a[i], b[i]))
            self.assertTrue(torch.isnan(mi[i]), "min(a, b): {}, a: {}, b: {}".format(mi[i], a[i], b[i]))

        for i in range(750, 1000):
            self.assertFalse(torch.isnan(ma[i]), "max(a, b): {}, a: {}, b: {}".format(ma[i], a[i], b[i]))
            self.assertFalse(torch.isnan(mi[i]), "min(a, b): {}, a: {}, b: {}".format(mi[i], a[i], b[i]))

    def test_min_mixed_devices(self, device):
        a = torch.randn(10, device=device)
        values = torch.randn(10).npu()
        indices = torch.npu.LongTensor()
        self.assertRaises(RuntimeError, lambda: torch.min(a, 0, out=(values, indices)))

    def test_min_max_nan(self, device):
        tests = [(lambda x: x.min(), 'min'),
                 (lambda x: x.max(), 'max'),
                 (lambda x: x.min(0)[0], 'min_dim'),
                 (lambda x: x.max(0)[0], 'max_dim')]
        for f, name in tests:
            a = torch.arange(25.0).view(5, 5)
            a[2, 2] = nan
            actual = f(a.to(device)).cpu()
            expected = f(a).cpu()
            self.assertEqual(torch.isnan(actual), torch.isnan(expected), 'nans for {}'.format(name))
            self.assertEqual(actual[~torch.isnan(actual)],
                             expected[~torch.isnan(expected)], 'nans for {}'.format(name))

    def test_min_1(self, device):
        self._testSelection(torch.min, min)

    def test_min_elementwise_1(self, device):
        self._testCSelection(torch.min, min)


class TestNe(TestCase):
    exact_dtype = True


class TestPow(TestCase):
    exact_dtype = True


class TestMul(TestCase):
    exact_dtype = True

    def test_mul_1(self, device):
        m1 = torch.randn(10, 10, device=device)
        res1 = m1.clone()
        res1[:, 3].mul_(2)
        res2 = m1.clone()
        for i in range(res1.size(0)):
            res2[i, 3] = res2[i, 3] * 2
        self.assertEqual(res1, res2)

        a1 = torch.tensor([True, False, False, True], dtype=torch.bool, device=device)
        a2 = torch.tensor([True, False, True, False], dtype=torch.bool, device=device)
        self.assertEqual(a1 * a2, torch.tensor([True, False, False, False], dtype=torch.bool, device=device))

        if device == 'cpu':
            a1 = torch.tensor([0.1, 0.1], dtype=torch.bfloat16, device=device)
            a2 = torch.tensor([1.1, 0.1], dtype=torch.bfloat16, device=device)
            self.assertEqual(a1 * a2, torch.tensor([0.11, 0.01], dtype=torch.bfloat16, device=device), 0.01)
            self.assertEqual(a1.mul(a2), a1 * a2)

    @dtypesIfCUDA(torch.half, torch.float, torch.double)
    @dtypes(torch.float, torch.double)
    def test_mul_intertype_scalar(self, device, dtype):
        x = torch.tensor(1.5, dtype=dtype, device=device)
        y = torch.tensor(3, dtype=torch.int32, device=device)

        self.assertEqual(x * y, 4.5)
        self.assertEqual(y * x, 4.5)

        with self.assertRaisesRegex(RuntimeError, "can't be cast to the desired output type"):
            y *= x
        x *= y
        self.assertEqual(x, 4.5)


class TestProd(TestCase):
    exact_dtype = True

    @dtypes(torch.float16, torch.float32)
    def test_prod_npu(self, device, dtype):
        x = torch.tensor([2, 3, 6, 9, 8], dtype=dtype, device=device)

        # Check all combinations: fp16 input - fp16 output, fp16 input - fp32
        # output, fp32 input - fp16 output, fp32 input - fp32 output
        for dtype_output in [torch.float16, torch.float32]:
            result_expected = torch.tensor(2592, dtype=dtype_output, device=device)
            output = torch.prod(x, dtype=dtype_output)
            self.assertEqual(output.cpu(), result_expected)

            output = x.prod(dtype=dtype_output)
            self.assertEqual(output.cpu(), result_expected)

    @dtypes(torch.float)
    def test_prod_1(self, device, dtype):
        x = torch.rand(100, 100, dtype=dtype, device=device)
        res1 = torch.prod(x, 1)
        res2 = torch.tensor((), dtype=dtype, device=device)
        torch.prod(x, 1, out=res2)
        self.assertEqual(res1.cpu(), res2)

    def _test_reduce_integer_upcast(self, fn, has_out=True):
        shape = (3, 4, 5)
        reduced_shape = fn(torch.ones(shape)).shape

        def _test_out(dtype, other_dtype):
            out = torch.ones(reduced_shape, dtype=dtype)
            result = fn(x, out=out)
            self.assertIs(out.dtype, result.dtype)
            self.assertEqual(fn(x.type(dtype)), result, exact_dtype=False)
            result = fn(x, out=out, dtype=dtype)
            self.assertIs(out.dtype, result.dtype)
            self.assertEqual(fn(x.type(dtype)), result, exact_dtype=False)
            # 'out' is favored over dtype, check error
            self.assertRaises(RuntimeError, lambda: fn(x, out=out, dtype=other_dtype))

        for dtype in [dtype for dtype in torch.testing.get_all_math_dtypes('cpu') if dtype != torch.float16]:
            x = torch.ones(shape, dtype=dtype)
            expected_dtype = dtype if dtype.is_floating_point else torch.int64
            self.assertIs(expected_dtype, fn(x).dtype)
            self.assertEqual(fn(x.type(expected_dtype)), fn(x))

            if dtype.is_floating_point:
                other_dtype = torch.float32 if dtype == torch.float64 else torch.float64
            else:
                other_dtype = torch.int32 if dtype != torch.int32 else torch.int16
            self.assertIs(other_dtype, fn(x, dtype=other_dtype).dtype)
            self.assertEqual(fn(x.type(other_dtype)), fn(x, dtype=other_dtype), exact_dtype=False)

            # test mixed int/float
            mixed_dtype = torch.int32 if dtype.is_floating_point else torch.float32
            self.assertIs(mixed_dtype, fn(x, dtype=mixed_dtype).dtype)
            self.assertEqual(fn(x.type(mixed_dtype)), fn(x, dtype=mixed_dtype), exact_dtype=False)

            if has_out:
                _test_out(dtype, other_dtype)
                _test_out(dtype, mixed_dtype)

    def test_prod_integer_upcast(self, device):
        self._test_reduce_integer_upcast(lambda x, **kwargs: torch.prod(x, **kwargs), False)
        self._test_reduce_integer_upcast(lambda x, **kwargs: torch.prod(x, 0, **kwargs))


class TestRemainder(TestCase):
    exact_dtype = True

    def test_remainder_overflow(self, device):
        # Check Integer Overflows
        x = torch.tensor(23500, dtype=torch.int64, device=device)
        q = 392486996410368
        self.assertEqual(x % q, x)
        self.assertEqual(-x % q, q - x)
        self.assertEqual(x % -q, x - q)
        self.assertEqual(-x % -q, -x)

    @onlyCPU
    @dtypes(torch.float, torch.long)
    def test_remainder(self, device, dtype):
        for use_item in [True, False]:
            if dtype == torch.float:
                m1 = torch.Tensor(10, 10).uniform_(-10., 10.).to(dtype=dtype, device=device)
                res1 = m1.clone()
                res2 = m1.clone()
                qs = torch.arange(-5.1, 4.1, dtype=dtype, device=device)
                # Check the case where the divisor is a simple float
                for col_idx, q in enumerate(qs):
                    # Reference
                    for i in range(m1.size(0)):
                        res2[i, col_idx] = res2[i, col_idx] % q
                    # To test
                    res1[:, col_idx].remainder_(q if not use_item else q.item())
                self.assertEqual(res1, res2)
                # Check the case where the divisor is a tensor
                res1 = m1.clone()
                res1.remainder_(qs.unsqueeze(0).expand_as(res1))
                self.assertEqual(res1, res2)
            elif dtype == torch.long:
                long_m1 = torch.LongTensor(10, 10).random_(-10, 10)
                long_res1 = long_m1.clone()
                long_res2 = long_m1.clone()
                long_qs = torch.arange(-5, 5, dtype=dtype, device=device)
                long_qs[5] = 5  # Can't handle the divisor=0 case
                for col_idx, long_q in enumerate(long_qs):
                    # Reference
                    for i in range(long_m1.size(0)):
                        long_res2[i, col_idx] = long_res2[i, col_idx] % long_q
                    # To test
                    long_res1[:, col_idx].remainder_(long_q if not use_item else long_q.item())
                self.assertEqual(long_res1, long_res2)
                # Divisor is a tensor case
                long_res1 = long_m1.clone()
                long_res1.remainder_(long_qs.unsqueeze(0).expand_as(long_res1))

    @dtypes(torch.int64, torch.float64)
    def test_remainder_edge_cases(self, device, dtype):
        # Test variations of negative values used as input
        a = torch.tensor([6, -6, -6, 6, 27, -27, -27, 27], dtype=dtype, device=device)
        b = torch.tensor([-3, 3, -3, 3, -5, 5, -5, 5], dtype=dtype, device=device)
        r = a.remainder(b)
        r_expected = torch.tensor([0, 0, 0, 0, -3, 3, -2, 2], dtype=dtype, device=device)
        self.assertEqual(r, r_expected)

        if dtype == torch.float64:
            # Test cases where result should be nan
            a = torch.tensor([-34, 0, 34], dtype=dtype, device=device)
            b = torch.zeros(3, dtype=dtype, device=device)
            self.assertTrue(torch.isnan(a.remainder(b)).all())

            # Need to test a fairly large tensor with float cpu to run
            # the Vec256 implementation
            if device == 'cpu':
                a = torch.tensor([6, -6, -6, 6, 27, -27, -27, 27] * 10000, dtype=dtype, device=device)
                b = torch.tensor([-3, 3, -3, 3, -5, 5, -5, 5] * 10000, dtype=dtype, device=device)
                r = a.remainder(b)
                r_expected = torch.tensor([0, 0, 0, 0, -3, 3, -2, 2] * 10000, dtype=dtype, device=device)
                self.assertEqual(r, r_expected)

                # Test nan cases
                a = torch.tensor([-34, 0, 34] * 20000, dtype=dtype, device=device)
                b = torch.zeros(3 * 20000, dtype=dtype, device=device)
                self.assertTrue(torch.isnan(a.remainder(b)).all())

        elif dtype == torch.int64:
            if device == 'cpu':
                # Test int divide by zero causes an exception
                a = torch.ones(1000, dtype=dtype, device=device)
                b = torch.ones(1000, dtype=dtype, device=device)
                b[500] = 0
                self.assertRaises(RuntimeError, lambda: a.remainder(b))

        # Check scalar type is promoted to match tensor
        a = torch.ones(1, dtype=dtype, device=device)
        b = 1.0 if dtype == torch.int64 else 1
        r = a.remainder(b)
        self.assertEqual(r.dtype, a.dtype)


class TestNonzero(TestCase):
    exact_dtype = True

    def test_nonzero_empty(self, device):
        def assert_tuple_empty(tup, dim):
            self.assertEqual(dim, len(tup))
            for t in tup:
                self.assertEqual(torch.Size([0]), t.shape)

        x = torch.randn(0, 2, 0, 5, 0, device=device)
        y = torch.nonzero(x)
        z = torch.nonzero(x, as_tuple=True)

        self.assertEqual(0, y.numel())
        self.assertEqual(torch.Size([0, 5]), y.shape)
        assert_tuple_empty(z, 5)

        x = torch.tensor(0.5, device=device)
        y = torch.nonzero(x)
        # nonzero with as_tuple returns a
        # tuple of len 1 for a zero-dim tensor.
        # This is done to match Numpy behavior.
        z = torch.nonzero(x, as_tuple=True)
        self.assertEqual(1, len(z))
        self.assertEqual(torch.zeros(1, dtype=torch.long), z[0])

        x = torch.zeros((), device=device)
        y = torch.nonzero(x)
        z = torch.nonzero(x, as_tuple=True)
        self.assertEqual(torch.Size([0, 0]), y.shape)
        self.assertEqual(1, len(z))
        self.assertEqual(torch.empty(0, dtype=torch.long), z[0])

    def test_nonzero_1(self, device):
        num_srcs = [
            12, 12, 12, 12, 12, 125,
        ]

        types = [
            'torch.ByteTensor',
            'torch.CharTensor',
            'torch.ShortTensor',
            'torch.IntTensor',
            'torch.FloatTensor',
            'torch.DoubleTensor',
            'torch.LongTensor',
        ]

        shapes = [
            torch.Size((12,)),
            torch.Size((12, 1)),
            torch.Size((1, 12)),
            torch.Size((6, 2)),
            torch.Size((3, 2, 2)),
            torch.Size((5, 5, 5)),
        ]

        def is_lexicographically_sorted(inds):
            """Check sorted ascending with
            i -> j -> k changing slowest to fastest"""
            assert inds.size(1) == 3
            if inds.size(0) > 1:
                i0, j0, k0 = inds[:-1].t()
                i1, j1, k1 = inds[+1:].t()
                i_ok = (i1 >= i0)
                j_ok = (j1 >= j0) | (i1 > i0)
                k_ok = (k1 >= k0) | (j1 > j0) | (i1 > i0)
                lex = torch.stack((i_ok, j_ok, k_ok), dim=1)
                return lex
            return torch.full_like(inds, 1)

        def gen_nontrivial_input(num_src, dtype, device):
            while True:
                tensor = torch.rand(num_src).mul(2).floor().type(dtype).to(device)
                if tensor.sum() > 0:
                    return tensor

        for dtype in types:
            for shape, num_src in zip(shapes, num_srcs):
                tensor = gen_nontrivial_input(num_src, dtype, device)
                tensor = tensor.clone().resize_(shape)
                dst1 = torch.nonzero(tensor)
                dst2 = tensor.nonzero()
                dst3 = torch.LongTensor().to(device)
                torch.nonzero(tensor, out=dst3)

                self.assertRaisesRegex(
                    TypeError,
                    "received an invalid combination of arguments",
                    lambda: torch.nonzero(tensor, as_tuple=True, out=dst3))
                if len(shape) == 1:
                    dst = []
                    for i in range(num_src):
                        if tensor[i] != 0:
                            dst += [i]
                    dst = torch.LongTensor(dst).to(device)
                    self.assertEqual(dst1.select(1, 0), dst, 0)
                    self.assertEqual(dst2.select(1, 0), dst, 0)
                    self.assertEqual(dst3.select(1, 0), dst, 0)
                elif len(shape) == 2:
                    # This test will allow through some False positives. It only checks
                    # that the elements flagged positive are indeed non-zero.
                    for i in range(dst1.size(0)):
                        self.assertNotEqual(tensor[dst1[i, 0], dst1[i, 1]].item(), 0)
                elif len(shape) == 3:
                    # This test will allow through some False positives. It only checks
                    # that the elements flagged positive are indeed non-zero.
                    for i in range(dst1.size(0)):
                        self.assertNotEqual(tensor[dst1[i, 0], dst1[i, 1], dst1[i, 2]].item(), 0)
                    lex = is_lexicographically_sorted(dst1)
                    self.assertEqual(torch.ones_like(lex), lex)
                if TEST_NUMPY:
                    tup1 = torch.nonzero(tensor, as_tuple=True)
                    tup2 = tensor.nonzero(as_tuple=True)
                    tup3 = torch.where(tensor)
                    np1 = tensor.cpu().numpy().nonzero()
                    for t in (tup1, tup2, tup3):
                        self.assertEqual(len(t), len(np1))
                        for i in range(len(t)):
                            self.assertEqual(t[i].cpu().numpy(), np1[i])

    def test_nonzero_non_diff(self, device):
        x = torch.randn(10, requires_grad=True)
        nz = x.nonzero()
        self.assertFalse(nz.requires_grad)


class TestLog(TestCase):
    exact_dtype = True

    def test_log_1(self, device):
        def log(x):
            if x == 0:
                return -inf
            elif x < 0:
                return nan
            return math.log(x)

        _test_math(torch.log, log)


class TestLog2(TestCase):
    exact_dtype = True

    def test_log2_1(self, device):
        def log2(x):
            if x == 0:
                return -inf
            elif x < 0:
                return nan
            try:
                return math.log2(x)
            except AttributeError:
                return math.log(x, 2)

        _test_math(torch.log2, log2)


class TestNeg(TestCase):
    exact_dtype = True

    def test_neg_1(self, device):
        int_types = [torch.int, torch.short, torch.int8, torch.uint8]
        float_types = [torch.float, torch.double, torch.long]

        # Tests bool tensor negation raises the correct error
        self.assertRaisesRegex(
            RuntimeError,
            r"Negation, the `\-` operator, on a bool tensor is not supported. "
            r"If you are trying to invert a mask, use the `\~` or `logical_not\(\)` operator instead.",
            lambda: - torch.tensor([False, True], device=device))

        for dtype in float_types + int_types:
            if dtype in float_types:
                a = torch.randn(100, 90).type(dtype).to(device)
            if dtype == torch.uint8:
                a = torch.randint(0, 256, (100, 90), dtype=dtype, device=device)
            else:
                a = torch.randint(-128, 128, (100, 90), dtype=dtype, device=device)
            zeros = torch.Tensor().type(dtype).resize_as_(a).zero_().to(device)

            if dtype == torch.uint8:
                res_add = torch.add(zeros, a, alpha=255)
            else:
                res_add = torch.add(zeros, a, alpha=-1)

            res_neg = a.clone()
            res_neg.neg_()
            self.assertEqual(res_neg, res_add)

            # test out of place as well
            res_neg_out_place = a.clone().neg()
            self.assertEqual(res_neg_out_place, res_add)

            # test via __neg__ operator
            res_neg_op = -a.clone()
            self.assertEqual(res_neg_op, res_add)


class TestMaskedFill(TestCase):
    def test_masked_fill(self, device):
        with warnings.catch_warnings(record=True) as w:
            for dt in torch.testing.get_all_dtypes():
                for dtype in [torch.uint8]:
                    num_dest = 10
                    dst = torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=dt)
                    mask = torch.rand(num_dest).mul(2).floor().to(dtype)
                    val = random.random()
                    dst2 = dst.clone()
                    dst2 = dst2.npu()

                    if dt == torch.half:
                        self.assertRaises(RuntimeError, lambda: dst.masked_fill_(mask, val))
                        continue

                    dst.masked_fill_(mask, val)
                    for i in range(num_dest):
                        if mask[i]:
                            dst2[i] = val
                    self.assertEqual(dst, dst2, 0)

                    # test non-contiguous case
                    dst = torch.randn(num_dest, num_dest, num_dest).permute((2, 0, 1))
                    dst2 = dst.clone()
                    dst2 = dst2.npu()
                    dst.masked_fill_((dst > 0).to(dtype), val)
                    dst2.masked_fill_((dst2 > 0).to(dtype), val)
                    self.assertEqual(dst, dst2, 0)
            # Only 27 (not 28) here as the warning in the assertRaises are not caught on the python side
            self.assertEqual(len(w), 27)

            warn = 'masked_fill_ received a mask with dtype torch.uint8,'
            for wi in w:
                self.assertEqual(str(wi.message)[0:52], str(warn))

    def test_masked_fill_bool_tensor(self, device):
        dst = torch.tensor([True, False, True], device=device)
        mask = torch.tensor([False, True, False], device=device)

        dst.masked_fill_(mask, True)
        self.assertEqual(dst, torch.tensor([True, True, True], device=device))

        dst = dst.masked_fill(mask, False)
        self.assertEqual(dst, torch.tensor([True, False, True], device=device))


class TestMaskedScatter(TestCase):
    def test_masked_scatter(self, device):
        with warnings.catch_warnings(record=True) as w:
            for maskType in [torch.uint8, torch.bool]:
                for dt in torch.testing.get_all_dtypes():
                    num_copy, num_dest = 3, 10
                    dest = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=dt)
                    dest2 = dest.clone()
                    src = torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=dt)
                    mask = torch.tensor((0, 0, 0, 0, 1, 0, 1, 0, 1, 0), dtype=maskType)

                    if dt == torch.bool:
                        # torch.bool is a special case and is being tested
                        # in a separate test
                        continue

                    if dt == torch.half:
                        self.assertRaises(RuntimeError, lambda: dest.masked_scatter_(mask, src))
                        continue

                    dest.masked_scatter_(mask, src)
                    j = 0
                    for i in range(num_dest):
                        if mask[i]:
                            dest2[i] = src[j]
                            j += 1
                    self.assertEqual(dest, dest2, 0)

                    # make source bigger than number of 1s in mask
                    src = torch.tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=dt)
                    dest.masked_scatter_(mask, src)

                    # make src smaller. this should fail
                    src = torch.randn(num_copy - 1)
                    with self.assertRaises(RuntimeError):
                        dest.masked_scatter_(mask, src)
        # Only 16 (not 25) here as the warnings in the assertRaises are not caught on the python side
        self.assertEqual(len(w), 16)

        warn = 'masked_scatter_ received a mask with dtype torch.uint8,'
        for wi in w:
            self.assertEqual(str(wi.message)[0:55], str(warn))

    def test_masked_scatter_bool_tensor(self, device):
        src = torch.tensor([True, True, True], device=device)
        dst = torch.tensor([False, False, False], device=device)
        mask = torch.tensor([False, True, False], device=device)

        dst.masked_scatter_(mask, src)
        self.assertEqual(dst, torch.tensor([False, True, False], device=device))

        mask = torch.tensor([True, False, True], device=device)
        dst = dst.masked_scatter(mask, src)
        self.assertEqual(dst, torch.tensor([True, True, True], device=device))


class TestMaskedSelect(TestCase):
    def test_masked_select(self, device):
        warn = 'masked_select received a mask with dtype torch.uint8,'
        for dt in torch.testing.get_all_dtypes():
            with warnings.catch_warnings(record=True) as w:
                for maskType in [torch.uint8, torch.bool]:
                    num_src = 10
                    src = torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=dt, device=device)
                    mask = torch.rand(num_src, device=device).clamp(0, 1).mul(2).floor().to(maskType)

                    if dt == torch.half and torch.device(device).type == 'cpu':
                        self.assertRaises(RuntimeError, lambda: src.masked_select(mask))
                        continue

                    dst = src.masked_select(mask)
                    dst2 = []
                    for i in range(num_src):
                        if mask[i]:
                            dst2 += [src[i]]
                    self.assertEqual(dst, torch.tensor(dst2), 0)

                    dst3 = torch.empty_like(src, device=device)
                    torch.masked_select(src, mask, out=dst3)
                    self.assertEqual(dst3, torch.tensor(dst2, dtype=dst3.dtype), 0)
                    if maskType is torch.uint8:
                        self.assertEqual(len(w), 1)
                        self.assertEqual(str(w[0].message)[0:53], str(warn))


class TestMatMul(TestCase):
    def check_single_matmul(self, x, y, shape):
        a = np.array(x, copy=False)
        b = np.array(y, copy=False)
        expected = np.matmul(a, b)

        ans = torch.matmul(x, y)
        self.assertTrue(ans.is_contiguous())
        self.assertTrue(np.array_equal(ans, expected))

        out = torch.zeros(*shape, dtype=torch.int64)
        ans = torch.matmul(x, y, out=out)
        self.assertIs(ans, out)
        self.assertTrue(ans.is_contiguous())
        self.assertTrue(np.array_equal(ans, expected))

    @unittest.skipIf(not TEST_NUMPY, "Numpy not found")
    def test_matmul_small_brute_force_1d_Nd(self, device):
        # Issue #20452: range(0, 10) does not work.
        n = 1
        for m in range(1, 8):
            for p in range(1, 8):
                for o in range(1, 5):
                    # 1d, 3d, inner dimensions C
                    x = torch.arange(m)
                    y = torch.arange(o * m * p).reshape(o, m, p)
                    self.check_single_matmul(x, y, (o, n, p))

                    # 1d, 3d, inner dimensions Fortran
                    x = torch.arange(m)
                    y = torch.arange(o * p * m).reshape(o, p, m).transpose(-1, -2)
                    self.check_single_matmul(x, y, (o, n, p))

                    # 1d, 3d, inner dimensions non-contiguous
                    x = torch.arange(2 * m)[::2]
                    y = torch.arange(o * m * 2 * p).reshape(o, m, 2 * p)[:, :, ::2]
                    self.check_single_matmul(x, y, (o, n, p))

                    for r in range(1, 5):
                        # 1d, 4d, inner dimensions C
                        x = torch.arange(m)
                        y = torch.arange(r * o * m * p).reshape(r, o, m, p)
                        self.check_single_matmul(x, y, (r, o, n, p))

                        # 1d, 4d, inner dimensions Fortran
                        x = torch.arange(m)
                        y = torch.arange(r * o * p * m).reshape(r, o, p, m).transpose(-1, -2)
                        self.check_single_matmul(x, y, (r, o, n, p))

                        # 1d, 4d, inner dimensions non-contiguous
                        x = torch.arange(2 * m)[::2]
                        y = torch.arange(r * o * m * 2 * p).reshape(r, o, m, 2 * p)[:, :, :, ::2]
                        self.check_single_matmul(x, y, (r, o, n, p))

    @unittest.skipIf(not TEST_NUMPY, "Numpy not found")
    def test_matmul_small_brute_force_2d_Nd(self, device):
        # Issue #20452: range(0, 10) does not work.
        for n in range(1, 5):
            for m in range(1, 5):
                for p in range(1, 5):
                    for o in range(1, 3):
                        # 2d, 3d, inner dimensions C
                        x = torch.arange(n * m).reshape(n, m)
                        y = torch.arange(o * m * p).reshape(o, m, p)
                        self.check_single_matmul(x, y, (o, n, p))

                        # 2d, 3d, inner dimensions Fortran
                        x = torch.arange(m * n).reshape(m, n).transpose(-1, -2)
                        y = torch.arange(o * p * m).reshape(o, p, m).transpose(-1, -2)
                        self.check_single_matmul(x, y, (o, n, p))

                        # 2d, 3d, inner dimensions non-contiguous
                        x = torch.arange(n * 2 * m).reshape(n, 2 * m)[:, ::2]
                        y = torch.arange(o * m * 2 * p).reshape(o, m, 2 * p)[:, :, ::2]
                        self.check_single_matmul(x, y, (o, n, p))

                        for r in range(1, 2):
                            # 2d, 4d, inner dimensions C
                            x = torch.arange(n * m).reshape(n, m)
                            y = torch.arange(r * o * m * p).reshape(r, o, m, p)
                            self.check_single_matmul(x, y, (r, o, n, p))

                            # 2d, 4d, inner dimensions Fortran
                            x = torch.arange(m * n).reshape(m, n).transpose(-1, -2)
                            y = torch.arange(r * o * p * m).reshape(r, o, p, m).transpose(-1, -2)
                            self.check_single_matmul(x, y, (r, o, n, p))

                            # 2d, 4d, inner dimensions non-contiguous
                            x = torch.arange(n * 2 * m).reshape(n, 2 * m)[:, ::2]
                            y = torch.arange(r * o * m * 2 * p).reshape(r, o, m, 2 * p)[:, :, :, ::2]
                            self.check_single_matmul(x, y, (r, o, n, p))


class TestMM(TestCase):
    @slowTest
    def test_mm(self, device):
        def _test_mm(n, m, p, dtype, genf):
            # helper function
            def matrixmultiply(mat1, mat2):
                n = mat1.size(0)
                m = mat1.size(1)
                p = mat2.size(1)
                res = torch.zeros(n, p, dtype=dtype, device=device)
                for i, j in iter_indices(res):
                    res[i, j] = sum(mat1[i, k] * mat2[k, j] for k in range(m))
                return res

            # contiguous case
            mat1 = genf(n, m)
            mat2 = genf(m, p)
            res = torch.mm(mat1, mat2)

            res2 = matrixmultiply(mat1, mat2)
            self.assertEqual(res.cpu(), res2)

            # non contiguous case 1
            mat1 = genf(n, m)
            mat2 = genf(p, m).t()
            res = torch.mm(mat1, mat2)

            res2 = matrixmultiply(mat1, mat2)
            self.assertEqual(res.cpu(), res2)

            # non contiguous case 2
            mat1 = genf(m, n).t()
            mat2 = genf(m, p)
            res = torch.mm(mat1, mat2)

            res2 = matrixmultiply(mat1, mat2)
            self.assertEqual(res.cpu(), res2)

            # non contiguous case 3
            mat1 = genf(m, n).t()
            mat2 = genf(p, m).t()
            res = torch.mm(mat1, mat2)

            res2 = matrixmultiply(mat1, mat2)
            self.assertEqual(res.cpu(), res2)

            # test with zero stride
            mat1 = genf(n, m)
            mat2 = genf(m, 1).expand(m, p)
            res = torch.mm(mat1, mat2)

            res2 = matrixmultiply(mat1, mat2)
            self.assertEqual(res.cpu(), res2)

            # explicitly exercise the _out variant in torch.mm().
            # contiguous case
            mat1 = genf(n, m)
            mat2 = genf(m, p)
            res = genf(n, p)
            torch.mm(mat1, mat2, out=res)

            res2 = matrixmultiply(mat1, mat2)
            self.assertEqual(res.cpu(), res2)

            # explicitly exercise the _out variant in torch.mm().
            # non contiguous case 3
            mat1 = genf(m, n).t()
            mat2 = genf(p, m).t()
            res = genf(n, p)
            torch.mm(mat1, mat2, out=res)

            res2 = matrixmultiply(mat1, mat2)
            self.assertEqual(res.cpu(), res2)

        for (n, m, p) in [(20, 10, 5), (15, 5, 10), (5, 18, 10)]:
            _test_mm(n, m, p, torch.float32, lambda x, y: torch.randn(x, y, dtype=torch.float32, device=device))
            # _test_mm(n, m, p, torch.float64, lambda x, y: torch.randn(x, y, dtype=torch.float64, device=device))
            _test_mm(n, m, p, torch.int32, lambda x, y: torch.randint(0, 100, (x, y), dtype=torch.int32, device=device))
            # _test_mm(n, m, p, torch.int64, lambda x, y: torch.randint(0, 100, (x, y), dtype=torch.int64, device=device))
            # _test_mm(n, m, p, torch.bfloat16,
            # lambda x, y: torch.randn(x, y, dtype=torch.float32, device=device).bfloat16())


class TestOneslike(TestCase):
    def test_ones_like(self, device):
        expected = torch.ones(100, 100)

        res1 = torch.ones_like(expected)
        self.assertEqual(res1, expected)

        # test boolean tensor
        expected = torch.tensor([True, True], dtype=torch.bool)
        res1 = torch.ones_like(expected)
        self.assertEqual(res1, expected)

    def test_ones_like_1(self, device):
        expected = torch.ones(100, 100, device=device)

        res1 = torch.ones_like(expected)
        self.assertEqual(res1, expected)

    @deviceCountAtLeast(2)
    def test_ones_like_multiple_device(self, devices):
        expected = torch.ones(100, 100, device=devices[0])
        x = torch.randn(100, 100, device=devices[1], dtype=torch.float32)
        output = torch.ones_like(x)
        self.assertEqual(output, expected)


class TestRandom(TestCase):
    def test_random_neg_values(self, device):
        signed_types = ['torch.DoubleTensor', 'torch.FloatTensor', 'torch.LongTensor',
                        'torch.IntTensor', 'torch.ShortTensor']
        for tname in signed_types:
            res = torch.rand(SIZE, SIZE).type(tname).to(device)
            res.random_(-10, -1)
            self.assertLessEqual(res.max().item(), 9)
            self.assertGreaterEqual(res.min().item(), -10)

    @dtypes(torch.float, torch.double, torch.int8, torch.int16, torch.int32, torch.int64)
    def test_random(self, device, dtype):
        # This test is flaky with p<=(2/(ub-lb))^200=6e-36
        t = torch.empty(200, dtype=dtype, device=device)
        lb = 1
        ub = 4

        t.fill_(-1)
        t.random_(lb, ub)
        self.assertEqual(t.min(), lb)
        self.assertEqual(t.max(), ub - 1)

        t.fill_(-1)
        t.random_(ub)
        self.assertEqual(t.min(), 0)
        self.assertEqual(t.max(), ub - 1)

    def test_random_bool(self, device):
        size = 2000
        t = torch.empty(size, dtype=torch.bool, device=device)

        t.fill_(False)
        t.random_()
        self.assertEqual(t.min(), False)
        self.assertEqual(t.max(), True)
        self.assertTrue(0.4 < (t.eq(True)).to(torch.int).sum().item() / size < 0.6)

        t.fill_(True)
        t.random_()
        self.assertEqual(t.min(), False)
        self.assertEqual(t.max(), True)
        self.assertTrue(0.4 < (t.eq(True)).to(torch.int).sum().item() / size < 0.6)

    def test_random_from_to_bool(self, device):
        size = 2000

        int64_min_val = torch.iinfo(torch.int64).min
        int64_max_val = torch.iinfo(torch.int64).max

        min_val = 0
        max_val = 1

        froms = [int64_min_val, -42, min_val - 1, min_val, max_val, max_val + 1, 42]
        tos = [-42, min_val - 1, min_val, max_val, max_val + 1, 42, int64_max_val]

        for from_ in froms:
            for to_ in tos:
                t = torch.empty(size, dtype=torch.bool, device=device)
                if to_ > from_:
                    if not (min_val <= from_ <= max_val) or not (min_val <= (to_ - 1) <= max_val):
                        if not (min_val <= from_ <= max_val):
                            self.assertWarnsRegex(
                                lambda: t.random_(from_, to_),
                                "from is out of bounds"
                            )
                        if not (min_val <= (to_ - 1) <= max_val):
                            self.assertWarnsRegex(
                                lambda: t.random_(from_, to_),
                                "to - 1 is out of bounds"
                            )
                    else:
                        t.random_(from_, to_)
                        range_ = to_ - from_
                        delta = 1
                        self.assertTrue(from_ <= t.to(torch.int).min() < (from_ + delta))
                        self.assertTrue((to_ - delta) <= t.to(torch.int).max() < to_)
                else:
                    self.assertRaisesRegex(
                        RuntimeError,
                        "random_ expects 'from' to be less than 'to', but got from=" + str(from_) + " >= to=" + str(
                            to_),
                        lambda: t.random_(from_, to_)
                    )

    @dtypes(torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64,
            torch.float, torch.double, torch.half, torch.bfloat16)
    @dtypesIfCUDA(torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64,
                  torch.float, torch.double, torch.half, torch.bfloat16)
    def test_random_full_range(self, device, dtype):
        # TODO: https://github.com/pytorch/pytorch/issues/33793
        if IS_WINDOWS and device.startswith('cuda') and dtype == torch.bfloat16:
            return

        size = 2000
        alpha = 0.1

        int64_min_val = torch.iinfo(torch.int64).min
        int64_max_val = torch.iinfo(torch.int64).max

        t = torch.empty(size, dtype=dtype, device=device)

        if dtype in [torch.float, torch.double, torch.half]:
            from_ = int(max(torch.finfo(dtype).min, int64_min_val))
            to_inc_ = int(min(torch.finfo(dtype).max, int64_max_val))
        elif dtype == torch.bfloat16:
            from_ = int(max(-3.389531389251535e+38, int64_min_val))
            to_inc_ = int(min(3.389531389251535e+38, int64_max_val))
        else:
            from_ = int(max(torch.iinfo(dtype).min, int64_min_val))
            to_inc_ = int(min(torch.iinfo(dtype).max, int64_max_val))
        range_ = to_inc_ - from_ + 1

        t.random_(from_, None)
        delta = max(1, alpha * range_)
        self.assertTrue(from_ <= t.to(torch.double).min() < (from_ + delta))
        self.assertTrue((to_inc_ - delta) < t.to(torch.double).max() <= to_inc_)

    @dtypes(torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64,
            torch.float, torch.double, torch.half, torch.bfloat16)
    @dtypesIfCUDA(torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64,
                  torch.float, torch.double, torch.half, torch.bfloat16)
    def test_random_from_to(self, device, dtype):
        # TODO: https://github.com/pytorch/pytorch/issues/33793
        if IS_WINDOWS and device.startswith('cuda') and dtype == torch.bfloat16:
            return

        size = 2000
        alpha = 0.1

        int64_min_val = torch.iinfo(torch.int64).min
        int64_max_val = torch.iinfo(torch.int64).max

        if dtype in [torch.float, torch.double, torch.half]:
            min_val = int(max(torch.finfo(dtype).min, int64_min_val))
            max_val = int(min(torch.finfo(dtype).max, int64_max_val))
            froms = [min_val, -42, 0, 42]
            tos = [-42, 0, 42, max_val >> 1]
        elif dtype == torch.bfloat16:
            min_val = int64_min_val
            max_val = int64_max_val
            froms = [min_val, -42, 0, 42]
            tos = [-42, 0, 42, max_val >> 1]
        elif dtype == torch.uint8:
            min_val = torch.iinfo(dtype).min
            max_val = torch.iinfo(dtype).max
            froms = [int64_min_val, -42, min_val - 1, min_val, 42, max_val, max_val + 1]
            tos = [-42, min_val - 1, min_val, 42, max_val, max_val + 1, int64_max_val]
        elif dtype == torch.int64:
            min_val = int64_min_val
            max_val = int64_max_val
            froms = [min_val, -42, 0, 42]
            tos = [-42, 0, 42, max_val]
        else:
            min_val = torch.iinfo(dtype).min
            max_val = torch.iinfo(dtype).max
            froms = [int64_min_val, min_val - 1, min_val, -42, 0, 42, max_val, max_val + 1]
            tos = [min_val - 1, min_val, -42, 0, 42, max_val, max_val + 1, int64_max_val]

        for from_ in froms:
            for to_ in tos:
                t = torch.empty(size, dtype=dtype, device=device)
                if to_ > from_:
                    if not (min_val <= from_ <= max_val) or not (min_val <= (to_ - 1) <= max_val):
                        if not (min_val <= from_ <= max_val):
                            self.assertWarnsRegex(
                                lambda: t.random_(from_, to_),
                                "from is out of bounds"
                            )
                        if not (min_val <= (to_ - 1) <= max_val):
                            self.assertWarnsRegex(
                                lambda: t.random_(from_, to_),
                                "to - 1 is out of bounds"
                            )
                    else:
                        t.random_(from_, to_)
                        range_ = to_ - from_
                        delta = max(1, alpha * range_)
                        if dtype == torch.bfloat16:
                            # Less strict checks because of rounding errors
                            # TODO investigate rounding errors
                            self.assertTrue(from_ <= t.to(torch.double).min() < (from_ + delta))
                            self.assertTrue((to_ - delta) < t.to(torch.double).max() <= to_)
                        else:
                            self.assertTrue(from_ <= t.to(torch.double).min() < (from_ + delta))
                            self.assertTrue((to_ - delta) <= t.to(torch.double).max() < to_)
                else:
                    self.assertRaisesRegex(
                        RuntimeError,
                        "random_ expects 'from' to be less than 'to', but got from=" + str(from_) + " >= to=" + str(
                            to_),
                        lambda: t.random_(from_, to_)
                    )

    @dtypes(torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64,
            torch.float, torch.double, torch.half, torch.bfloat16)
    @dtypesIfCUDA(torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64,
                  torch.float, torch.double, torch.half, torch.bfloat16)
    def test_random_to(self, device, dtype):
        # TODO: https://github.com/pytorch/pytorch/issues/33793
        if IS_WINDOWS and device.startswith('cuda') and dtype == torch.bfloat16:
            return

        size = 2000
        alpha = 0.1

        int64_min_val = torch.iinfo(torch.int64).min
        int64_max_val = torch.iinfo(torch.int64).max

        if dtype in [torch.float, torch.double, torch.half]:
            min_val = int(max(torch.finfo(dtype).min, int64_min_val))
            max_val = int(min(torch.finfo(dtype).max, int64_max_val))
            tos = [-42, 0, 42, max_val >> 1]
        elif dtype == torch.bfloat16:
            min_val = int64_min_val
            max_val = int64_max_val
            tos = [-42, 0, 42, max_val >> 1]
        elif dtype == torch.uint8:
            min_val = torch.iinfo(dtype).min
            max_val = torch.iinfo(dtype).max
            tos = [-42, min_val - 1, min_val, 42, max_val, max_val + 1, int64_max_val]
        elif dtype == torch.int64:
            min_val = int64_min_val
            max_val = int64_max_val
            tos = [-42, 0, 42, max_val]
        else:
            min_val = torch.iinfo(dtype).min
            max_val = torch.iinfo(dtype).max
            tos = [min_val - 1, min_val, -42, 0, 42, max_val, max_val + 1, int64_max_val]

        from_ = 0
        for to_ in tos:
            t = torch.empty(size, dtype=dtype, device=device)
            if to_ > from_:
                if not (min_val <= (to_ - 1) <= max_val):
                    self.assertWarnsRegex(
                        lambda: t.random_(to_),
                        "to - 1 is out of bounds"
                    )
                else:
                    t.random_(to_)
                    range_ = to_ - from_
                    delta = max(1, alpha * range_)
                    if dtype == torch.bfloat16:
                        # Less strict checks because of rounding errors
                        # TODO investigate rounding errors
                        self.assertTrue(from_ <= t.to(torch.double).min() < (from_ + delta))
                        self.assertTrue((to_ - delta) < t.to(torch.double).max() <= to_)
                    else:
                        self.assertTrue(from_ <= t.to(torch.double).min() < (from_ + delta))
                        self.assertTrue((to_ - delta) <= t.to(torch.double).max() < to_)
            else:
                self.assertRaisesRegex(
                    RuntimeError,
                    "random_ expects 'from' to be less than 'to', but got from=" + str(from_) + " >= to=" + str(to_),
                    lambda: t.random_(from_, to_)
                )

    @dtypes(torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64,
            torch.float, torch.double, torch.half, torch.bfloat16)
    @dtypesIfCUDA(torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64,
                  torch.float, torch.double, torch.half, torch.bfloat16)
    def test_random_default(self, device, dtype):
        # TODO: https://github.com/pytorch/pytorch/issues/33793
        if IS_WINDOWS and device.startswith('cuda') and dtype == torch.bfloat16:
            return

        size = 2000
        alpha = 0.1

        if dtype == torch.float:
            to_inc = 1 << 24
        elif dtype == torch.double:
            to_inc = 1 << 53
        elif dtype == torch.half:
            to_inc = 1 << 11
        elif dtype == torch.bfloat16:
            to_inc = 1 << 8
        else:
            to_inc = torch.iinfo(dtype).max

        t = torch.empty(size, dtype=dtype, device=device)
        t.random_()
        self.assertTrue(0 <= t.to(torch.double).min() < alpha * to_inc)
        self.assertTrue((to_inc - alpha * to_inc) < t.to(torch.double).max() <= to_inc)


class TestRandperm(TestCase):
    @slowTest
    def test_randperm(self, device):
        if device == 'cpu':
            rng_device = None
        else:
            rng_device = [device]

        # Test core functionality. On CUDA, for small n, randperm is offloaded to CPU instead. For large n, randperm is
        # executed on GPU.
        for n in (100, 50000, 100000):
            # Ensure both integer and floating-point numbers are tested. Half follows an execution path that is
            # different from others on CUDA.
            for dtype in (torch.long, torch.half, torch.float):
                if n > 2049 and dtype == torch.half:  # Large n for torch.half will raise an exception, do not test here.
                    continue
                with torch.random.fork_rng(devices=rng_device):
                    res1 = torch.randperm(n, dtype=dtype, device=device)
                res2 = torch.empty(0, dtype=dtype, device=device)
                torch.randperm(n, out=res2, dtype=dtype, device=device)
                self.assertEqual(res1, res2, 0)

        # Default type is long
        for n in (100, 10000):
            self.assertEqual(torch.randperm(n, device=device).dtype, torch.long)

        # randperm of 0 elements is an empty tensor
        res1 = torch.randperm(0)
        res2 = torch.tensor(5, dtype=dtype, device=device)
        torch.randperm(0, out=res2)
        self.assertEqual(res1.numel(), 0)
        self.assertEqual(res2.numel(), 0)

        # Test exceptions when n is too large for a floating point type
        for dtype, small_n, large_n in ((torch.half, 2 ** 11 + 1, 2 ** 11 + 2),
                                        (torch.float, 2 ** 24 + 1, 2 ** 24 + 2),
                                        (torch.double, 2 ** 25,  # 2**53 + 1 is too large to run
                                         2 ** 53 + 2)):
            res = torch.empty(0, dtype=dtype, device=device)
            torch.randperm(small_n, out=res)  # No exception expected
            self.assertRaises(RuntimeError, lambda: torch.randperm(large_n, out=res, device=device))

        # Test non-contiguous tensors
        for n in (4, 5, 6, 10, 20):
            non_contiguous_tensor = torch.zeros((2, 3), dtype=torch.long, device=device).t()
            self.assertFalse(non_contiguous_tensor.is_contiguous())
            with torch.random.fork_rng(devices=rng_device):
                res = torch.randperm(n, dtype=torch.long, device=device)
            torch.randperm(n, out=non_contiguous_tensor)
            self.assertEqual(non_contiguous_tensor, res)


class TestReciprocal(TestCase):
    @onlyCPU
    @dtypes(torch.float, torch.double)
    def test_reciprocal(self, device, dtype):
        a = torch.randn(100, 89, device=device, dtype=dtype)
        res_div = 1 / a
        res_reciprocal = a.clone()
        res_reciprocal.reciprocal_()
        self.assertEqual(res_reciprocal, res_div)


class TestHardtanh(TestCase):
    def test_hardtanh_inplace_gradgrad(self, device):
        v = torch.randn(8, requires_grad=True)

        def func(root):
            x = root.clone()
            return F.hardtanh(x, inplace=True)

        gradcheck(func, [v])
        gradgradcheck(func, [v])

    # test hardtanh backward froo large tensor
    def test_hardtanh_backward(self, device):
        x = torch.randn(128, 10000, requires_grad=True)
        grad = torch.randn(128, 10000)
        z = torch.zeros(128, 10000)
        y = F.hardtanh(x)
        y.backward(grad)
        # ref backward path for hardtanh
        mask = (x > -1) & (x < 1)
        x_grad_ref = torch.where(mask, grad, z)
        self.assertEqual(x.grad, x_grad_ref)


class TestMaxPool(TestCase):
    @onlyCUDA
    @dtypesIfCUDA(torch.half, torch.float, torch.double)
    def test_max_pool2d_nhwc(self, device, dtype):
        def helper(n, c, h, w, kernel_size, stride=None):
            if stride is None:
                stride = kernel_size
            input = torch.randn(n, c, h, w, dtype=dtype, device=device)
            input = input.contiguous(memory_format=torch.channels_last).requires_grad_()
            grad = torch.randn(n, c, (h - kernel_size) // stride + 1, (w - kernel_size) // stride + 1,
                               dtype=dtype, device=device)
            pool = torch.nn.MaxPool2d(kernel_size, stride).to(device)

            ref_input = input.detach().clone().contiguous().requires_grad_(True)
            ref_grad = grad.detach().clone().contiguous()
            ref_pool = torch.nn.MaxPool2d(kernel_size, stride).to(device)

            out = pool(input)
            out.backward(grad)
            ref_out = ref_pool(ref_input)
            ref_out.backward(ref_grad)

            self.assertTrue(out.is_contiguous(memory_format=torch.channels_last))
            self.assertTrue(ref_out.is_contiguous())
            self.assertTrue(torch.allclose(out, ref_out))
            self.assertTrue(torch.allclose(input.grad, ref_input.grad))

        helper(4, 8, 8, 8, 7)
        helper(200, 512, 28, 28, 2)
        helper(4, 8, 7, 7, 3, stride=1)
        helper(10, 512, 31, 31, 3, stride=2)
        helper(1, 129, 8, 8, 3, stride=2)

    def _test_maxpool_indices(self, num_dim, adaptive=False, device="cpu", dtype=torch.float):
        def expected_indices(dim):
            if dim == 1:
                return torch.tensor([1, 3], dtype=torch.double).repeat(2, 2, 1)
            if dim == 2:
                return torch.tensor([[5, 7], [13, 15]], dtype=torch.double).repeat(2, 2, 1, 1)

        def expected_grad(dim):
            if dim == 1:
                return torch.tensor([0, 1, 0, 1], dtype=torch.double).repeat(2, 2, 1)
            grad = expected_grad(dim - 1)
            zero = torch.zeros(grad.size())
            return torch.stack((zero, grad, zero, grad), 2)

        def expected_output(dim):
            if dim == 1:
                return torch.arange(2, 17, 2).view(2, 2, 2)
            if dim == 2:
                col = torch.arange(6, 63, 8)
                return torch.stack([col, col + 2], 1).view(2, 2, 2, 2)

        if adaptive:
            cls_name = 'AdaptiveMaxPool{}d'.format(num_dim)
        else:
            cls_name = 'MaxPool{}d'.format(num_dim)
        module_cls = getattr(nn, cls_name)
        module = module_cls(2, return_indices=True).to(device, dtype=dtype)
        numel = 4 ** (num_dim + 1)
        input = torch.arange(1, numel + 1).view(2, 2, *repeat(4, num_dim)).to(device, dtype=dtype)
        input_var = input.clone().detach().requires_grad_()

        # Check forward
        output, indices = module(input_var)
        if num_dim != 3:
            expected_indices = expected_indices(num_dim)
            expected_output = expected_output(num_dim)
            self.assertEqual(indices.dim(), input.dim())
            self.assertEqual(indices.data.squeeze(), expected_indices)
            self.assertEqual(output.data.squeeze(), expected_output)
        self.assertTrue(output.requires_grad)
        self.assertFalse(indices.requires_grad)

        # Make sure backward works
        grad_output = torch.ones(output.size(), device=device, dtype=dtype)
        output.backward(grad_output, retain_graph=True)
        expected_grad = expected_grad(num_dim)
        self.assertEqual(input_var.grad.data, expected_grad.view_as(input))

        # Make sure backward after changing indices will result in an error
        indices.add_(1)
        self.assertRaises(RuntimeError, lambda: output.backward(grad_output))

        # Make sure -Infinity is handled correctly
        t = torch.tensor([[[float("-inf")]]])
        m = nn.MaxPool1d(kernel_size=1, return_indices=True)
        output, indices = m(t)
        self.assertEqual(output[0, 0, 0], float("-inf"), allow_inf=True)
        self.assertEqual(indices[0, 0, 0], 0)

        t = torch.tensor([[[float("-inf")]]])
        m = nn.MaxPool2d(kernel_size=1, return_indices=True)
        output, indices = m(t)
        self.assertEqual(output[0, 0, 0], float("-inf"), allow_inf=True)
        self.assertEqual(indices[0, 0, 0], 0)

        t = torch.tensor([[[[float("-inf")]]]])
        m = nn.MaxPool3d(kernel_size=1, return_indices=True)
        output, indices = m(t)
        self.assertEqual(output[0, 0, 0, 0], float("-inf"), allow_inf=True)
        self.assertEqual(indices[0, 0, 0, 0], 0)

    @dtypesIfCUDA(*ALL_TENSORTYPES2)
    @dtypes(torch.float)
    def test_MaxPool1d_indices(self, device, dtype):
        self._test_maxpool_indices(1, device=device, dtype=dtype)

    @dtypesIfCUDA(*ALL_TENSORTYPES2)
    @dtypes(torch.float)
    def test_MaxPool2d_indices(self, device, dtype):
        self._test_maxpool_indices(2, device=device, dtype=dtype)

    @dtypesIfCUDA(*ALL_TENSORTYPES2)
    @dtypes(torch.float)
    def test_MaxPool3d_indices(self, device, dtype):
        self._test_maxpool_indices(3, device=device, dtype=dtype)

    @dtypesIfCUDA(*ALL_TENSORTYPES2)
    @dtypes(torch.float)
    def _test_AdaptiveMaxPool1d_indices(self, device, dtype):
        self._test_maxpool_indices(1, adaptive=True, device=device, dtype=dtype)

    @dtypesIfCUDA(*ALL_TENSORTYPES2)
    @dtypes(torch.float)
    def _test_AdaptiveMaxPool2d_indices(self, device, dtype):
        self._test_maxpool_indices(2, adaptive=True, device=device, dtype=dtype)

    @dtypesIfCUDA(*ALL_TENSORTYPES2)
    @dtypes(torch.float)
    def test_AdaptiveMaxPool3d_indices(self, device, dtype):
        self._test_maxpool_indices(3, adaptive=True, device=device, dtype=dtype)

    @dtypesIfCUDA(*ALL_TENSORTYPES2)
    @dtypes(torch.float)
    def test_max_pool_nan(self, device, dtype):
        for adaptive in ['', 'adaptive_']:
            for num_dim in [1, 2, 3]:
                fn_name = '{}max_pool{}d'.format(adaptive, num_dim)
                fn = getattr(F, fn_name)
                x = torch.full([1, 1] + num_dim * [3], nan)
                res = fn(x, 1 if adaptive else 3)
                self.assertTrue(math.isnan(res.item()))

    @dtypesIfCUDA(*ALL_TENSORTYPES2)
    @dtypes(torch.float)
    def test_pool_large_size(self, device, dtype):
        for op in ('max',):
            for num_dim in [1, 2, 3]:
                fn_name = '{}_pool{}d'.format(op, num_dim)
                fn = getattr(F, fn_name)
                # 16777217 is the smallest integer not expressible in float32
                x = torch.ones([1, 1, 16777217] + (num_dim - 1) * [1],
                               device=device, dtype=dtype)
                res = fn(x, 1, stride=1, padding=0)
                # check if the output shape was still computed correctly
                self.assertEqual(x.shape[2], res.shape[2])

    @dtypesIfCUDA(*ALL_TENSORTYPES2)
    @dtypes(torch.float)
    def test_pool_invalid_size(self, device, dtype):
        for op in ('max',):
            for num_dim in [1, 2, 3]:
                fn_name = '{}_pool{}d'.format(op, num_dim)
                fn = getattr(F, fn_name)
                # use a configuration that gives zero outputs only
                # when doing a correct floor division by the stride
                x = torch.ones([1, 1] + num_dim * [4],
                               device=device, dtype=dtype)
                with self.assertRaisesRegex(RuntimeError, r"too small|smaller than"):
                    try:
                        res = fn(x, 3, stride=2, padding=0, dilation=2)
                    except TypeError:
                        # some implementations do not support dilation
                        res = fn(x, 6, stride=2, padding=0)

    @onlyCUDA
    def test_pool3d_size_one_feature_dim(self, device):
        # Tests crazy strides for feature dim of size 1
        x = torch.randn(7, 1, 5, 3, 2, device=device)
        strange_strides = [30, 1234, 6, 2, 1]
        y = x.as_strided(x.size(), strange_strides)
        x = x.cpu().as_strided(x.size(), strange_strides)

        to_test = {
            'max_pool3d': lambda t: F.max_pool3d(t, (5, 1, 1), stride=(5, 1, 1)),
            'avg_pool3d': lambda t: F.avg_pool3d(t, (5, 1, 1), stride=(5, 1, 1)),
        }

        for test, fn in to_test.items():
            # Should not crash
            out_y = fn(y)
            out_x = fn(x)
            self.assertEqual(out_y, out_x.to(device), test)


class TestNllLoss(TestCase):
    def test_nll_loss_mismatched_batch(self, device):
        x = torch.randn((10, 3), requires_grad=True, device=device)
        # t should have size (10,)
        t = torch.zeros((3,), dtype=torch.int64, device=device)
        with self.assertRaisesRegex(ValueError, 'Expected.*batch_size'):
            F.nll_loss(x, t)

    def test_nll_loss_out_of_bounds_ignore_index(self, device):
        x = torch.randn(6, 3, requires_grad=True, device=device)
        t = torch.tensor([0, 1, 255, 0, 1, 2], dtype=torch.int64, device=device)
        for reduction in ['mean', 'none']:
            F.nll_loss(x, t, ignore_index=255, reduction=reduction).sum().backward()

    def _nll_loss_helper(self, input_size, reduction, expected, device):
        input = torch.rand(input_size, requires_grad=True, device=device)
        num_channels = input_size[1]
        target_size = (input_size[0],) + tuple(input_size[2:])
        target = torch.randint(num_channels, target_size, device=device)

        output = F.nll_loss(input, target, reduction=reduction)
        self.assertEqual(output, expected)

        output.sum().backward()
        self.assertEqual(input.grad.size(), input.size())

    def test_nll_loss_empty_tensor_reduction_none(self, device):
        self._nll_loss_helper([0, 3], "none", torch.empty([0], device=device), device)
        self._nll_loss_helper([0, 3, 5, 7], "none", torch.empty([0, 5, 7], device=device), device)
        self._nll_loss_helper([2, 3, 0, 7], "none", torch.empty([2, 0, 7], device=device), device)
        self._nll_loss_helper([2, 3, 5, 0], "none", torch.empty([2, 5, 0], device=device), device)
        self._nll_loss_helper([2, 3, 5, 7, 0], "none", torch.empty([2, 5, 7, 0], device=device), device)

    @unittest.skipIf(TEST_WITH_UBSAN, "division-by-zero error with UBSAN")
    def test_nll_loss_empty_tensor_reduction_mean(self, device):
        nan = torch.tensor(float('nan'), device=device)
        self._nll_loss_helper([0, 3], "mean", nan, device)
        self._nll_loss_helper([0, 3, 5, 7], "mean", nan, device)
        self._nll_loss_helper([2, 3, 0, 7], "mean", nan, device)
        self._nll_loss_helper([2, 3, 5, 0], "mean", nan, device)
        self._nll_loss_helper([2, 3, 5, 7, 0], "mean", nan, device)

    def test_nll_loss_empty_tensor_reduction_sum(self, device):
        zero = torch.tensor(0, device=device)
        self._nll_loss_helper([0, 3], "sum", zero, device)
        self._nll_loss_helper([0, 3, 5, 7], "sum", zero, device)
        self._nll_loss_helper([2, 3, 0, 7], "sum", zero, device)
        self._nll_loss_helper([2, 3, 5, 0], "sum", zero, device)
        self._nll_loss_helper([2, 3, 5, 7, 0], "sum", zero, device)

    def test_nll_loss_total_weight_is_zero(self, device):
        def helper(input_size):
            input = torch.ones(input_size, requires_grad=True, device=device)
            num_channels = input_size[1]
            target_size = (input_size[0],) + tuple(input_size[2:])
            target = torch.zeros(target_size, dtype=torch.long, device=device)
            weight = torch.zeros([num_channels], device=device)
            self.assertEqual(F.nll_loss(input, target, weight).item(), 0)

        helper([2, 3])
        helper([2, 3, 5, 7])
        helper([2, 3, 5, 7, 9])


class TestRelu(TestCase):
    def test_nonlinearity_propagate_nan(self, device):
        def test(nonlinearity, *args, **kwargs):
            x = torch.tensor([nan], device=device)
            fn = getattr(F, nonlinearity)
            try:
                self.assertTrue(math.isnan(fn(x, *args, **kwargs).item()))
            except Exception as e:
                if 'not implemented' not in str(e):
                    raise

        test('relu')
        test('relu', inplace=True)


class TestLogSoftmax(TestCase):
    def test_nonlinearity_propagate_nan(self, device):
        def test(nonlinearity, *args, **kwargs):
            x = torch.tensor([nan], device=device)
            fn = getattr(F, nonlinearity)
            try:
                self.assertTrue(math.isnan(fn(x, *args, **kwargs).item()))
            except Exception as e:
                if 'not implemented' not in str(e):
                    raise

        test('log_softmax', 0)

    @dtypesIfCUDA(torch.half, torch.float)
    @dtypes(torch.float)
    def test_softmax_backward(self, device, dtype):
        sizes = [(0, 10), (32, 20), (10, 0)]
        for fn in [F.softmax, F.log_softmax]:
            for size in sizes:
                input = torch.rand(size, device=device, dtype=dtype, requires_grad=True)
                for dim in [0, 1]:
                    output = fn(input, dtype=torch.float, dim=dim).sum()
                    grad_input, = torch.autograd.grad(output, input, create_graph=True)
                    grad_input.sum().backward()


class TestPut(TestCase):
    def test_put_(self, device):
        def check(dst, idx, value):
            expected = dst.clone(memory_format=torch.contiguous_format).view(-1).index_copy_(
                0, idx.contiguous().view(-1), value.contiguous().view(-1))
            expected = expected.view_as(dst)
            dst.npu().put_(idx.npu(), value.npu())
            self.assertEqual(expected, dst.npu())

        dst = torch.randn(2, 3, 5)
        idx = torch.LongTensor([[0, 2], [3, 4]])
        values = torch.randn(2, 2)
        check(dst, idx, values)
        check(dst.transpose(1, 2), idx, values)

        values = torch.tensor([[False, False], [False, False]])
        check(dst.bool(), idx, values)

    def test_put_accumulate(self, device):
        dst = torch.ones(2, 2).npu()
        idx = torch.LongTensor([[0, 1], [0, 1]]).npu()
        src = torch.Tensor([1, 2, 3, 4]).npu()
        dst.npu().put_(idx.npu(), src.npu(), accumulate=True)
        self.assertEqual(dst.tolist(), [[5, 7], [1, 1]])

    def test_put_empty(self, device):
        for dst_shape in [(0,), (0, 1, 2, 0), (1, 2, 3)]:
            for indices_shape in [(0,), (0, 1, 2, 0)]:
                for accumulate in [False, True]:
                    dst = torch.randn(dst_shape, device=device)
                    indices = torch.empty(indices_shape, dtype=torch.int64, device=device)
                    src = torch.randn(indices_shape, device=device)
                    self.assertEqual(dst, dst.put_(indices, src, accumulate=accumulate))


class TestIndexput(TestCase):
    def test_index_put_accumulate_large_tensor(self, device):
        # This test is for tensors with number of elements >= INT_MAX (2^31 - 1).
        N = (1 << 31) + 5
        dt = torch.int8
        a = torch.ones(N, dtype=dt, device=device)
        indices = torch.LongTensor([0, 1, -2, -1])
        values = torch.tensor([10, 11, 12, 13], dtype=dt, device=device)

        a.index_put_((indices,), values, accumulate=True)

        self.assertEqual(a[0], 11)
        self.assertEqual(a[1], 12)
        self.assertEqual(a[2], 1)
        self.assertEqual(a[-100], 1)
        self.assertEqual(a[-2], 13)
        self.assertEqual(a[-1], 14)

    # @dtypes(torch.float, torch.bfloat16, torch.long, torch.bool)
    # @dtypesIfCPU(torch.float, torch.long, torch.bool, torch.bfloat16)
    # @dtypesIfCUDA(torch.half, torch.long, torch.bool)
    def test_index_put_src_datatype(self, device, dtype):
        src = torch.ones(3, 2, 4, device=device, dtype=dtype)
        vals = torch.ones(3, 2, 4, device=device, dtype=dtype)
        indices = (torch.tensor([0, 2, 1]),)
        res = src.index_put_(indices, vals, accumulate=True)
        self.assertEqual(res.shape, src.shape)


class TestRepeat(TestCase):
    def test_repeat(self):
        initial_shape = (8, 4)
        tensor = torch.rand(*initial_shape)

        size = (3, 1, 1)
        torchSize = torch.Size(size)
        target = [3, 8, 4]
        self.assertEqual(tensor.repeat(*size).size(), target, 'Error in repeat')
        self.assertEqual(tensor.repeat(torchSize).size(), target,
                         'Error in repeat using LongStorage')
        result = tensor.repeat(*size)
        self.assertEqual(result.size(), target, 'Error in repeat using result')
        result = tensor.repeat(torchSize)
        self.assertEqual(result.size(), target, 'Error in repeat using result and LongStorage')
        self.assertEqual(result.mean(0).view(8, 4), tensor, 'Error in repeat (not equal)')

        zeroDimTarget = torch.Size([24, 0])
        self.assertEqual(tensor.repeat((3, 0)).size(), zeroDimTarget, "Error when calling with 0 repeats")

    def test_repeat_interleave(self):
        x = torch.tensor([0, 1, 2, 3])
        expected = torch.tensor([1, 2, 2, 3, 3, 3])
        self.assertEqual(torch.repeat_interleave(x), expected)

        with self.assertRaises(RuntimeError):
            torch.repeat_interleave(torch.arange(4).reshape(2, 2))

        with self.assertRaises(RuntimeError):
            torch.repeat_interleave(torch.arange(4.0))

        with self.assertRaises(RuntimeError):
            torch.repeat_interleave(torch.tensor([1, 2, -1, 3, 4]))

        y = torch.tensor([[1, 2], [3, 4]])

        y1_v1 = torch.repeat_interleave(y, 2)
        y1_v2 = torch.repeat_interleave(y, torch.tensor(2))
        y1_v3 = torch.repeat_interleave(y, torch.tensor([2]))
        y1_expect = torch.tensor([1, 1, 2, 2, 3, 3, 4, 4])
        self.assertEqual(y1_v1, y1_expect)
        self.assertEqual(y1_v2, y1_expect)
        self.assertEqual(y1_v3, y1_expect)

        y2 = torch.repeat_interleave(y, 3, dim=1)
        y2_expect = torch.tensor([[1, 1, 1, 2, 2, 2],
                                  [3, 3, 3, 4, 4, 4]])
        self.assertEqual(y2, y2_expect)

        y3 = torch.repeat_interleave(y, torch.tensor([1, 2]), dim=0)
        y3_expect = torch.tensor([[1, 2],
                                  [3, 4],
                                  [3, 4]])
        self.assertEqual(y3, y3_expect)

        with self.assertRaises(RuntimeError):
            torch.repeat_interleave(y, torch.tensor([1, 2, 3]), dim=0)

        with self.assertRaises(RuntimeError):
            torch.repeat_interleave(y, torch.arange(9).reshape(3, 3), dim=0)

        # test zero sized dimension
        x = torch.zeros((5, 0))
        y = torch.repeat_interleave(x, repeats=3, dim=1)
        self.assertEqual(y, x.new_zeros(5, 0))

        x = torch.tensor([], dtype=torch.int64)
        y = torch.repeat_interleave(x, x)
        self.assertEqual(y, x)

    @unittest.skipIf(not TEST_NUMPY, "Numpy not found")
    def test_repeat_tile(self):

        initial_shape = (8, 4)

        repeats = ((3, 1, 1),
                   (3, 3, 3),
                   (1, 2, 1),
                   (2, 2, 2, 2))

        def _generate_noncontiguous_input():

            out = np.broadcast_to(np.random.random((1, 4)),
                                  initial_shape)
            # Note: non-writeable NumPy arrays will warn if converted to tensors
            out.setflags(write=True)

            assert not (out.flags.c_contiguous or out.flags.f_contiguous)

            return out

        for repeat in repeats:
            for tensor in (torch.from_numpy(np.random.random(initial_shape)),
                           torch.from_numpy(_generate_noncontiguous_input()),):
                self.assertEqual(tensor.repeat(*repeat).numpy(),
                                 np.tile(tensor.numpy(), repeat))


class TestRsqrt(TestCase):
    def test_rsqrt_1(self, device):
        def rsqrt(x):
            if x == 0:
                return inf
            elif x < 0:
                return nan
            return 1.0 / math.sqrt(x)

        _test_math(torch.rsqrt, rsqrt)


# class TestSigmoid(TestCase):
#     #@onlyCPU
#     @dtypes(torch.float, torch.float16)
#     def test_sigmoid_1(self, device, dtype):
#         # TODO: why not simulate math.sigmoid like with rsqrt?
#         inputValues = [-1000, -1, 0, 0.5, 1, 2, 1000]
#         expectedOutput = [0.0000, 0.2689, 0.5, 0.6225, 0.7311, 0.8808, 1.000]
#         precision_4dps = 0.0002
#
#         self.assertEqual(torch.tensor(inputValues, dtype=dtype, device=device).sigmoid(),
#                          torch.tensor(expectedOutput, dtype=dtype, device=device), precision_4dps)


class TestSign(TestCase):
    def test_sign(self, device):
        for dtype in torch.testing.get_all_math_dtypes(device):

            # Include NaN for floating point numbers
            if dtype.is_floating_point:
                dt_info = torch.finfo(dtype)

                # Create tensor (with NaN checking)
                a = torch.tensor([float('nan'), -12, 0, 71, dt_info.min, dt_info.max], device=device, dtype=dtype)
                a_target = torch.tensor([0, -1, 0, 1, -1, 1], device=device, dtype=dtype)

            else:
                dt_info = torch.iinfo(dtype)

                # If unsigned type, everything should be >= 0
                if dt_info.min == 0:
                    a = torch.tensor([12, 0, 71, dt_info.min, dt_info.max], device=device, dtype=dtype)
                    a_target = torch.tensor([1, 0, 1, 0, 1], device=device, dtype=dtype)
                else:
                    a = torch.tensor([-12, 0, 71, dt_info.min, dt_info.max], device=device, dtype=dtype)
                    a_target = torch.tensor([-1, 0, 1, -1, 1], device=device, dtype=dtype)

            self.assertEqual(a.sign().cpu(), a_target, 'sign device={} dtype={}'.format(device, dtype))
            self.assertEqual(torch.sign(a).cpu(), a_target, 'sign device={} dtype={}'.format(device, dtype))

            out = torch.empty_like(a)
            torch.sign(a, out=out)
            self.assertEqual(out.cpu(), a_target, 'sign_out device={} dtype={}'.format(device, dtype))

            a.sign_()
            self.assertEqual(a.cpu(), a_target, 'sign_ device={} dtype={}'.format(device, dtype))

        # Include test for bool dtype
        a_bool = torch.tensor([True, True, False, float('nan')], device=device).bool()
        a_bool_target = torch.tensor([True, True, False, True], device=device).bool()
        self.assertEqual(a_bool.sign().cpu(), a_bool_target, 'sign device={} dtype=bool'.format(device))
        self.assertEqual(torch.sign(a_bool).cpu(), a_bool_target, 'sign device={} dtype=bool'.format(device))

        a_out = torch.empty_like(a_bool)
        torch.sign(a_bool, out=a_out)
        self.assertEqual(a_out.cpu(), a_bool_target, 'sign_out device={} dtype=bool'.format(device))

        a_bool.sign_()
        self.assertEqual(a_bool.cpu(), a_bool_target, 'sign_ device={} dtype=bool'.format(device))


class TestSort(TestCase):
    def test_sort_1(self, device):
        SIZE = 4
        x = torch.rand(SIZE, SIZE)
        res1val, res1ind = torch.sort(x)

        # Test use of result tensor
        res2val = torch.Tensor()
        res2ind = torch.LongTensor()
        torch.sort(x, out=(res2val, res2ind))
        self.assertEqual(res1val, res2val, 0)
        self.assertEqual(res1ind, res2ind, 0)
        self.assertEqual(torch.argsort(x), res1ind)
        self.assertEqual(x.argsort(), res1ind)

        # Test sorting of random numbers
        self.assertIsOrdered('ascending', x, res2val, res2ind, 'random')

        # Test simple sort
        self.assertEqual(
            torch.sort(torch.Tensor((50, 40, 30, 20, 10)))[0],
            torch.Tensor((10, 20, 30, 40, 50)),
            0
        )

        # Test that we still have proper sorting with duplicate keys
        x = torch.floor(torch.rand(SIZE, SIZE) * 10)
        torch.sort(x, out=(res2val, res2ind))
        self.assertIsOrdered('ascending', x, res2val, res2ind, 'random with duplicate keys')

        # DESCENDING SORT
        x = torch.rand(SIZE, SIZE)
        res1val, res1ind = torch.sort(x, x.dim() - 1, True)

        # Test use of result tensor
        res2val = torch.Tensor()
        res2ind = torch.LongTensor()
        torch.sort(x, x.dim() - 1, True, out=(res2val, res2ind))
        self.assertEqual(res1val, res2val, 0)
        self.assertEqual(res1ind, res2ind, 0)
        self.assertEqual(torch.argsort(x, x.dim() - 1, True), res1ind)
        self.assertEqual(x.argsort(x.dim() - 1, True), res1ind)

        # Test sorting of random numbers
        self.assertIsOrdered('descending', x, res2val, res2ind, 'random')

        # Test simple sort task
        self.assertEqual(
            torch.sort(torch.Tensor((10, 20, 30, 40, 50)), 0, True)[0],
            torch.Tensor((50, 40, 30, 20, 10)),
            0
        )

        # Test that we still have proper sorting with duplicate keys
        self.assertIsOrdered('descending', x, res2val, res2ind, 'random with duplicate keys')

        # Test sorting with NaNs
        x = torch.rand(SIZE, SIZE)
        x[1][2] = float('NaN')
        x[3][0] = float('NaN')
        torch.sort(x, out=(res2val, res2ind))
        self.assertIsOrdered('ascending', x, res2val, res2ind,
                             'random with NaNs')
        torch.sort(x, out=(res2val, res2ind), descending=True)
        self.assertIsOrdered('descending', x, res2val, res2ind,
                             'random with NaNs')


class TestSplit(TestCase):
    def test_split_1(self, device):
        tensor = torch.rand(7, 4)
        split_size = 3
        dim = 0
        target_sizes = ([3, 4], [3, 4], [1, 4])
        splits = tensor.split(split_size, dim)
        start = 0
        for target_size, split in zip(target_sizes, splits):
            self.assertEqual(split.size(), target_size)
            self.assertEqual(tensor.narrow(dim, start, target_size[dim]), split, 0)
            start = start + target_size[dim]

        # Variable sections split
        tensor = torch.randn(20, 10)
        dim = 0
        split_sizes = [5, 5, 10]
        target_sizes = ([[5, 10], [5, 10], [10, 10]])
        splits = tensor.split(split_sizes, dim)
        start = 0
        for target_size, split in zip(target_sizes, splits):
            self.assertEqual(split.size(), target_size)
            self.assertEqual(tensor.narrow(dim, start, target_size[dim]), split, 0)
            start = start + target_size[dim]

        split_sizes = [2, 2, 6]
        target_sizes = ([20, 2], [20, 2], [20, 6])
        dim = 1
        splits = tensor.split(split_sizes, dim)
        start = 0
        for target_size, split in zip(target_sizes, splits):
            self.assertEqual(split.size(), target_size)
            self.assertEqual(tensor.narrow(dim, start, target_size[dim]), split, 0)
            start = start + target_size[dim]

    @unittest.skip("See https://github.com/pytorch/pytorch/pull/32720")
    def test_split_view(self, device):
        t = torch.zeros(3, 3, device=device)
        l = torch.split(t, [1, 1, 1])

        for idx, v in enumerate(l):
            self.assertTrue(self.is_view_of(t, v))

            v[0, 0] = idx + 1
            self.assertEqual(t[idx, 0], v[0, 0])


class TestSqrt(TestCase):
    def test_sqrt_1(self, device):
        _test_math(torch.sqrt, lambda x: math.sqrt(x) if x >= 0 else nan)


class TestStack(TestCase):
    def test_stack(self, device):
        for dtype in (torch.half, torch.double, torch.int):
            x = torch.randint(low=-100, high=100, size=(2, 3, 4)).to(dtype)
            y = torch.randint(low=-100, high=100, size=(2, 3, 4)).to(dtype)
            z = torch.randint(low=-100, high=100, size=(2, 3, 4)).to(dtype)
            for dim in range(4):
                res = torch.stack((x, y, z), dim)
                res_neg = torch.stack((x, y, z), dim - 4)
                expected_size = x.size()[:dim] + (3,) + x.size()[dim:]
                self.assertEqual(res, res_neg)
                self.assertEqual(res.size(), expected_size)
                self.assertEqual(res.select(dim, 0), x, 0)
                self.assertEqual(res.select(dim, 1), y, 0)
                self.assertEqual(res.select(dim, 2), z, 0)

    def test_stack_out(self, device):
        for dtype in (torch.half, torch.double, torch.int):
            x = torch.randint(low=-100, high=100, size=(2, 3, 4)).to(dtype)
            y = torch.randint(low=-100, high=100, size=(2, 3, 4)).to(dtype)
            z = torch.randint(low=-100, high=100, size=(2, 3, 4)).to(dtype)
            for dim in range(4):
                expected_size = x.size()[:dim] + (3,) + x.size()[dim:]
                res_out = x.new(expected_size)
                res_neg_out = x.new(expected_size)
                res_out_dp = res_out.data_ptr()
                res_out_neg_dp = res_neg_out.data_ptr()
                torch.stack((x, y, z), dim, out=res_out)
                torch.stack((x, y, z), dim - 4, out=res_neg_out)
                self.assertEqual(res_out, res_neg_out)
                self.assertEqual(res_out.size(), expected_size)
                self.assertEqual(res_out_dp, res_out.data_ptr())
                self.assertEqual(res_out_neg_dp, res_neg_out.data_ptr())
                self.assertEqual(res_out.select(dim, 0), x, 0)
                self.assertEqual(res_out.select(dim, 1), y, 0)
                self.assertEqual(res_out.select(dim, 2), z, 0)


class TestSub(TestCase):
    def test_sub_typing(self, device):
        m1 = torch.tensor([True, False, False, True, False, False], dtype=torch.bool, device=device)
        m2 = torch.tensor([True, True, False, False, False, True], dtype=torch.bool, device=device)
        self.assertRaisesRegex(RuntimeError,
                               r"Subtraction, the `\-` operator, with two bool tensors is not supported. "
                               r"Use the `\^` or `logical_xor\(\)` operator instead.",
                               lambda: m1 - m2)
        self.assertRaisesRegex(RuntimeError,
                               r"Subtraction, the `\-` operator, with a bool tensor is not supported. "
                               r"If you are trying to invert a mask, use the `\~` or `logical_not\(\)` operator instead.",
                               lambda: 1 - m1)
        self.assertRaisesRegex(RuntimeError,
                               r"Subtraction, the `\-` operator, with a bool tensor is not supported. "
                               r"If you are trying to invert a mask, use the `\~` or `logical_not\(\)` operator instead.",
                               lambda: m2 - 1)

        # mismatched alpha
        m1 = torch.tensor([1], dtype=torch.int8, device=device)
        m2 = torch.tensor([2], dtype=torch.int8, device=device)
        self.assertRaisesRegex(RuntimeError,
                               r"Boolean alpha only supported for Boolean results\.",
                               lambda: torch.sub(m1, m2, alpha=True))
        self.assertRaisesRegex(RuntimeError,
                               r"For integral input tensors, argument alpha must not be a floating point number\.",
                               lambda: torch.sub(m1, m2, alpha=1.0))

    @onlyCPU
    @dtypes(*torch.testing.get_all_dtypes())
    def test_sub_1(self, device, dtype):
        m1 = torch.tensor([2.34, 4.44], dtype=dtype, device=device)
        m2 = torch.tensor([1.23, 2.33], dtype=dtype, device=device)

        if (dtype == torch.half or dtype == torch.bool):
            self.assertRaises(RuntimeError, lambda: m1 - m2)
        elif (dtype == torch.bfloat16):
            # bfloat16 has a lower precision so we have to have a separate check for it
            self.assertEqual(m1 - m2, torch.tensor([1.11, 2.11], dtype=dtype), 0.01)
        else:
            self.assertEqual(m1 - m2, torch.tensor([1.11, 2.11], dtype=dtype))


class TestSum(TestCase):
    @slowTest
    @unittest.skipIf(not TEST_NUMPY, 'Numpy not found')
    def test_sum_dim_1(self, device):
        self._test_dim_ops(
            lambda t, d: t.sum(d),
            lambda n, d: n.sum(d))

    def test_sum_integer_upcast(self, device):
        self._test_reduce_integer_upcast(lambda x, **kwargs: torch.sum(x, **kwargs), False)
        self._test_reduce_integer_upcast(lambda x, **kwargs: torch.sum(x, 0, **kwargs))

    @onlyCPU
    @dtypes(torch.bool, torch.double)
    def test_sum_all(self, device, dtype):
        def check_sum_all(tensor):
            pylist = tensor.reshape(-1).tolist()
            self.assertEqual(tensor.sum(), sum(pylist))

        if dtype != torch.bool:
            check_sum_all(torch.tensor([1, 2, 3, 4, 5], dtype=dtype, device=device))
            check_sum_all(torch.randn(200000, dtype=dtype, device=device))
            check_sum_all(torch.randn(2000, 2, dtype=dtype, device=device)[:, 0])
        else:
            check_sum_all(torch.tensor([True, False, True], dtype=torch.bool, device=device))

    @onlyCPU
    @dtypes(torch.double)
    def test_sum_out(self, device, dtype):
        x = torch.rand(100, 100, dtype=dtype, device=device)
        res1 = torch.sum(x, 1)
        res2 = torch.tensor((), dtype=dtype, device=device)
        torch.sum(x, 1, out=res2)
        self.assertEqual(res1, res2)
        x = torch.rand(100, 100, 100, dtype=dtype, device=device)
        res1 = x.sum(2).sum(1)
        res2 = torch.tensor((), dtype=dtype, device=device)
        torch.sum(x, (2, 1), out=res2)
        self.assertEqual(res1, res2)

    @skipCUDAIfRocm
    @dtypes(torch.double)
    def test_sum_noncontig(self, device, dtype):
        x = torch.randn(1, 75, 57, 20, dtype=dtype, device=device).permute(0, 3, 1, 2)
        y = x.cpu()
        self.assertEqual(x.sum().cpu(), y.sum())
        self.assertEqual(x.sum(dim=(-1, -2)).cpu(), y.sum(dim=(-1, -2)))
        self.assertEqual(x.sum(dim=(1, 3)).cpu(), y.sum(dim=(1, 3)))

    def test_sum_cpu_device_mismatch(self, device):
        x = torch.randn(20, dtype=torch.float32, device=device)
        y = torch.randn(1, dtype=torch.float32)

        err_string = "output with device cpu doesn't match the desired device {0}".format(device)

        with self.assertRaisesRegex(RuntimeError, err_string):
            torch.sum(x, dim=[0], dtype=torch.float32, out=y)

        # tests half to float promotion
        if self.device_type == 'cuda':
            x = x.half()
            with self.assertRaisesRegex(RuntimeError, err_string):
                torch.sum(x, dim=[0], dtype=torch.float32, out=y)


class TestFillDiagonal(TestCase):
    def test_fill_diagonal(self, device):
        a1 = torch.randn(7, 3)
        a2 = a1.clone()
        v = 1
        for i in range(3):
            a2[i][i] = v
        a1.fill_diagonal_(v)
        self.assertEqual(a1, a2)

        b1 = torch.randn(7, 3)
        b2 = b1.clone()
        for i in range(3):
            b2[i][i] = v
            b2[i + 4][i] = v
        b1.fill_diagonal_(v, wrap=True)
        self.assertEqual(b1, b2)

        c1 = torch.rand(3, 3, 3)
        c2 = c1.clone()
        for i in range(3):
            c2[i][i][i] = v
        c1.fill_diagonal_(v)
        self.assertEqual(c1, c2)

        # non-contiguous tensor
        d1 = torch.rand(3, 3, 3)[:, 1, ...]
        d2 = d1.clone()
        for i in range(3):
            d2[i][i] = v
        d1.fill_diagonal_(v)
        self.assertEqual(d1, d2)

        e1 = torch.rand(7, 3, 3)[:, 1, ...]
        e2 = e1.clone()
        for i in range(3):
            e2[i][i] = v
            e2[i + 4][i] = v
        e1.fill_diagonal_(v, wrap=True)
        self.assertEqual(e1, e2)


class TestTopk(TestCase):
    exact_dtype = True

    def test_topk(self, device):
        SIZE = 4

        def topKViaSort(t, k, dim, dir):
            sorted, indices = t.sort(dim, dir)
            return sorted.narrow(dim, 0, k), indices.narrow(dim, 0, k)

        def compareTensors(t, res1, ind1, res2, ind2, dim):
            # Values should be exactly equivalent
            self.assertEqual(res1.to('cpu'), res2, 0)

            # Indices might differ based on the implementation, since there is
            # no guarantee of the relative order of selection
            if not ind1.eq(ind2).all():
                # To verify that the indices represent equivalent elements,
                # gather from the input using the topk indices and compare against
                # the sort indices
                vals = t.gather(dim, ind2)
                self.assertEqual(res1.to('cpu'), vals, 0)

        def compare(t, k, dim, dir):
            topKVal, topKInd = t.topk(k, dim, dir, True)
            sortKVal, sortKInd = topKViaSort(t, k, dim, dir)
            compareTensors(t, sortKVal, sortKInd, topKVal, topKInd, dim)

        t = torch.rand(random.randint(1, SIZE),
                       random.randint(1, SIZE),
                       random.randint(1, SIZE))

        for _kTries in range(3):
            for _dimTries in range(3):
                for transpose in (True, False):
                    for dir in (True, False):
                        testTensor = t
                        if transpose:
                            dim1 = random.randrange(t.ndimension())
                            dim2 = dim1
                            while dim1 == dim2:
                                dim2 = random.randrange(t.ndimension())

                            testTensor = t.transpose(dim1, dim2)

                        dim = random.randrange(testTensor.ndimension())
                        k = random.randint(1, testTensor.size(dim))
                        compare(testTensor, k, dim, dir)

    def test_topk_arguments(self, device):
        q = torch.randn(10, 2, 10).to(device)
        # Make sure True isn't mistakenly taken as the 2nd dimension (interpreted as 1)
        self.assertRaises(TypeError, lambda: q.topk(4, True))

    # @onlyCUDA
    def test_topk_noncontiguous(self, device):
        t = torch.randn(20, device=device)[::2]
        top1, idx1 = t.topk(5)
        top2, idx2 = t.contiguous().topk(5)
        self.assertEqual(top1.to('cpu'), top2)
        self.assertEqual(idx1.to('cpu'), idx2)

    @dtypes(torch.int8, torch.uint8, torch.int16, torch.int32, torch.int64)
    def test_topk_integral(self, device, dtype):
        a = torch.randint(torch.iinfo(dtype).min, torch.iinfo(dtype).max, size=(10,),
                          dtype=dtype, device=device)
        # sort_topk = a.sort()[0][-5:].flip(0)
        sort_topk = a.to('cpu').sort()[0][-5:].flip(0)
        topk = a.topk(5)
        self.assertEqual(sort_topk.to('cpu'), topk[0])      # check values
        self.assertEqual(sort_topk.to('cpu'), a[topk[1]])   # check indices

    # @dtypesIfCUDA(torch.half, torch.float, torch.double)
    @dtypes(torch.half, torch.float, torch.double)
    def test_topk_nonfinite(self, device, dtype):
        x = torch.tensor([float('nan'), float('inf'), 1e4, 0, -1e4, -float('inf')], device=device, dtype=dtype)
        val, idx = x.topk(4)
        expect = torch.tensor([float('nan'), float('inf'), 1e4, 0], device=device, dtype=dtype)
        self.assertEqual(val.to('cpu'), expect, allow_inf=True)
        self.assertEqual(idx.to('cpu'), [0, 1, 2, 3])

        val, idx = x.topk(4, largest=False)
        expect = torch.tensor([-float('inf'), -1e4, 0, 1e4], device=device, dtype=dtype)
        self.assertEqual(val.to('cpu'), expect, allow_inf=True)
        self.assertEqual(idx.to('cpu'), [5, 4, 3, 2])


class TestWhere(TestCase):
    def test_where_invalid_device(self):
        if torch.npu.is_available():
            for devices in [('cpu', 'npu', 'npu'), ('npu', 'cpu', 'cpu'),
                            ('npu', 'cpu', 'npu'), ('cpu', 'npu', 'cpu')]:
                condition = torch.rand(16, device=devices[0])
                x = torch.rand(16, device=devices[1])
                y = torch.rand(16, device=devices[2])
                with self.assertRaisesRegex(RuntimeError,
                                            "expected condition, x and y to be on the same device"):
                    torch.where(condition, x, y)

    def test_where_bool_tensor(self):
        for d in torch.testing.get_all_device_types():
            a = torch.tensor([True, False], device=d)
            res = torch.where(a > 0)
            self.assertEqual(1, len(res))

    def test_where_tensor(self):
        def rand_tensor(size, dtype, device):
            if dtype.is_floating_point:
                return torch.rand(size=size, dtype=dtype, device=device)
            elif dtype == torch.uint8:
                return torch.randint(1, 5, size=size, dtype=dtype, device=device)
            elif dtype == torch.bool:
                return torch.randint(0, 1, size=size, dtype=dtype, device=device).bool()
            else:
                return torch.randint(-5, 5, size=size, dtype=dtype, device=device)

        def get_tensor(size, dtype, device, contiguous):
            if not contiguous and len(size) < 2:
                raise RuntimeError("Unable to generate non contiguous tensor with size < 2")
            t = rand_tensor(size, dtype, device)
            if contiguous:
                return t
            else:
                return t.transpose(0, 1)

        height = 5
        width = 5
        for device in torch.testing.get_all_device_types():
            for dt1 in torch.testing.get_all_math_dtypes(device):
                for dt2 in torch.testing.get_all_math_dtypes(device):
                    for contiguous in [True, False]:
                        x1 = get_tensor((height, width), dt1, device, contiguous)
                        x2 = get_tensor((height, width), dt2, device, contiguous)
                        if dt1 != dt2:
                            self.assertRaisesRegex(RuntimeError, "expected scalar type", lambda: torch.where(x1 == 1, x1, x2))
                        else:
                            if x1.is_floating_point():
                                condition = (x1 < 0.5)
                            else:
                                condition = (x1 == 1)
                            expected = condition.to(x1.dtype) * x1 + (~condition).to(x2.dtype) * x2
                            result = torch.where(condition, x1, x2)
                            self.assertEqual(expected, result)


class TestZero_(TestCase):
    exact_dtype = True


class TestZeros(TestCase):

    def test_zeros(self, device):

        res1 = torch.zeros(100, 100, device = device)
        res2 = torch.Tensor().to(device = device)
        torch.zeros(100, 100, device = device, out=res2)
        self.assertEqual(res1.to('cpu'), res2)

        boolTensor = torch.zeros(2, 2, dtype=torch.bool)
        expected = torch.tensor([[False, False], [False, False]], dtype=torch.bool)
        self.assertEqual(boolTensor.to('cpu'), expected)

        halfTensor = torch.zeros(1, 1, dtype=torch.half)
        expected = torch.tensor([[0.]], dtype=torch.float16)
        self.assertEqual(halfTensor.to('cpu'), expected)

        bfloat16Tensor = torch.zeros(1, 1, dtype=torch.bfloat16)
        expected = torch.tensor([[0.]], dtype=torch.bfloat16)
        self.assertEqual(bfloat16Tensor.to('cpu'), expected)

    def test_zeros_out(self, device):
        shape = (3, 4)
        out = torch.zeros(shape).to(device)
        torch.zeros(shape, device=device, out=out)

        # change the dtype, layout, device
        self.assertRaises(RuntimeError, lambda: torch.zeros(shape, dtype=torch.int64, device=device, out=out))
        self.assertRaises(RuntimeError, lambda: torch.zeros(shape, layout=torch.sparse_coo, device=device, out=out))
        if torch.npu.is_available():
            self.assertRaises(RuntimeError, lambda: torch.zeros(shape, device=device, out=out))

        # leave them the same
        self.assertEqual(torch.zeros(shape).to('cpu'), torch.zeros(shape, dtype=out.dtype, device=device, out=out))
        self.assertEqual(torch.zeros(shape).to('cpu'), torch.zeros(shape, layout=torch.strided, device=device, out=out))
        self.assertEqual(torch.zeros(shape).to('cpu'), torch.zeros(shape, device='cpu', out=out))

    def test_zeros_like(self, device):
        expected = torch.zeros((100, 100,), device=device)

        res1 = torch.zeros_like(expected)
        self.assertEqual(res1, expected)

    @deviceCountAtLeast(2)
    def test_zeros_like_multiple_device(self, devices):
        expected = torch.zeros(100, 100, device=devices[0])
        x = torch.randn(100, 100, device=devices[1], dtype=torch.float32)
        output = torch.zeros_like(x)
        self.assertEqual(output, expected)


class TestNNNPU(NNTestCase):
    _do_cuda_memory_leak_check = True
    _do_cuda_non_default_stream = True

    def _forward(self, module, input):
        with freeze_rng_state():
            return module(input)

    def _backward(self, module, input, output, grad_output, create_graph=False):
        output.backward(grad_output, retain_graph=True, create_graph=create_graph)
        if input.grad is None:
            return None
        return input.grad.data

    def _forward_criterion(self, criterion, input, target, extra_args=None):
        if extra_args is None:
            extra_args = tuple()
        if isinstance(input, tuple):
            args = input + (target,) + extra_args
            output = criterion(*args)
        else:
            output = criterion(input, target, *extra_args)
        return output

    def _backward_criterion(self, criterion, input, target, gradOutput=None, extra_args=None):
        if extra_args is None:
            extra_args = tuple()
        input_tuple = input if isinstance(input, tuple) else (input,)
        for i in input_tuple:
            if i.grad is not None:
                i.grad.data.zero_()
        args = input_tuple + (target,) + extra_args
        if gradOutput is None:
            gradOutput = torch.ones(())
        criterion(*args).backward(gradOutput.type_as(input_tuple[0]))
        if isinstance(input, tuple):
            return tuple(map(lambda i: i.grad.data, input))
        else:
            return input.grad.data

    def _zero_grad_parameters(self, module):
        for p in module.parameters():
            if p.grad is not None:
                with torch.no_grad():
                    p.grad.zero_()
                p.grad.detach_()

    def _get_parameters(self, module):
        params = []
        d_params = []
        for p in module.parameters():
            params.append(p)
            d_params.append(p.grad)
        return params, d_params

    def _create_basic_net(self):
        class Layer(nn.Module):
            def __init__(self):
                super(Layer, self).__init__()
                self.layer_dummy_param = Parameter(torch.Tensor(3, 5))
                self.register_buffer('layer_dummy_buf', torch.zeros(1, 3, 3, 7))

        class Net(nn.Module):
            def __init__(self):
                super(Net, self).__init__()
                self.l1 = Layer()
                self.dummy_param = Parameter(torch.Tensor(3, 5))
                self.register_buffer('dummy_buf', torch.zeros(7, 3, 3, 1))

        l = Layer()
        n = Net()
        s = nn.Sequential(n, n)

        return l, n, s

    @contextlib.contextmanager
    def _compatible_subtest(self, **kwargs):
        # Added for subtest compatibility with Python 2
        if PY3:
            with self.subTest(**kwargs):
                yield
        else:
            yield


class TestSigmoid(TestNNNPU):
    pass


class TestSoftmax(TestNNNPU):
    pass


class TestAvg_pool2d(TestNNNPU):
    pass


class TestBce_with_logistic(TestNNNPU):
    pass


class TestUnfold(TestNNNPU):
    pass


class TestLeakyrelu(TestNNNPU):
    pass


class TestEmbedding(NNTestCase):
    def test_embedding_dense_grad(self, device):
        embd = nn.Embedding(20, 20).to(device)
        weight = embd.weight

        def fn_wrapper(device):
            def fn(weight):
                inp = torch.tensor([[0, 1, 1, 2], [3, 5, 7, 11]], dtype=torch.long).to(device)
                return torch.nn.functional.embedding(inp, weight)
            return fn

        fn = fn_wrapper(device)
        _assertGradAndGradgradChecks(self, fn, (weight, ))
    # @dtypesIfCUDA(torch.float16, torch.float64)
    # @dtypes(torch.float64)
    @dtypes(torch.float32)
    def test_embedding_backward(self, device, dtype):
        pdb.set_trace()
        embedding = nn.Embedding(10, 3, sparse=True)
        tensor = torch.tensor([[7, 1, 3]])
        ones = torch.tensor(1.).expand(3, 3)
        tensorTwice = tensor.repeat(1, 2)
        onesTwice = torch.cat((ones, ones))

        embedding = embedding.to(dtype=dtype).to(device)
        tensor = tensor.to(device)
        ones = ones.to(device)
        tensorTwice = tensorTwice.to(device)
        onesTwice = onesTwice.to(device)

        embedding.zero_grad()
        embedding(tensor[0]).sum().backward()
        self.assertEqual(embedding.weight.grad._indices(), tensor)
        self.assertEqual(embedding.weight.grad._values(), ones)

        embedding.zero_grad()
        embedding(tensor[0]).sum().backward()
        embedding(tensor[0]).sum().backward()
        self.assertEqual(embedding.weight.grad._indices(), tensorTwice)
        self.assertEqual(embedding.weight.grad._values(), onesTwice)

        embedding.zero_grad()
        embedding(tensor[0]).sum().backward()
        tensor[0, 0] = 8
        embedding(tensor[0]).sum().backward()
        tensorTwice[0, 3] = 8
        self.assertEqual(embedding.weight.grad._indices(), tensorTwice)
        self.assertEqual(embedding.weight.grad._values(), onesTwice)

    def test_embedding_padding_idx(self, device):
        embedding = nn.Embedding(10, 20, padding_idx=0).to(device)
        input = torch.tensor([[0, 2, 4, 5], [4, 3, 0, 9]], dtype=torch.long).to(device)
        output = embedding(input)
        self.assertEqual(output[0][0].sum(), 0)
        self.assertEqual(output[1][2].sum(), 0)

        embedding = nn.Embedding(10, 20, padding_idx=0, sparse=True).to(device)
        input = torch.tensor([[0, 2, 4, 5], [4, 3, 0, 9]], dtype=torch.long).to(device)
        output = embedding(input)
        self.assertEqual(output[0][0].sum(), 0)
        self.assertEqual(output[1][2].sum(), 0)

        # negative indexing check for padding_idx
        # padding_idx=-2, num_embeddings=10 ==> index 8 padded
        embedding = nn.Embedding(10, 20, padding_idx=-2).to(device)
        input = torch.tensor([[0, 2, 8, 5], [4, 8, 0, 9]], dtype=torch.long).to(device)
        output = embedding(input)
        self.assertEqual(output[0][2].sum(), 0)
        self.assertEqual(output[1][1].sum(), 0)

        embedding = nn.Embedding(10, 20, padding_idx=-2, sparse=True).to(device)
        input = torch.tensor([[0, 2, 8, 5], [4, 8, 0, 9]], dtype=torch.long).to(device)
        output = embedding(input)
        self.assertEqual(output[0][2].sum(), 0)
        self.assertEqual(output[1][1].sum(), 0)

        # out of bounds check for padding_idx
        self.assertRaises(AssertionError, nn.Embedding, num_embeddings=10, embedding_dim=20, padding_idx=25)
        self.assertRaises(AssertionError, nn.Embedding, num_embeddings=10, embedding_dim=20, padding_idx=-25)

        # test backward when input contains padding_idx
        padding_idx = 0
        embedding = nn.Embedding(5, 2, padding_idx=padding_idx).to(device)
        for n in (1, 2, 1000):  # Need large N to trigger all the methods we have implemented
            for other_indices in ([], [1, 3], [2]):
                indices = torch.tensor(other_indices + [padding_idx] * n, dtype=torch.long).to(device)
                pre = embedding.weight[padding_idx].clone()
                embedding(indices).sum().backward()
                after = (embedding.weight + embedding.weight.grad)[padding_idx]
                embedding.zero_grad()
                self.assertEqual(after, pre)

                # test double backward
                emb_sum = embedding(indices).sum()
                emb_grad = torch.autograd.grad(outputs=emb_sum, inputs=list(embedding.parameters()), retain_graph=True)
                scalar = emb_grad[0].sum() + emb_sum
                scalar.backward()
                after = (embedding.weight + embedding.weight.grad)[padding_idx]
                embedding.zero_grad()
                self.assertEqual(after, pre)

    # @onlyCUDA
    # @dtypes(*ALL_TENSORTYPES2)
    @dtypes(torch.float)
    def test_embedding_max_norm_device(self, device, dtype):
        embedding = nn.Embedding(22, 5, max_norm=1.0).to(device, dtype=dtype)
        # nn.Embedding only takes LongTensor as input
        input = torch.tensor([2, 8, 8, 6], device=device, dtype=torch.long)
        output = embedding(input)
        self.assertEqual(output[1], output[2])
        self.assertTrue(output.data.norm(p=2, dim=1).le(1).all())


class TestUpsamplingNearest2d(NNTestCase):

    def test_upsamplingNearest2d_launch_config(self, device):
        m = nn.Upsample(scale_factor=2)
        inp = torch.rand(2**25, 1, 1, 1, device=device)
        out = m(inp)
        inp_ref = inp.cpu()
        out_ref = m(inp_ref)
        self.assertEqual(out_ref, out)

    @unittest.expectedFailure
    @skipIfRocm
    # @onlyCUDA
    def test_upsamplingNearest2d_launch_fail(self, device):
        m = nn.Upsample(scale_factor=2)
        # launch grid_y == 2**16 (larger than maximum y-dimension limit 65535)
        inp = torch.rand(1, 1, 2**15, 2**8, device=device)
        out = m(inp)


class TestLayernorm(NNTestCase):
    def _test_LayerNorm_general(self, device, dtype=torch.float):
        for i in range(2, 6):
            
            shape = torch.randint(3, 6, (i,), dtype=torch.long).tolist()
            x = torch.empty(*shape, device=device, dtype=dtype).uniform_(0, 10)
            normalized_ndim = random.randint(1, i - 1)  # inclusive
            normalized_shape = shape[-normalized_ndim:]
            unnormalized_shape = shape[:-normalized_ndim]

            # test that LN normalizes to mean 0 and stddev 1
            ln = nn.LayerNorm(normalized_shape, eps=0).to(device, dtype)
            ln.weight.data.fill_(1)
            ln.bias.data.fill_(0)
            output = ln(x)
            out_reshaped = output.view(*(unnormalized_shape + [-1]))
            mean = out_reshaped.mean(-1)
            var = out_reshaped.to('cpu').var(-1, unbiased=False)

            self.assertAlmostEqual(torch.abs(mean.data).mean(), 0, delta=1e-5)
            self.assertAlmostEqual(torch.abs(var.data).mean(), 1, delta=1e-5)
            # test that LN applies weight and bias correctly
            scale, bias = torch.empty(2).uniform_(0.2, 2).tolist()
            ln.weight.data.fill_(scale)
            ln.bias.data.fill_(bias)
            output = ln(x)
            out_reshaped = output.view(*(unnormalized_shape + [-1]))
            mean = out_reshaped.mean(-1)
            # var = out_reshaped.var(-1, unbiased=False)
            var = out_reshaped.to('cpu').var(-1, unbiased=False)
            self.assertAlmostEqual(torch.abs(mean.data).mean(), bias, delta=1e-5)
            self.assertAlmostEqual(torch.abs(var.data).mean(), scale ** 2, delta=1e-5)

        bad_norm_shape_input_shape = {
            (): (),
            (2, 3): (3,),
            (2,): (1, 2, 3),
            (10,): (2, 3),
            10: (2, 3),
        }
        for norm_shape, input_shape in bad_norm_shape_input_shape.items():
            ln = nn.LayerNorm(norm_shape)
            input = torch.empty(input_shape, device=device, dtype=dtype).uniform_(0, 10)
            self.assertRaises(RuntimeError, lambda: ln(input))

    def _test_LayerNorm_cuda_half(self, device):
        # input = torch.empty(2, 3, 3, 2, device=device, dtype=torch.half).random_(1, 10).requires_grad_(True)
        input = torch.empty(2, 3, 3, 2, device='cpu', dtype=torch.half).random_(1, 10).requires_grad_(True)
        input = input.to(device)
        m = nn.LayerNorm([3, 2]).to(device, torch.half)
        output = m(input)
        output.sum().backward()
        self.assertEqual(output.type(), input.type())

    def test_LayerNorm_general(self, device):
        self._test_LayerNorm_general(device)

        if self.device_type == 'npu':
            self._test_LayerNorm_cuda_half(device)


def add_test(test, decorator=None, clsname=TestSigmoid):
    def add(test_name, fn):
        if hasattr(clsname, test_name):
            raise RuntimeError('Found two tests with the same name: ' + test_name)
        if decorator is not None:
            fn = decorator(fn)
        setattr(clsname, test_name, fn)

    test_name = test.get_name()
    add(test_name, lambda self, test=test: test(self))
    npu_test_name = test_name + '_npu'
    cuda_test_name = test_name + '_cuda'
    # With dtype enable, it's good enough to test against three floating types
    kwargs = {}
    if 'extra_args' in get_function_arglist(test.test_npu):
        kwargs['extra_args'] = test.extra_args

    if 'dtype' in get_function_arglist(test.test_npu):
        add(npu_test_name + '_float', lambda self,
                                             test=test, kwargs=kwargs: test.test_npu(self, dtype=torch.float,
                                                                                     **kwargs))
        add(npu_test_name + '_double', lambda self,
                                              test=test, kwargs=kwargs: test.test_npu(self, dtype=torch.double,
                                                                                      **kwargs))

        def test_half(self, test=test, kwargs=kwargs):
            test.test_npu(self, dtype=torch.half, **kwargs)

        if getattr(test, 'check_half', True):
            add(npu_test_name + '_half', test_half)

        def test_bfloat16(self, test=test, kwargs=kwargs):
            test.test_npu(self, dtype=torch.bfloat16, **kwargs)

        if getattr(test, 'check_bfloat16', True):
            add(npu_test_name + '_bfloat16', test_bfloat16)

    else:
        add(npu_test_name, lambda self, test=test, kwargs=kwargs: test.test_npu(self, **kwargs))
    kwargs = {}
    if 'extra_args' in get_function_arglist(test.test_cuda):
        kwargs['extra_args'] = test.extra_args

    if 'dtype' in get_function_arglist(test.test_cuda):
        add(cuda_test_name + '_float', lambda self,
                                              test=test, kwargs=kwargs: test.test_cuda(self, dtype=torch.float,
                                                                                       **kwargs))
        add(cuda_test_name + '_double', lambda self,
                                               test=test, kwargs=kwargs: test.test_cuda(self, dtype=torch.double,
                                                                                        **kwargs))

        def test_half(self, test=test, kwargs=kwargs):
            test.test_cuda(self, dtype=torch.half, **kwargs)

        if getattr(test, 'check_half', True):
            add(cuda_test_name + '_half', test_half)

        def test_bfloat16(self, test=test, kwargs=kwargs):
            test.test_cuda(self, dtype=torch.bfloat16, **kwargs)

        if getattr(test, 'check_bfloat16', True):
            add(cuda_test_name + '_bfloat16', test_bfloat16)

    else:
        add(cuda_test_name, lambda self, test=test, kwargs=kwargs: test.test_cuda(self, **kwargs))


def bce_with_logistic_legacy_enum_test():
    t = Variable(torch.randn(15, 10).gt(0).double())
    sigmoid = nn.Sigmoid()
    return dict(
        fullname='BCEWithLogitsLoss_legacy_enum',
        constructor=wrap_functional(
            lambda i: F.binary_cross_entropy_with_logits(i, t.type_as(i), reduce=False)),
        cpp_function_call='''F::binary_cross_entropy_with_logits(
            i, t.to(i.options()), F::BinaryCrossEntropyWithLogitsFuncOptions().reduction(torch::kNone))''',
        input_fn=lambda: torch.rand(15, 10).clamp_(2.8e-2, 1 - 2.8e-2),
        cpp_var_map={'i': '_get_input()', 't': t},
        reference_fn=lambda i, *_: -(t * sigmoid(i).log() + (1 - t) * (1 - sigmoid(i)).log()),
        check_gradgrad=False,
        pickle=False,
    )


def bce_with_logistic_no_reduce_test():
    t = Variable(torch.randn(15, 10).gt(0).double())
    sigmoid = nn.Sigmoid()
    return dict(
        fullname='BCEWithLogitsLoss_no_reduce',
        constructor=wrap_functional(
            lambda i: F.binary_cross_entropy_with_logits(i, t.type_as(i), reduction='none')),
        cpp_function_call='''F::binary_cross_entropy_with_logits(
            i, t.to(i.options()), F::BinaryCrossEntropyWithLogitsFuncOptions().reduction(torch::kNone))''',
        input_fn=lambda: torch.rand(15, 10).clamp_(2.8e-2, 1 - 2.8e-2),
        cpp_var_map={'i': '_get_input()', 't': t},
        reference_fn=lambda i, *_: -(t * sigmoid(i).log() + (1 - t) * (1 - sigmoid(i)).log()),
        check_gradgrad=False,
        pickle=False,
    )


def bce_with_logistic_no_reduce_scalar_test():
    t = torch.randn(()).gt(0).double()
    sigmoid = nn.Sigmoid()
    return dict(
        fullname='BCEWithLogitsLoss_no_reduce_scalar',
        constructor=wrap_functional(
            lambda i: F.binary_cross_entropy_with_logits(i, t.type_as(i), reduction='none')),
        cpp_function_call='''F::binary_cross_entropy_with_logits(
            i, t.to(i.options()), F::BinaryCrossEntropyWithLogitsFuncOptions().reduction(torch::kNone))''',
        input_fn=lambda: torch.rand(()).clamp_(2.8e-2, 1 - 2.8e-2),
        cpp_var_map={'i': '_get_input()', 't': t},
        reference_fn=lambda i, *_: -(t * sigmoid(i).log() + (1 - t) * (1 - sigmoid(i)).log()),
        check_gradgrad=False,
        pickle=False
    )


sigmoid_tests = [
    dict(
        module_name='Sigmoid',
        input_size=(2, 3, 4, 5),
    ),
    dict(
        module_name='Sigmoid',
        input_size=(),
        desc='scalar',
    ),
]

avg_pool2d_test = [
    dict(
        module_name='AvgPool2d',
        constructor_args=((2, 2),),
        cpp_constructor_args='torch::nn::AvgPool2dOptions({2, 2})',
        input_size=(2, 3, 6, 6),
    ),
    dict(
        module_name='AvgPool2d',
        constructor_args=((2, 2), (2, 2)),
        cpp_constructor_args='torch::nn::AvgPool2dOptions({2, 2}).stride({2, 2})',
        input_size=(2, 3, 6, 6),
        desc='stride',
    ),
    dict(
        module_name='AvgPool2d',
        constructor_args=((2, 2), (2, 2), (1, 1)),
        cpp_constructor_args='torch::nn::AvgPool2dOptions({2, 2}).stride({2, 2}).padding({1, 1})',
        input_size=(2, 3, 6, 6),
        desc='stride_pad',
    ),
    dict(
        fullname='AvgPool2d_divisor',
        constructor=lambda: nn.AvgPool2d((2, 2), divisor_override=1),
        cpp_constructor_args='torch::nn::AvgPool2dOptions({2, 2}).divisor_override(1)',
        input_size=(2, 3, 6, 6),
        check_with_long_tensor=True,
    ),
    dict(
        fullname='AvgPool2d_divisor_stride',
        constructor=lambda: nn.AvgPool2d((2, 2), (2, 2), divisor_override=1),
        cpp_constructor_args='torch::nn::AvgPool2dOptions({2, 2}).stride({2, 2}).divisor_override(1)',
        input_size=(2, 3, 6, 6),
        check_with_long_tensor=True,
    ),
    dict(
        fullname='AvgPool2d_divisor_stride_pad',
        constructor=lambda: nn.AvgPool2d((2, 2), (2, 2), (1, 1), divisor_override=1),
        cpp_constructor_args='torch::nn::AvgPool2dOptions({2, 2}).stride({2, 2}).padding({1, 1}).divisor_override(1)',
        input_size=(2, 3, 6, 6),
        check_with_long_tensor=True,
    ),
]

bce_with_logistic_module_tests = [
    bce_with_logistic_legacy_enum_test(),
    bce_with_logistic_no_reduce_test(),
    bce_with_logistic_no_reduce_scalar_test(),
]

unfold_module_tests = [
    dict(
        fullname='Unfold',
        constructor=lambda: nn.Unfold((2, 2), (1, 1), (0, 0), (1, 1)),
        cpp_constructor_args='torch::nn::UnfoldOptions({2, 2}).dilation({1, 1}).padding({0, 0}).stride({1, 1})',
        input_size=(2, 4, 3, 3),
        check_gradgrad=False,
        test_cuda=True,
    ),

    dict(
        fullname='Unfold_int_input',
        constructor=lambda: nn.Unfold(2, 1, 0, 1),
        cpp_constructor_args='torch::nn::UnfoldOptions(2).dilation(1).padding(0).stride(1)',
        input_size=(2, 4, 3, 3),
        check_gradgrad=False,
        test_cuda=True,
    ),
]

leakyrelu_module_tests = [
    dict(
        module_name='LeakyReLU',
        constructor_args=(0.5,),
        cpp_constructor_args='torch::nn::LeakyReLUOptions().negative_slope(0.5)',
        input_size=(),
        check_inplace=True,
        desc='with_negval_scalar'
    ),
]

softmax_module_tests = [
    dict(
        module_name='Softmax',
        constructor_args=(1,),
        cpp_constructor_args='torch::nn::SoftmaxOptions(1)',
        input_size=(10, 20),
        reference_fn=lambda i, *_: torch.exp(i).div(torch.exp(i).sum(1, True).expand(10, 20)),
    ),
    dict(
        constructor=wrap_functional(F.softmax, dim=-1),
        cpp_options_args='F::SoftmaxFuncOptions(-1)',
        input_size=(2, 128),  # trigger the last-dim algo in CUDA
        fullname='softmax_lastdim',
        pickle=False,
    ),
    dict(
        constructor=wrap_functional(F.softmax, dim=1, dtype=torch.float64),
        cpp_options_args='F::SoftmaxFuncOptions(1).dtype(torch::kFloat64)',
        input_size=(2, 128),
        fullname='softmax_lastdim_dtype',
        pickle=False,
        test_cuda=False
    ),
    dict(
        constructor=wrap_functional(F.softmax, dim=1),
        cpp_options_args='F::SoftmaxFuncOptions(1)',
        input_size=(2, 128, 2, 2),  # trigger special case of spatial CUDA algo
        fullname='softmax_spatial_special',
        pickle=False,
        test_cuda=(not TEST_WITH_ROCM)
    ),
    dict(
        constructor=wrap_functional(F.softmax, dim=1),
        cpp_options_args='F::SoftmaxFuncOptions(1)',
        input_size=(2, 2, 4, 4),  # regular spatial algorithm
        fullname='softmax_spatial',
        pickle=False,
    ),
    dict(
        constructor=wrap_functional(F.softmax, dim=1, dtype=torch.float64),
        cpp_options_args='F::SoftmaxFuncOptions(1).dtype(torch::kFloat64)',
        input_size=(2, 2, 4, 4),  # regular spatial algorithm
        fullname='softmax_spatial_dtype',
        pickle=False,
        test_cuda=False
    ),
    dict(
        constructor=wrap_functional(F.softmax, dim=0),
        cpp_options_args='F::SoftmaxFuncOptions(0)',
        input_size=(2, 3, 4, 5),
        fullname='softmax_functional_dim0',
        test_cuda=False,
        pickle=False,
    ),
    dict(
        constructor=wrap_functional(F.softmax, dim=3),
        cpp_options_args='F::SoftmaxFuncOptions(3)',
        input_size=(2, 3, 4, 5),
        fullname='softmax_functional_dim3',
        test_cuda=False,
        pickle=False,
    ),
    dict(
        constructor=wrap_functional(F.softmax, dim=-1),
        cpp_options_args='F::SoftmaxFuncOptions(-1)',
        input_size=(),
        fullname='softmax_functional_scalar',
        test_cuda=False,
        pickle=False,
    ),
    dict(
        module_name='Softmax',
        constructor_args=(0,),
        cpp_constructor_args='torch::nn::SoftmaxOptions(0)',
        input_size=(),
        reference_fn=lambda i, *_: torch.exp(i).div(torch.exp(i).sum(0, True)),
        desc='scalar',
    ),
]


def runtests(tests, clsname):
    for test_params in tests:
        if 'constructor' not in test_params:
            name = test_params.pop('module_name')
            test_params['constructor'] = getattr(nn, name)
        decorator = test_params.pop('decorator', None)
        test = NewModuleTest(**test_params)
        add_test(test, decorator, clsname=clsname)


runtests(sigmoid_tests, TestSigmoid)
runtests(softmax_module_tests, TestSoftmax)
runtests(avg_pool2d_test, TestAvg_pool2d)
runtests(bce_with_logistic_module_tests, TestBce_with_logistic)
runtests(unfold_module_tests, TestUnfold)
runtests(leakyrelu_module_tests, TestLeakyrelu)

generate_tensor_op_tests(TestGt, tensor_gt_tests)
generate_tensor_op_tests(TestSum, tensor_sum_tests)
generate_tensor_op_tests(TestSub, tensor_sub_tests)
generate_tensor_op_tests(TestSqrt, tensor_sqrt_tests)
generate_tensor_op_tests(TestSplit, tensor_split_tests)
generate_tensor_op_tests(TestSort, tensor_sort_tests)
generate_tensor_op_tests(TestLe, tensor_le_tests)
generate_tensor_op_tests(TestLt, tensor_lt_tests)
generate_tensor_op_tests(TestMax, tensor_max_tests)
generate_tensor_op_tests(TestMin, tensor_min_tests)
generate_tensor_op_tests(TestNe, tensor_ne_tests)
generate_tensor_op_tests(TestPow, tensor_pow_tests)
generate_tensor_op_tests(TestMul, tensor_mul_tests)
generate_tensor_op_tests(TestProd, tensor_prod_tests)
generate_tensor_op_tests(TestRemainder, tensor_remainder_tests)
generate_tensor_op_tests(TestLog, tensor_log_tests)
generate_tensor_op_tests(TestLog2, tensor_log2_tests)
generate_tensor_op_tests(TestNeg, tensor_neg_tests)
generate_tensor_op_tests(TestNonzero, tensor_nonzero_tests)
generate_tensor_op_tests(TestRsqrt, tensor_rsqrt_tests)
generate_tensor_op_tests(TestAbs, tensor_abs_tests)
generate_tensor_op_tests(TestAdd, tensor_add_tests)
generate_tensor_op_tests(TestAddmm, tensor_addmm_tests)
generate_tensor_op_tests(TestBmm, tensor_bmm_tests)
generate_tensor_op_tests(TestClamp, tensor_clamp_tests)
generate_tensor_op_tests(TestDiv, tensor_div_tests)
generate_tensor_op_tests(TestEq, tensor_equ_tests)
generate_tensor_op_tests(TestExp, tensor_exp_tests)
generate_tensor_op_tests(TestFill_, tensor_fill_tests)
generate_tensor_op_tests(TestFloor, tensor_floor_tests)
generate_tensor_op_tests(TestFmod, tensor_fmod_tests)
generate_tensor_op_tests(TestGe, tensor_ge_tests)
generate_tensor_op_tests(TestTopk, tensor_topk_tests)
generate_tensor_op_tests(TestZero_, tensor_zero_tests)

instantiate_device_type_tests(TestSign, globals(), except_for='cpu')
instantiate_device_type_tests(TestRsqrt, globals(), except_for='cpu')
instantiate_device_type_tests(TestLog, globals(), except_for='cpu')
instantiate_device_type_tests(TestLog2, globals(), except_for='cpu')
instantiate_device_type_tests(TestNeg, globals(), except_for='cpu')
instantiate_device_type_tests(TestNonzero, globals(), except_for='cpu')
instantiate_device_type_tests(TestRemainder, globals(), except_for='cpu')
instantiate_device_type_tests(TestProd, globals(), except_for='cpu')
instantiate_device_type_tests(TestMul, globals(), except_for='cpu')
instantiate_device_type_tests(TestPow, globals(), except_for='cpu')
instantiate_device_type_tests(TestNe, globals(), except_for='cpu')
instantiate_device_type_tests(TestMin, globals(), except_for='cpu')
instantiate_device_type_tests(TestMax, globals(), except_for='cpu')
instantiate_device_type_tests(TestLt, globals(), except_for='cpu')
instantiate_device_type_tests(TestGt, globals(), except_for='cpu')
instantiate_device_type_tests(TestLe, globals(), except_for='cpu')
instantiate_device_type_tests(TestMaskedFill, globals(), except_for='cpu')
instantiate_device_type_tests(TestMaskedScatter, globals(), except_for='cpu')
instantiate_device_type_tests(TestMaskedSelect, globals(), except_for='cpu')
instantiate_device_type_tests(TestMatMul, globals(), except_for='cpu')
instantiate_device_type_tests(TestMM, globals(), except_for='cpu')
instantiate_device_type_tests(TestOneslike, globals(), except_for='cpu')
instantiate_device_type_tests(TestRandom, globals(), except_for='cpu')
instantiate_device_type_tests(TestRandperm, globals(), except_for='cpu')
instantiate_device_type_tests(TestReciprocal, globals(), except_for='cpu')
instantiate_device_type_tests(TestMaxPool, globals(), except_for='cpu')
instantiate_device_type_tests(TestNllLoss, globals(), except_for='cpu')
instantiate_device_type_tests(TestRelu, globals(), except_for='cpu')
instantiate_device_type_tests(TestLogSoftmax, globals(), except_for='cpu')
instantiate_device_type_tests(TestHardtanh, globals(), except_for='cpu')
instantiate_device_type_tests(TestPut, globals(), except_for='cpu')
instantiate_device_type_tests(TestAbs, globals(), except_for='cpu')
instantiate_device_type_tests(TestAdd, globals(), except_for='cpu')
instantiate_device_type_tests(TestAddmm, globals(), except_for='cpu')
instantiate_device_type_tests(TestArange, globals(), except_for='cpu')
instantiate_device_type_tests(TestBmm, globals(), except_for='cpu')
instantiate_device_type_tests(TestCat, globals(), except_for="cpu")
instantiate_device_type_tests(TestClamp, globals(), except_for="cpu")
instantiate_device_type_tests(TestDiv, globals(), except_for="cpu")
instantiate_device_type_tests(TestDropout, globals(), except_for="cpu")
instantiate_device_type_tests(TestEq, globals(), except_for="cpu")
instantiate_device_type_tests(TestExp, globals(), except_for="cpu")
instantiate_device_type_tests(TestFill_, globals(), except_for="cpu")
instantiate_device_type_tests(TestFill_diagonal, globals(), except_for="cpu")
instantiate_device_type_tests(TestFloor, globals(), except_for="cpu")
instantiate_device_type_tests(TestFmod, globals(), except_for="cpu")
instantiate_device_type_tests(TestGe, globals(), except_for="cpu")
instantiate_device_type_tests(TestFull, globals(), except_for="cpu")
instantiate_device_type_tests(TestAdaptiveAvgPool2d, globals(), except_for="cpu")
instantiate_device_type_tests(TestIndexput, globals(), except_for="cpu")
instantiate_device_type_tests(TestRepeat, globals(), except_for="cpu")
instantiate_device_type_tests(TestSort, globals(), except_for="cpu")
instantiate_device_type_tests(TestSplit, globals(), except_for="cpu")
instantiate_device_type_tests(TestSqrt, globals(), except_for="cpu")
instantiate_device_type_tests(TestStack, globals(), except_for="cpu")
instantiate_device_type_tests(TestSub, globals(), except_for="cpu")
instantiate_device_type_tests(TestSum, globals(), except_for="cpu")
instantiate_device_type_tests(TestFillDiagonal, globals(), except_for="cpu")
instantiate_device_type_tests(TestTopk, globals(), except_for="cpu")
instantiate_device_type_tests(TestZero_, globals(), except_for="cpu")
instantiate_device_type_tests(TestZeros, globals(), except_for="cpu")
instantiate_device_type_tests(TestEmbedding, globals(), except_for="cpu")
instantiate_device_type_tests(TestUpsamplingNearest2d, globals(), except_for="cpu")
instantiate_device_type_tests(TestLayernorm, globals(), except_for="cpu")

if __name__ == '__main__':
    run_tests()
