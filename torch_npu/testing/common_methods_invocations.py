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

from typing import List
from functools import partial
import unittest

import torch
from torch.testing._internal import common_methods_invocations
from torch.testing._internal.common_dtype import _dispatch_dtypes, floating_and_complex_types_and
from torch.testing._internal.common_methods_invocations import (OpInfo as Of_OpInfo,
                                                                UnaryUfuncInfo as Of_UnaryUfuncInfo,
                                                                BinaryUfuncInfo as Of_BinaryUfuncInfo,
                                                                DecorateInfo,
                                                                wrapper_set_seed,
                                                                sample_inputs_normal_common,
                                                                sample_inputs_binary_cross_entropy_with_logits)


class OpInfo(Of_OpInfo):
    def __init__(
            self,
            name,  # the string name of the function
            dtypes=_dispatch_dtypes((torch.float32,)),
            formats=(2, ),
            dtypesIfNPU=(torch.float16,),
            backward_dtypes=None,
            backward_dtypesIfNPU=None,
            skipSample=None,
            skips=tuple(),
            **kwargs):

        super().__init__(
            name,
            dtypes=dtypes,
            backward_dtypes=backward_dtypes,
            skips=skips,
            **kwargs)

        self.dtypesIfNPU = set(dtypesIfNPU) if dtypesIfNPU is not None else self.dtypes
        self.backward_dtypesIfNPU = set(backward_dtypesIfNPU) if backward_dtypesIfNPU is not None else (
            backward_dtypes if backward_dtypes is not None
            else dtypesIfNPU if dtypesIfNPU is not None
            else dtypes
        )
        self.formats = formats
        self.skipSample = skipSample
    
    def supported_dtypes(self, device_type):
        return self.dtypesIfNPU if device_type == 'npu' else self.dtypes

    def supported_backward_dtypes(self, device_type):
        if not self.supports_autograd:
            return set()

        backward_dtypes = None
        if device_type == 'npu':
            backward_dtypes = self.backward_dtypesIfNPU
        else:
            backward_dtypes = self.backward_dtypes

        allowed_backward_dtypes = floating_and_complex_types_and(torch.bfloat16, torch.float16)
        return set(allowed_backward_dtypes).intersection(backward_dtypes)


class UnaryUfuncInfo(OpInfo, Of_UnaryUfuncInfo):
    def __init__(
            self,
            name,  # the string name of the function
            ref=None,
            sample_inputs_func=common_methods_invocations.sample_inputs_elementwise_unary,
            **kwargs):

        super().__init__(
            name,
            ref=ref,
            sample_inputs_func=sample_inputs_func,
            **kwargs)


class BinaryUfuncInfo(OpInfo, Of_BinaryUfuncInfo):
    def __init__(self,
                 name,
                 sample_inputs_func=common_methods_invocations.sample_inputs_elementwise_binary,
                 **kwargs):
        super().__init__(name,sample_inputs_func=sample_inputs_func, **kwargs)


def sample_inputs_normal_tensor_second(self, device, dtype, requires_grad, **kwargs):
    cases = [
        ([3, 4], 0.3, {}),
        ([3, 55], 0, {}),
        ([5, 6, 7, 8], [5, 6, 7, 8], {})
    ]
    return sample_inputs_normal_common(self, device, dtype, requires_grad, cases, **kwargs)


op_db: List[OpInfo] = [
    UnaryUfuncInfo(
        'abs',
        aliases=('absolute', ),
        dtypes=_dispatch_dtypes((torch.int8, torch.int16, torch.int32, torch.int64, 
        torch.float16, torch.float32, )),
        dtypesIfNPU=_dispatch_dtypes((torch.int8, torch.int16, torch.int32, torch.int64, 
        torch.float16, torch.float32, torch.float64)),
        formats=(0, 3),
        supports_inplace_autograd=False,
        skips=(
            DecorateInfo(unittest.skip("skipped!"), 'TestOps', 'test_correctness', 
            dtypes=[torch.int16, torch.int64]),
        ),
    ),
    UnaryUfuncInfo(
        'acos',
        aliases=('arccos', ),
        dtypes=_dispatch_dtypes((torch.float32, )),
        dtypesIfNPU=_dispatch_dtypes((torch.float16, torch.float32, torch.float64)),
        formats=(2, ),
        skipSample={
            'test_correctness' : (0, 2), 
        },
    ),
    UnaryUfuncInfo(
        'acosh',
        aliases=('arccosh', ),
        dtypes=_dispatch_dtypes((torch.float32, )),
        dtypesIfNPU=_dispatch_dtypes((torch.float32, )),
        formats=(0, ),
        supports_inplace_autograd=False,
        skipSample={
            'test_correctness' : (0, 2),
        },
    ),
    BinaryUfuncInfo(
        'add',
        dtypes=_dispatch_dtypes((torch.float32, )),
        dtypesIfNPU=_dispatch_dtypes((torch.float16, torch.float32)),
        sample_inputs_func=partial(common_methods_invocations.sample_inputs_add_sub, alpha=2),
        formats=(0, 3, 29),
        supports_inplace_autograd=False,
        skipSample={
            'test_variant_consistency_eager' : (8, 18),
        },
    ),
    OpInfo(
        'addbmm',
        dtypes=_dispatch_dtypes((torch.float32, )),
        dtypesIfNPU=_dispatch_dtypes((torch.float32, )),
        sample_inputs_func=common_methods_invocations.sample_inputs_addbmm,
        skips=(
            DecorateInfo(unittest.skip("skipped!"), 'TestOps', 'test_correctness', 
            dtypes=[torch.float32]),
            DecorateInfo(unittest.skip("skipped!"), 'TestOps', 'test_variant_consistency_eager', 
            dtypes=[torch.float32]),
        ),
    ),
    OpInfo(
        'addcdiv',
        dtypes=_dispatch_dtypes((torch.float32, torch.float64, )),
        dtypesIfNPU=_dispatch_dtypes((torch.float16, torch.float32, torch.float64, )),
        sample_inputs_func=common_methods_invocations.sample_inputs_addcmul_addcdiv,
        formats=(2, ),
        supports_inplace_autograd=False,
        skipSample={
            'test_correctness' : (0, 4, 5, 10, 11),
        },
        skips=(
            DecorateInfo(unittest.skip("skipped!"), 'TestOps', 'test_correctness', 
            dtypes=[torch.float16]),
            DecorateInfo(unittest.skip("skipped!"), 'TestOps', 'test_variant_consistency_eager', 
            dtypes=[torch.float32]),
        ),
    ),
    OpInfo(
        'addcmul',
        dtypes=_dispatch_dtypes((torch.float32, )),
        dtypesIfNPU=_dispatch_dtypes((torch.float16, torch.float32, )),
        sample_inputs_func=common_methods_invocations.sample_inputs_addcmul_addcdiv,
        formats=(2, ),
        supports_inplace_autograd=False,
        skipSample={
            'test_correctness' : (0, 1, 3, 5, 8, 9, 11),
            'test_variant_consistency_eager' : (8, 9, 20, 21),
        },
        skips=(
            DecorateInfo(unittest.skip("skipped!"), 'TestOps', 'test_correctness', 
            dtypes=[torch.float16]),
        ),
    ),
    OpInfo(
        'addmm',
        dtypes=_dispatch_dtypes((torch.float32, )),
        dtypesIfNPU=_dispatch_dtypes((torch.float16, torch.float32, )),
        sample_inputs_func=common_methods_invocations.sample_inputs_addmm,
        supports_inplace_autograd=False,
        skips=(
            DecorateInfo(unittest.skip("skipped!"), 'TestOps', 'test_correctness', 
            dtypes=[torch.float16, torch.float32]),
            DecorateInfo(unittest.skip("skipped!"), 'TestOps', 'test_variant_consistency_eager', 
            dtypes=[torch.float16, torch.float32]),
        ),
    ),
    OpInfo(
        'addmv',
        dtypes=_dispatch_dtypes((torch.float32, )),
        dtypesIfNPU=_dispatch_dtypes((torch.float16, torch.float32, )),
        sample_inputs_func=common_methods_invocations.sample_inputs_addmv,
        formats=(0, 3),
        supports_inplace_autograd=False,
        skips=(
            DecorateInfo(unittest.skip("skipped!"), 'TestOps', 'test_correctness', 
            dtypes=[torch.float16, torch.float32]),
            DecorateInfo(unittest.skip("skipped!"), 'TestOps', 'test_variant_consistency_eager', 
            dtypes=[torch.float16, torch.float32]),
        ),
    ),
    OpInfo(
        'addr',
        dtypes=_dispatch_dtypes((torch.int32, torch.float32)),
        dtypesIfNPU=_dispatch_dtypes((torch.int32, torch.float32, )),
        sample_inputs_func=common_methods_invocations.sample_inputs_addr,
        supports_inplace_autograd=False,
        skipSample={
            'test_variant_consistency_eager' : (1, 3, 5, 7, ),
        },
        skips=(
            DecorateInfo(unittest.skip("skipped!"), 'TestOps', 'test_correctness', 
            dtypes=[torch.int32, torch.float32]),
        ),
    ),
    OpInfo(
        'argsort',
        dtypes=_dispatch_dtypes((torch.float32, )),
        dtypesIfNPU=_dispatch_dtypes((torch.float16, torch.float32, )),
        sample_inputs_func=common_methods_invocations.sample_inputs_argsort,
        supports_autograd=False,
        supports_out=False,
        skipSample={
            'test_variant_consistency_eager' : (25, 26, 27, 53, 54, 55),
            'test_correctness' : (0, 25, 26, 27),
        },
    ),
    OpInfo(
        'as_strided',
        dtypes=_dispatch_dtypes((torch.int32, torch.float16, torch.float32)),
        dtypesIfNPU=_dispatch_dtypes((torch.int32, torch.float16, torch.float32)),
        sample_inputs_func=common_methods_invocations.sample_inputs_as_strided,
        formats=(0, 2),
        supports_out=False,
    ),
    UnaryUfuncInfo(
        'asin',
        aliases=('arcsin', ),
        dtypes=_dispatch_dtypes((torch.float32, )),
        dtypesIfNPU=_dispatch_dtypes((torch.float32, )),
        skipSample={
            'test_correctness' : (0, 1, 2, 3,),
        },
    ),
    UnaryUfuncInfo(
        'asinh',
        aliases=('arcsinh', ),
        dtypes=_dispatch_dtypes((torch.float32, )),
        dtypesIfNPU=_dispatch_dtypes((torch.float32, )),
        supports_inplace_autograd=False,
    ),
    UnaryUfuncInfo(
        'atan',
        aliases=('arctan', ),
        dtypes=_dispatch_dtypes((torch.float32, )),
        dtypesIfNPU=_dispatch_dtypes((torch.float32, )),
    ),
    BinaryUfuncInfo(
        'atan2',
        aliases=('arctan2', ),
        dtypes=_dispatch_dtypes((torch.float32, )),
        dtypesIfNPU=_dispatch_dtypes((torch.float16, torch.float32)),
        formats=(0, 2, 3),
        skips=(
            DecorateInfo(unittest.skip("skipped!"), 'TestOps', 'test_correctness', 
            dtypes=[torch.float16]),
        ),
    ),
    UnaryUfuncInfo(
        'atanh',
        aliases=('arctanh', ),
        dtypes=_dispatch_dtypes((torch.float32, )),
        dtypesIfNPU=_dispatch_dtypes((torch.float32, )),
        supports_inplace_autograd=False,
        formats=(0, 3, 4, 29),
        skipSample={
            'test_correctness' : (0, 2, ),
        },
    ),
    OpInfo(
        'baddbmm',
        dtypes=_dispatch_dtypes((torch.float32, )),
        dtypesIfNPU=_dispatch_dtypes((torch.float16, torch.float32)),
        sample_inputs_func=common_methods_invocations.sample_inputs_baddbmm,
        formats=(0, 2, ),
        skips=(
            DecorateInfo(unittest.skip("skipped!"), 'TestOps', 'test_correctness', 
            dtypes=[torch.float16, torch.float32]),
            DecorateInfo(unittest.skip("skipped!"), 'TestOps', 'test_variant_consistency_eager', 
            dtypes=[torch.float16, torch.float32]),
        ),
    ),
    OpInfo(
        'bernoulli',
        dtypes=_dispatch_dtypes((torch.float32, )),
        dtypesIfNPU=_dispatch_dtypes((torch.float16, torch.float32)),
        sample_inputs_func=common_methods_invocations.sample_inputs_bernoulli,
        formats=(0, 3),
        inplace_variant=None,
        skips=(
            DecorateInfo(unittest.skip("skipped!"), 'TestOps', 'test_correctness', 
            dtypes=[torch.float16, torch.float32]),
            DecorateInfo(unittest.skip("skipped!"), 'TestOps', 'test_variant_consistency_eager', 
            dtypes=[torch.float16, torch.float32]),
        ),
    ),
    OpInfo(
        'bincount',
        dtypes=_dispatch_dtypes((torch.int8, torch.int16, )),
        dtypesIfNPU=_dispatch_dtypes((torch.int8, torch.int16, torch.int32, torch.int64, 
        torch.float16, torch.float32)),
        sample_inputs_func=common_methods_invocations.sample_inputs_bincount,
        supports_autograd=False,
        formats=(2, ),
        supports_out=False,
        skips=(
            DecorateInfo(unittest.skip("skipped!"), 'TestOps', 'test_correctness', 
            dtypes=[torch.int16, torch.int32, torch.int64, torch.float16, torch.float32]),
        ),
    ),
    BinaryUfuncInfo(
        'bitwise_and',
        dtypes=_dispatch_dtypes((torch.int8, torch.int16, )),
        dtypesIfNPU=_dispatch_dtypes((torch.int8, torch.int16, )),
        supports_autograd=False,
        formats=(0, 2),
        skips=(
            DecorateInfo(unittest.skip("skipped!"), 'TestOps', 'test_variant_consistency_eager', 
            dtypes=[torch.int8, torch.int16]),
        ),
    ),
    UnaryUfuncInfo(
        'bitwise_not',
        dtypes=_dispatch_dtypes((torch.int8, torch.int16, torch.int32, torch.int64, torch.bool, )),
        dtypesIfNPU=_dispatch_dtypes((torch.int8, torch.int16, torch.int32, torch.int64, torch.bool, )),
        supports_autograd=False,
        formats=(2, ),
    ),
    OpInfo(
        'bmm',
        dtypes=_dispatch_dtypes((torch.float32, )),
        dtypesIfNPU=_dispatch_dtypes((torch.float16, torch.float32)),
        sample_inputs_func=common_methods_invocations.sample_inputs_bmm,
        formats=(0, 3, 29),
        skips=(
            DecorateInfo(unittest.skip("skipped!"), 'TestOps', 'test_correctness', 
            dtypes=[torch.float16, torch.float32]),
        ),
    ),
    OpInfo(
        'cdist',
        dtypes=_dispatch_dtypes((torch.float32, )),
        dtypesIfNPU=_dispatch_dtypes((torch.float16, torch.float32)),
        sample_inputs_func=common_methods_invocations.sample_inputs_cdist,
        formats=(2, ),
        supports_out=False,
        skips=(
            DecorateInfo(unittest.skip("skipped!"), 'TestOps', 'test_correctness', 
            dtypes=[torch.float16, torch.float32]),
            DecorateInfo(unittest.skip("skipped!"), 'TestOps', 'test_variant_consistency_eager', 
            dtypes=[torch.float16, torch.float32]),
        ),
    ),
    OpInfo(
        'clamp',
        aliases=('clip',),
        dtypes=_dispatch_dtypes((torch.float32, )),
        dtypesIfNPU=_dispatch_dtypes((torch.int32, torch.float16, torch.float32)),
        sample_inputs_func=common_methods_invocations.sample_inputs_clamp,
        skips=(
            DecorateInfo(unittest.skip("skipped!"), 'TestOps', 'test_correctness', 
            dtypes=[torch.int32]),
        ),
    ),
    UnaryUfuncInfo(
        'ceil',
        dtypes=_dispatch_dtypes((torch.float32, )),
        dtypesIfNPU=_dispatch_dtypes((torch.float32, )),
        formats=(0, 3, 4, 29),
    ),
    UnaryUfuncInfo(
        'cos',
        dtypes=_dispatch_dtypes((torch.float32, )),
        dtypesIfNPU=_dispatch_dtypes((torch.float32, )),
    ),
    UnaryUfuncInfo(
        'cosh',
        dtypes=_dispatch_dtypes((torch.float32, )),
        dtypesIfNPU=_dispatch_dtypes((torch.float16, torch.float32, )),
        formats=(2, ),
    ),
    OpInfo(
        'nn.functional.conv_transpose2d',
        aten_name='conv_transpose2d',
        aliases=('conv_transpose2d',),
        dtypes=_dispatch_dtypes((torch.int64, torch.float32, torch.float16)),
        dtypesIfNPU=_dispatch_dtypes((torch.float16, torch.float32, )),
        sample_inputs_func=common_methods_invocations.sample_inputs_conv_transpose2d,
        supports_forward_ad=True,
        supports_fwgrad_bwgrad=True,
        skips=(
            DecorateInfo(unittest.skip("skipped!"), 'TestOps', 'test_correctness',
            dtypes=[torch.float16, torch.float32]),
            DecorateInfo(unittest.skip("skipped!"), 'TestOps', 'test_variant_consistency_eager',
            dtypes=[torch.float16, torch.float32]),
        ),
        supports_out=False,),
    OpInfo(
        'cross',
        dtypes=_dispatch_dtypes((torch.float32, )),
        dtypesIfNPU=_dispatch_dtypes((torch.float32, )),
        sample_inputs_func=common_methods_invocations.sample_inputs_cross,
        formats=(2, ),
        skipSample={
            'test_variant_consistency_eager' : (2, 5, ),
        },
        skips=(
            DecorateInfo(unittest.skip("skipped!"), 'TestOps', 'test_correctness', 
            dtypes=[torch.float32]),
        ),
    ),
    OpInfo(
        'nn.functional.ctc_loss',
        dtypes=_dispatch_dtypes((torch.float32, )),
        dtypesIfNPU=_dispatch_dtypes((torch.float32, )),
        sample_inputs_func=common_methods_invocations.sample_inputs_ctc_loss,
        formats=(2, ),
        supports_out=False,
        skips=(
            DecorateInfo(unittest.skip("skipped!"), 'TestOps', 'test_correctness', 
            dtypes=[torch.float32]),
        ),
    ),
    UnaryUfuncInfo(
        'isnan',
        dtypes=_dispatch_dtypes((torch.float32, )),
        dtypesIfNPU=_dispatch_dtypes((torch.float32, )),
        supports_autograd=False,
        supports_out=False,
    ),
    OpInfo(
        'diag',
        dtypes=_dispatch_dtypes((torch.float32, )),
        dtypesIfNPU=_dispatch_dtypes((torch.float16, torch.float32)),
        sample_inputs_func=common_methods_invocations.sample_inputs_diag,
        formats=(2, ),
        skips=(
            DecorateInfo(unittest.skip("skipped!"), 'TestOps', 'test_correctness', 
            dtypes=[torch.float16, torch.float32]),
            DecorateInfo(unittest.skip("skipped!"), 'TestOps', 'test_variant_consistency_eager', 
            dtypes=[torch.float16, torch.float32]),
        ),
    ),
    BinaryUfuncInfo(
        'div',
        aliases=('divide',),
        dtypes=_dispatch_dtypes((torch.int32, torch.float32)),
        dtypesIfNPU=_dispatch_dtypes((torch.int32, torch.float16, torch.float32)),
        formats=(0, 3, 29),
        skipSample={
            'test_variant_consistency_eager' : (8, 17, ),
        },
        skips=(
            DecorateInfo(unittest.skip("skipped!"), 'TestOps', 'test_correctness', 
            dtypes=[torch.int32, torch.float16, torch.float32]),
        ),
    ),
    OpInfo(
        'dot',
        dtypes=_dispatch_dtypes((torch.float32, )),
        dtypesIfNPU=_dispatch_dtypes((torch.float16, torch.float32)),
        sample_inputs_func=common_methods_invocations.sample_inputs_dot_vdot,
        formats=(2, ),
    ),
    OpInfo(
        'nn.functional.dropout',
        dtypes=_dispatch_dtypes((torch.float32, )),
        dtypesIfNPU=_dispatch_dtypes((torch.float16, torch.float32)),
        sample_inputs_func=common_methods_invocations.sample_inputs_dropout,
        formats=(2, ),
        supports_out=False,
        inplace_variant=lambda input, *args, **kwargs:
            wrapper_set_seed(torch.nn.functional.dropout, input, *args, **kwargs, inplace=True),
        skips=(
            DecorateInfo(unittest.skip("skipped!"), 'TestOps', 'test_correctness', 
            dtypes=[torch.float16, torch.float32]),
            DecorateInfo(unittest.skip("skipped!"), 'TestOps', 'test_variant_consistency_eager', 
            dtypes=[torch.float32]),
        ),
    ),
    OpInfo(
        'nn.functional.embedding',
        dtypes=_dispatch_dtypes((torch.float32, )),
        dtypesIfNPU=_dispatch_dtypes((torch.float16, torch.float32)),
        sample_inputs_func=common_methods_invocations.sample_inputs_embedding,
        formats=(0, 29),
        supports_out=False,
        skips=(
            DecorateInfo(unittest.skip("skipped!"), 'TestOps', 'test_correctness', 
            dtypes=[torch.float16, torch.float32]),
            DecorateInfo(unittest.skip("skipped!"), 'TestOps', 'test_variant_consistency_eager', 
            dtypes=[torch.float16, torch.float32]),
        ),
    ),
    BinaryUfuncInfo(
        'eq',
        aliases=('equal',),
        dtypes=_dispatch_dtypes((torch.float32, )),
        dtypesIfNPU=_dispatch_dtypes((torch.float16, torch.float32)),
        sample_inputs_func=common_methods_invocations.sample_inputs_comparison_ops,
        supports_autograd=False,
        formats=(0, 3),
        skips=(
            DecorateInfo(unittest.skip("skipped!"), 'TestOps', 'test_variant_consistency_eager', 
            dtypes=[torch.float16, torch.float32]),
        ),
    ),
    UnaryUfuncInfo(
        'erf',
        dtypes=_dispatch_dtypes((torch.float32, )),
        dtypesIfNPU=_dispatch_dtypes((torch.float32, )),
        formats=(0, 3, 4, 29),
        skipSample={
            'test_correctness' : (0, ),
        },
    ),   
    UnaryUfuncInfo(
        'erfc',
        aliases=('special.erfc', ),
        dtypes=_dispatch_dtypes((torch.float32, )),
        dtypesIfNPU=_dispatch_dtypes((torch.float32, )),
        formats=(0, 2, 3, 4, 30),
        skipSample={
            'test_correctness' : (0, ),
        },
    ),  
    UnaryUfuncInfo(
        'erfinv',
        aliases=('special.erfinv', ),
        dtypes=_dispatch_dtypes((torch.float32, )),
        dtypesIfNPU=_dispatch_dtypes((torch.float16, torch.float32, )),
        formats=(2, ),
        skipSample={
            'test_correctness' : (0, 2, ),
            'test_variant_consistency_eager' : (3, ),
        },
        skips=(
            DecorateInfo(unittest.skip("skipped!"), 'TestOps', 'test_out', dtypes=[torch.float32]),
        ),  
    ),  
    UnaryUfuncInfo(
        'exp',
        dtypes=_dispatch_dtypes((torch.float32, )),
        dtypesIfNPU=_dispatch_dtypes((torch.float16, torch.float32, )),
        formats=(0, 3),
    ),  
    UnaryUfuncInfo(
        'exp2',
        aliases=('special.exp2', ),
        dtypes=_dispatch_dtypes((torch.float32, )),
        dtypesIfNPU=_dispatch_dtypes((torch.float16, torch.float32, )),
        formats=(0, 2, 3),
    ),  
    UnaryUfuncInfo(
        'expm1',
        aliases=('special.expm1', ),
        dtypes=_dispatch_dtypes((torch.float32, )),
        dtypesIfNPU=_dispatch_dtypes((torch.float16, torch.float32, )),
        formats=(0, 3, 4, 29),
        skipSample={
            'test_correctness' : (0, 2, ),
        },
    ), 
    OpInfo(
        'flip',
        dtypes=_dispatch_dtypes((torch.int32, torch.float32)),
        dtypesIfNPU=_dispatch_dtypes((torch.int32, torch.float32)),
        sample_inputs_func=common_methods_invocations.sample_inputs_flip,
        supports_out=False,
    ),
    UnaryUfuncInfo(
        'floor',
        dtypes=_dispatch_dtypes((torch.float32, )),
        dtypesIfNPU=_dispatch_dtypes((torch.float16, torch.float32, )),
        formats=(0, 3),
    ), 
    BinaryUfuncInfo(
        'fmod',
        dtypes=_dispatch_dtypes((torch.float32, )),
        dtypesIfNPU=_dispatch_dtypes((torch.float16, torch.float32)),
        formats=(0, 3),
        skips=(
            DecorateInfo(unittest.skip("skipped!"), 'TestOps', 'test_correctness', 
            dtypes=[torch.float16, torch.float32]),
            DecorateInfo(unittest.skip("skipped!"), 'TestOps', 'test_variant_consistency_eager', 
            dtypes=[torch.float32]),
        ),        
    ),
    UnaryUfuncInfo(
        'frac',
        dtypes=_dispatch_dtypes((torch.float32, )),
        dtypesIfNPU=_dispatch_dtypes((torch.float32, )),
        formats=(2, ),
    ), 
    OpInfo(
        'gather',
        dtypes=_dispatch_dtypes((torch.int16, torch.int32, torch.float16, torch.float32)),
        dtypesIfNPU=_dispatch_dtypes((torch.int16, torch.int32, torch.float16, torch.float32)),
        sample_inputs_func=common_methods_invocations.sample_inputs_gather,
        skips=(
            DecorateInfo(unittest.skip("skipped!"), 'TestOps', 'test_correctness', 
            dtypes=[torch.int16, torch.int32, torch.float16, torch.float32]),
            DecorateInfo(unittest.skip("skipped!"), 'TestOps', 'test_variant_consistency_eager', 
            dtypes=[torch.int16, torch.int32, torch.float16, torch.float32]),
        ),
    ),
    BinaryUfuncInfo(
        'ge',
        aliases=('greater_equal',),
        dtypes=_dispatch_dtypes((torch.float32, )),
        dtypesIfNPU=_dispatch_dtypes((torch.float16, torch.float32)),
        sample_inputs_func=common_methods_invocations.sample_inputs_comparison_ops,
        supports_autograd=False,
    ),
    OpInfo(
        'nn.functional.gelu',
        dtypes=_dispatch_dtypes((torch.float32, )),
        dtypesIfNPU=_dispatch_dtypes((torch.float16, torch.float32)),
        sample_inputs_func=common_methods_invocations.sample_inputs_gelu,
        formats=(2, ),
        supports_out=False,
        skips=(
            DecorateInfo(unittest.skip("skipped!"), 'TestOps', 'test_correctness', 
            dtypes=[torch.float16, torch.float32]),
        ),
    ),
    OpInfo(
        'nn.functional.glu',
        dtypes=_dispatch_dtypes((torch.float32, )),
        dtypesIfNPU=_dispatch_dtypes((torch.float16, torch.float32)),
        sample_inputs_func=common_methods_invocations.sample_inputs_glu,
        formats=(2, ),
        supports_out=False,
        skips=(
            DecorateInfo(unittest.skip("skipped!"), 'TestOps', 'test_correctness', 
            dtypes=[torch.float16, torch.float32]),
        ),
    ),
    BinaryUfuncInfo(
        'gt',
        aliases=('greater',),
        dtypes=_dispatch_dtypes((torch.int16, torch.float32)),
        dtypesIfNPU=_dispatch_dtypes((torch.int16, torch.float16, torch.float32)),
        sample_inputs_func=common_methods_invocations.sample_inputs_comparison_ops,
        supports_autograd=False,
        formats=(0, 2),
    ),
    OpInfo(
        'nn.functional.hardshrink',
        dtypes=_dispatch_dtypes((torch.float32, )),
        dtypesIfNPU=_dispatch_dtypes((torch.float16, torch.float32)),
        sample_inputs_func=common_methods_invocations.sample_inputs_hardshrink,
        formats=(2, ),
        supports_out=False,
    ),
    OpInfo(
        'nn.functional.hardswish',
        dtypes=_dispatch_dtypes((torch.float32, )),
        dtypesIfNPU=_dispatch_dtypes((torch.float16, torch.float32)),
        sample_inputs_func=common_methods_invocations.sample_inputs_hardswish,
        formats=(0, 3, 29),
        supports_out=False,
        skipSample={
            'test_correctness' : (0, 3, ),
        },
    ),
    UnaryUfuncInfo(
        'nn.functional.hardsigmoid',
        dtypes=_dispatch_dtypes((torch.float32, )),
        dtypesIfNPU=_dispatch_dtypes((torch.float16, torch.float32, )),
        formats=(2, ),
        supports_out=False,
    ), 
    OpInfo(
        'nn.functional.hardtanh',
        dtypes=_dispatch_dtypes((torch.float32, )),
        dtypesIfNPU=_dispatch_dtypes((torch.float16, torch.float32)),
        sample_inputs_func=common_methods_invocations.sample_inputs_hardtanh,
        formats=(0, 3, 4, 29),
        supports_out=False,
    ),
    OpInfo(
        'index_add',
        dtypes=_dispatch_dtypes((torch.int8, torch.float16, torch.float32)),
        dtypesIfNPU=_dispatch_dtypes((torch.int8, torch.int32, torch.float16, torch.float32)),
        sample_inputs_func=common_methods_invocations.sample_inputs_index,
        formats=(2, 3, 4),
        skips=(
            DecorateInfo(unittest.skip("skipped!"), 'TestOps', 'test_correctness', 
            dtypes=[torch.int8, torch.int32, torch.float16, torch.float32]),
            DecorateInfo(unittest.skip("skipped!"), 'TestOps', 'test_variant_consistency_eager', 
            dtypes=[torch.int8, torch.int32, torch.float16, torch.float32]),
            
        ),
    ),
    OpInfo(
        'index_copy',
        dtypes=_dispatch_dtypes((torch.float32, )),
        dtypesIfNPU=_dispatch_dtypes((torch.float32, )),
        sample_inputs_func=common_methods_invocations.sample_inputs_index,
        formats=(2, ),
        supports_inplace_autograd=False,
        supports_out=False,
        skips=(
            DecorateInfo(unittest.skip("skipped!"), 'TestOps', 'test_correctness', 
            dtypes=[torch.float32]),
            DecorateInfo(unittest.skip("skipped!"), 'TestOps', 'test_variant_consistency_eager', 
            dtypes=[torch.float32]),
        ),
    ),
    OpInfo(
        'index_put',
        dtypes=_dispatch_dtypes((torch.float32, )),
        dtypesIfNPU=_dispatch_dtypes((torch.float16, torch.float32)),
        sample_inputs_func=common_methods_invocations.sample_inputs_index_put,
        supports_inplace_autograd=False,
        supports_out=False,
    ),
    OpInfo(
        'inverse',
        dtypes=_dispatch_dtypes((torch.float32, )),
        dtypesIfNPU=_dispatch_dtypes((torch.float32, )),
        sample_inputs_func=common_methods_invocations.sample_inputs_linalg_invertible,
        formats=(0, 3),
        skipSample={
            'test_correctness' : (0, 4, 6, ),
        },
    ),
    BinaryUfuncInfo(
        'isclose',
        dtypes=_dispatch_dtypes((torch.int32, torch.float32)),
        dtypesIfNPU=_dispatch_dtypes((torch.int32, torch.float16, torch.float32)),
        sample_inputs_func=common_methods_invocations.sample_inputs_isclose,
        supports_autograd=False,
        supports_out=False,
        skips=(
            DecorateInfo(unittest.skip("skipped!"), 'TestOps', 'test_correctness', 
            dtypes=[torch.int32, torch.float16, torch.float32]),
            DecorateInfo(unittest.skip("skipped!"), 'TestOps', 'test_variant_consistency_eager', 
            dtypes=[torch.int32, torch.float16, torch.float32]),
        ),
    ),
    UnaryUfuncInfo(
        'isfinite',
        dtypes=_dispatch_dtypes((torch.int16, torch.int32, torch.int64, torch.float32, )),
        dtypesIfNPU=_dispatch_dtypes((torch.int16, torch.int32, torch.int64, torch.float32, )),
        supports_autograd=False,
        supports_out=False,
    ), 
    OpInfo(
        'nn.functional.kl_div',
        dtypes=_dispatch_dtypes((torch.float32, )),
        dtypesIfNPU=_dispatch_dtypes((torch.float16, torch.float32)),
        sample_inputs_func=common_methods_invocations.sample_inputs_kl_div,
        supports_out=False,
        skips=(
            DecorateInfo(unittest.skip("skipped!"), 'TestOps', 'test_correctness', 
            dtypes=[torch.float16, torch.float32]),
        ),
    ),
    OpInfo(
        'kthvalue',
        dtypes=_dispatch_dtypes((torch.int32, torch.float32)),
        dtypesIfNPU=_dispatch_dtypes((torch.int32, torch.float16, torch.float32)),
        sample_inputs_func=common_methods_invocations.sample_inputs_kthvalue,
        formats=(2, ),
        skips=(
            DecorateInfo(unittest.skip("skipped!"), 'TestOps', 'test_correctness', 
            dtypes=[torch.int32, torch.float16, torch.float32]),
            DecorateInfo(unittest.skip("skipped!"), 'TestOps', 'test_variant_consistency_eager', 
            dtypes=[torch.int32, torch.float16, torch.float32]),
        ),
    ),
    OpInfo(
        'nn.functional.layer_norm',
        aliases=('layer_norm', ),
        dtypes=_dispatch_dtypes((torch.float32, )),
        dtypesIfNPU=_dispatch_dtypes((torch.float16, torch.float32)),
        sample_inputs_func=common_methods_invocations.sample_inputs_layer_norm,
        formats=(0, 2, 3, 29),
        supports_out=False,
        skips=(
            DecorateInfo(unittest.skip("skipped!"), 'TestOps', 'test_correctness', 
            dtypes=[torch.float16, torch.float32]),
            DecorateInfo(unittest.skip("skipped!"), 'TestOps', 'test_variant_consistency_eager', 
            dtypes=[torch.float16, torch.float32]),
        ),
    ),
    OpInfo(
        'nn.functional.leaky_relu',
        dtypes=_dispatch_dtypes((torch.float32, )),
        dtypesIfNPU=_dispatch_dtypes((torch.float16, torch.float32)),
        sample_inputs_func=common_methods_invocations.sample_inputs_leaky_relu,
        formats=(0, 3),
        supports_out=False,
        skips=(
            DecorateInfo(unittest.skip("skipped!"), 'TestOps', 'test_correctness', 
            dtypes=[torch.float16, torch.float32]),
        ),
    ),
    BinaryUfuncInfo(
        'le',
        aliases=('less_equal',),
        dtypes=_dispatch_dtypes((torch.float16, torch.float32)),
        dtypesIfNPU=_dispatch_dtypes((torch.float16, torch.float32)),
        sample_inputs_func=common_methods_invocations.sample_inputs_comparison_ops,
        supports_autograd=False,
        formats=(0, 3),
    ),
    OpInfo(
        'lerp',
        dtypes=_dispatch_dtypes((torch.float32, )),
        dtypesIfNPU=_dispatch_dtypes((torch.float16, torch.float32)),
        sample_inputs_func=common_methods_invocations.sample_inputs_lerp,
        formats=(2, ),
        skips=(
            DecorateInfo(unittest.skip("skipped!"), 'TestOps', 'test_correctness', 
            dtypes=[torch.float16, torch.float32]),
            DecorateInfo(unittest.skip("skipped!"), 'TestOps', 'test_variant_consistency_eager', 
            dtypes=[torch.float16, torch.float32]),
        ),
    ),
    OpInfo(
        'linalg.svd',
        dtypes=_dispatch_dtypes((torch.float32, )),
        dtypesIfNPU=_dispatch_dtypes((torch.float32, )),
        sample_inputs_func=common_methods_invocations.sample_inputs_svd,
        skips=(
            DecorateInfo(unittest.skip("skipped!"), 'TestOps', 'test_correctness', 
            dtypes=[torch.float32]),
        ),
    ),
    OpInfo(
        'nn.functional.linear',
        dtypes=_dispatch_dtypes((torch.float32,)),
        dtypesIfNPU=_dispatch_dtypes((torch.float16, torch.float32,)),
        sample_inputs_func=common_methods_invocations.sample_inputs_linear,
        supports_out=False,
        skips=(
            DecorateInfo(unittest.skip("skipped!"), 'TestOps', 'test_correctness', 
            dtypes=[torch.float16, torch.float32]),
            DecorateInfo(unittest.skip("skipped!"), 'TestOps', 'test_variant_consistency_eager', 
            dtypes=[torch.float16, torch.float32]),
        ),
    ),
    UnaryUfuncInfo(
        'log',
        dtypes=_dispatch_dtypes((torch.float32, )),
        dtypesIfNPU=_dispatch_dtypes((torch.float16, torch.float32, )),
        formats=(3, ),
        skipSample={
            'test_correctness' : (0, 2, ),
        },
    ), 
    UnaryUfuncInfo(
        'log10',
        dtypes=_dispatch_dtypes((torch.float32, )),
        dtypesIfNPU=_dispatch_dtypes((torch.float16, torch.float32, )),
        formats=(0, 3),
        skips=(
            DecorateInfo(unittest.skip("skipped!"), 'TestOps', 'test_correctness', 
            dtypes=[torch.float16, torch.float32]),
        ),
    ), 
    UnaryUfuncInfo(
        'log1p',
        dtypes=_dispatch_dtypes((torch.float32, )),
        dtypesIfNPU=_dispatch_dtypes((torch.float16, torch.float32, )),
        formats=(0, 2, 4, 29),
        skipSample={
            'test_correctness' : (0, 2, ),
        },
    ), 
    UnaryUfuncInfo(
        'log2',
        dtypes=_dispatch_dtypes((torch.float32, )),
        dtypesIfNPU=_dispatch_dtypes((torch.float16, torch.float32, )),
        formats=(0, 3),
        skipSample={
            'test_correctness' : (0, 2, ),
        },
    ), 
    OpInfo(
        'log_softmax',
        aliases=('special.log_softmax', 'nn.functional.log_softmax'),
        dtypes=_dispatch_dtypes((torch.float32, )),
        dtypesIfNPU=_dispatch_dtypes((torch.float16, torch.float32)),
        sample_inputs_func=common_methods_invocations.sample_inputs_softmax_variant,
        formats=(0, 3),
        supports_out=False,
        skips=(
            DecorateInfo(unittest.skip("skipped!"), 'TestOps', 'test_correctness', 
            dtypes=[torch.float16]),
        ),
    ),
    OpInfo(
        'logaddexp',
        dtypes=_dispatch_dtypes((torch.float32, )),
        dtypesIfNPU=_dispatch_dtypes((torch.float16, torch.float32)),
        sample_inputs_func=common_methods_invocations.sample_inputs_masked_select,
        skips=(
            DecorateInfo(unittest.skip("skipped!"), 'TestOps', 'test_correctness', 
            dtypes=[torch.float16, torch.float32]),
            DecorateInfo(unittest.skip("skipped!"), 'TestOps', 'test_variant_consistency_eager', 
            dtypes=[torch.float16, torch.float32]),
            DecorateInfo(unittest.skip("skipped!"), 'TestOps', 'test_out', 
            dtypes=[torch.float16, torch.float32]),
        ),
    ),
    BinaryUfuncInfo(
        'logical_and',
        dtypes=_dispatch_dtypes((torch.bool, )),
        dtypesIfNPU=_dispatch_dtypes((torch.bool, )),
        supports_autograd=False,
    ),  
    # np.int8 np.int32 np.uint8 np.float16 np.float32 np.bool
    UnaryUfuncInfo(
        'logical_not',
        dtypes=_dispatch_dtypes((torch.bool, )),
        dtypesIfNPU=_dispatch_dtypes((torch.bool, )),
        supports_autograd=False,
    ),  
    BinaryUfuncInfo(
        'logical_or',
        dtypes=_dispatch_dtypes((torch.bool, )),
        dtypesIfNPU=_dispatch_dtypes((torch.bool, )),
        supports_autograd=False,
    ),  
    UnaryUfuncInfo(
        'nn.functional.logsigmoid',
        dtypes=_dispatch_dtypes((torch.float32, )),
        dtypesIfNPU=_dispatch_dtypes((torch.float16, torch.float32)),
        supports_out=False,
    ), 
    OpInfo(
        'logsumexp',
        dtypes=_dispatch_dtypes((torch.float32, )),
        dtypesIfNPU=_dispatch_dtypes((torch.float16, torch.float32)),
        sample_inputs_func=common_methods_invocations.sample_inputs_logsumexp,
        skips=(
            DecorateInfo(unittest.skip("skipped!"), 'TestOps', 'test_correctness', 
            dtypes=[torch.float16, torch.float32]),
            DecorateInfo(unittest.skip("skipped!"), 'TestOps', 'test_variant_consistency_eager', 
            dtypes=[torch.float16, torch.float32]),
            DecorateInfo(unittest.skip("skipped!"), 'TestOps', 'test_out', 
            dtypes=[torch.float16, torch.float32]),
        ),
    ),
    BinaryUfuncInfo(
        'lt',
        aliases=('less',),
        dtypes=_dispatch_dtypes((torch.float32, )),
        dtypesIfNPU=_dispatch_dtypes((torch.float16, torch.float32)),
        sample_inputs_func=common_methods_invocations.sample_inputs_comparison_ops,
        supports_autograd=False,
        formats=(0, 2, 3),
    ),
    OpInfo(
        'masked_fill',
        dtypes=_dispatch_dtypes((torch.int32, torch.int64, torch.float32)),
        dtypesIfNPU=_dispatch_dtypes((torch.int32, torch.int64, torch.float16, torch.float32)),
        sample_inputs_func=common_methods_invocations.sample_inputs_masked_fill,
        supports_out=False,
    ),
    OpInfo(
        'masked_scatter',
        dtypes=_dispatch_dtypes((torch.int32, torch.int64, torch.float32)),
        dtypesIfNPU=_dispatch_dtypes((torch.int32, torch.int64, torch.float16, torch.float32)),
        sample_inputs_func=common_methods_invocations.sample_inputs_masked_scatter,
        supports_out=False,
        skipSample={
            'test_variant_consistency_eager' : (3, 7, ),
        },
    ),
    OpInfo(
        'masked_select',
        dtypes=_dispatch_dtypes((torch.int32, torch.int64, torch.float32)),
        dtypesIfNPU=_dispatch_dtypes((torch.int32, torch.int64, torch.float16, torch.float32)),
        sample_inputs_func=common_methods_invocations.sample_inputs_masked_select,
        formats=(0, 2),
        skips=(
            DecorateInfo(unittest.skip("skipped!"), 'TestOps', 'test_variant_consistency_eager', 
            dtypes=[torch.float32]),
        ),
    ),
    OpInfo(
        'matmul',
        dtypes=_dispatch_dtypes((torch.float32, )),
        dtypesIfNPU=_dispatch_dtypes((torch.float16, torch.float32)),
        sample_inputs_func=common_methods_invocations.sample_inputs_matmul,
        formats=(2, ),
        skips=(
            DecorateInfo(unittest.skip("skipped!"), 'TestOps', 'test_correctness', 
            dtypes=[torch.float16, torch.float32]),
            DecorateInfo(unittest.skip("skipped!"), 'TestOps', 'test_variant_consistency_eager', 
            dtypes=[torch.float16, torch.float32]),
        ),
    ),
    BinaryUfuncInfo(
        'max',
        aliases=('maximum',),
        dtypes=_dispatch_dtypes((torch.float32, )),
        dtypesIfNPU=_dispatch_dtypes((torch.float16, torch.float32)),
        formats=(0, 3, 4, 29),
        supports_out=False,
        skipSample={
            'test_correctness' : (2, 3, 7, ),
        },
    ),
    OpInfo(
        'median',
        dtypes=_dispatch_dtypes((torch.float32, )),
        dtypesIfNPU=_dispatch_dtypes((torch.float16, torch.float32)),
        sample_inputs_func=partial(common_methods_invocations.sample_inputs_reduction, 
        supports_multiple_dims=False),
        supports_out=False,
        skips=(
            DecorateInfo(unittest.skip("skipped!"), 'TestOps', 'test_correctness', 
            dtypes=[torch.float16, torch.float32]),
            DecorateInfo(unittest.skip("skipped!"), 'TestOps', 'test_variant_consistency_eager', 
            dtypes=[torch.float16, torch.float32]),
        ),
    ),
    BinaryUfuncInfo(
        'min',
        aliases=('minimum',),
        dtypes=_dispatch_dtypes((torch.float32, )),
        dtypesIfNPU=_dispatch_dtypes((torch.float16, torch.float32, torch.int8, torch.int32)),
        formats=(0, 2),
    ),
    UnaryUfuncInfo(
        'nn.functional.mish',
        dtypes=_dispatch_dtypes((torch.float32, )),
        dtypesIfNPU=_dispatch_dtypes((torch.float16, torch.float32)),
        supports_out=False,
        skips=(
            DecorateInfo(unittest.skip("skipped!"), 'TestOps', 'test_correctness', 
            dtypes=[torch.float16, torch.float32]),
            DecorateInfo(unittest.skip("skipped!"), 'TestOps', 'test_variant_consistency_eager', 
            dtypes=[torch.float16, torch.float32]),
        ),
    ), 
    OpInfo(
        'mm',
        dtypes=_dispatch_dtypes((torch.float32, )),
        dtypesIfNPU=_dispatch_dtypes((torch.float16, torch.float32)),
        sample_inputs_func=common_methods_invocations.sample_inputs_mm,
        formats=(2, 29),
        skips=(
            DecorateInfo(unittest.skip("skipped!"), 'TestOps', 'test_correctness', 
            dtypes=[torch.float16, torch.float32]),
        ),
    ),
    OpInfo(
        'nn.functional.mse_loss',
        dtypes=_dispatch_dtypes((torch.float32, )),
        dtypesIfNPU=_dispatch_dtypes((torch.float32, )),
        sample_inputs_func=common_methods_invocations.sample_inputs_loss,
        formats=(2, ),
        supports_out=False,
    ),
    BinaryUfuncInfo(
        'mul',
        aliases=('multiply',),
        dtypes=_dispatch_dtypes((torch.int32, torch.float32)),
        dtypesIfNPU=_dispatch_dtypes((torch.int32, torch.float16, torch.float32)),
        formats=(0, 3, 4, 29),
        skipSample={
            'test_variant_consistency_eager' : (8, 17, ),
        },
    ),
    OpInfo(
        'multinomial',
        dtypes=_dispatch_dtypes((torch.float32, )),
        dtypesIfNPU=_dispatch_dtypes((torch.float16, torch.float32, )),
        sample_inputs_func=common_methods_invocations.sample_inputs_multinomial,
        supports_autograd=False,
        method_variant=lambda inp, *args, **kwargs:
            wrapper_set_seed(torch.Tensor.multinomial, inp, *args, **kwargs),
        skips=(
            DecorateInfo(unittest.skip("skipped!"), 'TestOps', 'test_correctness', 
            dtypes=[torch.float16, torch.float32]),
            DecorateInfo(unittest.skip("skipped!"), 'TestOps', 'test_variant_consistency_eager', 
            dtypes=[torch.float16, torch.float32]),
        ),
    ),
    OpInfo(
        'mv',
        dtypes=_dispatch_dtypes((torch.float32, )),
        dtypesIfNPU=_dispatch_dtypes((torch.float32, )),
        sample_inputs_func=common_methods_invocations.sample_inputs_mv,
        skips=(
            DecorateInfo(unittest.skip("skipped!"), 'TestOps', 'test_correctness', 
            dtypes=[torch.float32]),
        ),
    ),
    BinaryUfuncInfo(
        'ne',
        aliases=('not_equal',),
        dtypes=_dispatch_dtypes((torch.float16, torch.float32)),
        dtypesIfNPU=_dispatch_dtypes((torch.float16, torch.float32)),
        sample_inputs_func=common_methods_invocations.sample_inputs_comparison_ops,
        supports_autograd=False,
        formats=(0, 3),
    ),
    UnaryUfuncInfo(
        'neg',
        aliases=('negative',),
        dtypes=_dispatch_dtypes((torch.float16, torch.float32)),
        dtypesIfNPU=_dispatch_dtypes((torch.float16, torch.float32)),
        formats=(0, 3, 29),
    ),
    OpInfo(
        'nn.functional.nll_loss',
        dtypes=_dispatch_dtypes((torch.float32, )),
        dtypesIfNPU=_dispatch_dtypes((torch.float16, torch.float32, )),
        sample_inputs_func=common_methods_invocations.sample_inputs_nll_loss,
        supports_out=False,
    ),
    OpInfo(
        'nonzero',
        dtypes=_dispatch_dtypes((torch.int32, torch.int64, torch.float16, torch.float32)),
        dtypesIfNPU=_dispatch_dtypes((torch.int32, torch.int64, torch.float16, torch.float32)),
        sample_inputs_func=common_methods_invocations.sample_inputs_nonzero,
        supports_autograd=False,
        skips=(
            DecorateInfo(unittest.skip("skipped!"), 'TestOps', 'test_correctness', 
            dtypes=[torch.int32, torch.int64, torch.float16, torch.float32]),
            DecorateInfo(unittest.skip("skipped!"), 'TestOps', 'test_variant_consistency_eager', 
            dtypes=[torch.float32]),
        )
    ),
    OpInfo(
        'norm',
        dtypes=_dispatch_dtypes((torch.float32, )),
        dtypesIfNPU=_dispatch_dtypes((torch.float32, )),
        sample_inputs_func=common_methods_invocations.sample_inputs_norm,
        skipSample={
            'test_correctness' : (3, 6, 7),
        },
        skips=(
            DecorateInfo(unittest.skip("skipped!"), 'TestOps', 'test_correctness', 
            dtypes=[torch.float32]),
        ),
    ),
    # normal has first and second test, imp first only 
    OpInfo(
        'normal',
        dtypes=_dispatch_dtypes((torch.float16, torch.float32)),
        dtypesIfNPU=_dispatch_dtypes((torch.float16, torch.float32)),
        sample_inputs_func=sample_inputs_normal_tensor_second,
        supports_autograd=False,
        formats=(2, ),
        inplace_variant=None,
        skips=(
            DecorateInfo(unittest.skip("skipped!"), 'TestOps', 'test_correctness',
            dtypes=[torch.float16, torch.float32]),
        ),
    ),
    OpInfo(
        'nn.functional.one_hot',
        dtypes=_dispatch_dtypes((torch.int32, )),
        dtypesIfNPU=_dispatch_dtypes((torch.int32, )),
        sample_inputs_func=common_methods_invocations.sample_inputs_one_hot,
        supports_out=False,
        skips=(
            DecorateInfo(unittest.skip("skipped!"), 'TestOps', 'test_correctness', 
            dtypes=[torch.int32]),
        ),
    ),
    OpInfo(
        'ones_like',
        dtypes=_dispatch_dtypes((torch.uint8, torch.int8, torch.int32, torch.float16, torch.float32)),
        dtypesIfNPU=_dispatch_dtypes((torch.uint8, torch.int8, torch.int32, torch.float16, torch.float32)),
        sample_inputs_func=common_methods_invocations.sample_inputs_like_fns,
        supports_autograd=False,
        formats=(2, ),
        supports_out=False,
        skips=(
            DecorateInfo(unittest.skip("skipped!"), 'TestOps', 'test_variant_consistency_eager', 
            dtypes=[torch.float32]),
        )
    ),
    BinaryUfuncInfo(
        'pow',
        dtypes=_dispatch_dtypes((torch.float32, )),
        dtypesIfNPU=_dispatch_dtypes((torch.float16, torch.float32)),
        formats=(2, ),
        supports_inplace_autograd=False,
        skips=(
            DecorateInfo(unittest.skip("skipped!"), 'TestOps', 'test_correctness', 
            dtypes=[torch.float16, torch.float32]),
        ),
    ),
    OpInfo(
        'nn.functional.prelu',
        dtypes=_dispatch_dtypes((torch.float32, )),
        dtypesIfNPU=_dispatch_dtypes((torch.float16, torch.float32)),
        sample_inputs_func=common_methods_invocations.sample_inputs_prelu,
        supports_out=False,
        skips=(
            DecorateInfo(unittest.skip("skipped!"), 'TestOps', 'test_correctness', 
            dtypes=[torch.float16, torch.float32]),
            DecorateInfo(unittest.skip("skipped!"), 'TestOps', 'test_variant_consistency_eager', 
            dtypes=[torch.float32,]),
        ),
    ),
    OpInfo(
        'put',
        dtypes=_dispatch_dtypes((torch.float16, torch.float32)),
        dtypesIfNPU=_dispatch_dtypes((torch.float16, torch.float32)),
        sample_inputs_func=common_methods_invocations.sample_inputs_put,
        supports_out=False,
        skips=(
            DecorateInfo(unittest.skip("skipped!"), 'TestOps', 'test_correctness', 
            dtypes=[torch.float16, torch.float32]),
            DecorateInfo(unittest.skip("skipped!"), 'TestOps', 'test_variant_consistency_eager', 
            dtypes=[torch.float32,]),
        ),
    ),
    OpInfo(
        'qr',
        dtypes=_dispatch_dtypes((torch.float32, )),
        dtypesIfNPU=_dispatch_dtypes((torch.float32, )),
        sample_inputs_func=common_methods_invocations.sample_inputs_linalg_qr_geqrf,
        skips=(
            DecorateInfo(unittest.skip("skipped!"), 'TestOps', 'test_correctness', 
            dtypes=[torch.float32]),
        ),
    ),
    UnaryUfuncInfo(
        'reciprocal',
        dtypes=_dispatch_dtypes((torch.float32, )),
        dtypesIfNPU=_dispatch_dtypes((torch.float16, torch.float32, torch.int32, torch.int64)),
        formats=(2,),
    ),  
    OpInfo(
        'nn.functional.relu',
        dtypes=_dispatch_dtypes((torch.float32, )),
        dtypesIfNPU=_dispatch_dtypes((torch.float16, torch.float32)),
        sample_inputs_func=common_methods_invocations.sample_inputs_nn_activation_relu,
        formats=(2, ),
        supports_out=False,
    ),
    BinaryUfuncInfo(
        'remainder',
        dtypes=_dispatch_dtypes((torch.float32, )),
        dtypesIfNPU=_dispatch_dtypes((torch.float16, torch.float32)),
        skips=(
            DecorateInfo(unittest.skip("skipped!"), 'TestOps', 'test_correctness', 
            dtypes=[torch.float16, torch.float32]),
            DecorateInfo(unittest.skip("skipped!"), 'TestOps', 'test_variant_consistency_eager', 
            dtypes=[torch.float32]),
        ),
    ),  
    OpInfo(
        'renorm',
        dtypes=_dispatch_dtypes((torch.float16, torch.float32)),
        dtypesIfNPU=_dispatch_dtypes((torch.float16, torch.float32)),
        sample_inputs_func=common_methods_invocations.sample_inputs_renorm,
        skips=(
            DecorateInfo(unittest.skip("skipped!"), 'TestOps', 'test_correctness', 
            dtypes=[torch.float16, torch.float32]),
            DecorateInfo(unittest.skip("skipped!"), 'TestOps', 'test_variant_consistency_eager', 
            dtypes=[torch.float32]),
        ),
    ),
    OpInfo(
        'repeat_interleave',
        dtypes=_dispatch_dtypes((torch.int32, torch.float16, torch.float32)),
        dtypesIfNPU=_dispatch_dtypes((torch.int32, torch.float16, torch.float32)),
        sample_inputs_func=common_methods_invocations.sample_inputs_repeat_interleave,
        supports_out=False,
        skips=(
            DecorateInfo(unittest.skip("skipped!"), 'TestOps', 'test_correctness', 
            dtypes=[torch.int32, torch.float16, torch.float32]),
            DecorateInfo(unittest.skip("skipped!"), 'TestOps', 'test_variant_consistency_eager', 
            dtypes=[torch.float32]),
        ),
    ),
    OpInfo(
        'reshape',
        dtypes=_dispatch_dtypes((torch.bool, torch.int32, torch.float16, torch.float32)),
        dtypesIfNPU=_dispatch_dtypes((torch.bool, torch.int32, torch.float16, torch.float32)),
        sample_inputs_func=common_methods_invocations.sample_inputs_view_reshape,
        supports_out=False,
    ),
    # np
    OpInfo(
        'roll',
        dtypes=_dispatch_dtypes((torch.float16, torch.float32)),
        dtypesIfNPU=_dispatch_dtypes((torch.float16, torch.float32)),
        sample_inputs_func=common_methods_invocations.sample_inputs_roll,
        supports_out=False,
    ),
    UnaryUfuncInfo(
        'round',
        dtypes=_dispatch_dtypes((torch.float32, )),
        dtypesIfNPU=_dispatch_dtypes((torch.float16, torch.float32)),
    ),
    UnaryUfuncInfo(
        'nn.functional.celu',
        dtypes=_dispatch_dtypes((torch.float32, )),
        dtypesIfNPU=_dispatch_dtypes((torch.float16, torch.float32)),
        supports_forward_ad=True,
        supports_fwgrad_bwgrad=False,
        supports_autograd=True,
        assert_autodiffed=False,
        supports_gradgrad=True,
        supports_out=False,
        sample_kwargs=lambda device, dtype, input:
            ({'alpha': 0.8}, {'alpha': 0.8}),
        inplace_variant=lambda x, alpha=1.0:
            torch.nn.functional.celu(x, alpha, inplace=True),
    ),
    OpInfo(
        "nn.functional.binary_cross_entropy_with_logits",
        aten_name="binary_cross_entropy_with_logits",
        supports_autograd=True,
        supports_forward_ad=True,
        supports_fwgrad_bwgrad=True,
        supports_out=False,
        dtypes=_dispatch_dtypes((torch.float32, )),
        dtypesIfNPU=_dispatch_dtypes((torch.float32, )),
        sample_inputs_func=sample_inputs_binary_cross_entropy_with_logits,
        formats=(2,),
    ),
    UnaryUfuncInfo(
        'nn.functional.rrelu',
        dtypes=_dispatch_dtypes((torch.float32, )),
        dtypesIfNPU=_dispatch_dtypes((torch.float16, torch.float32)),
        supports_out=False,
    ),
    BinaryUfuncInfo(
        'rsub',
        dtypes=_dispatch_dtypes((torch.int32, torch.float16, torch.float32)),
        dtypesIfNPU=_dispatch_dtypes((torch.int32, torch.float16, torch.float32)),
        sample_inputs_func=partial(common_methods_invocations.sample_inputs_add_sub),
        formats=(0, 2, 3, 29),
        supports_inplace_autograd=False,
        supports_out=False,
    ), 
    OpInfo(
        'scatter',
        dtypes=_dispatch_dtypes((torch.float32, )),
        dtypesIfNPU=_dispatch_dtypes((torch.float16, torch.float32)),
        sample_inputs_func=common_methods_invocations.sample_inputs_scatter,
        skips=(
            DecorateInfo(unittest.skip("skipped!"), 'TestOps', 'test_correctness', 
            dtypes=[torch.float16, torch.float32]),
            DecorateInfo(unittest.skip("skipped!"), 'TestOps', 'test_variant_consistency_eager', 
            dtypes=[torch.float32]),
        ),
    ),    
    OpInfo(
        'scatter_add',
        dtypes=_dispatch_dtypes((torch.float32, )),
        dtypesIfNPU=_dispatch_dtypes((torch.float16, torch.float32)),
        sample_inputs_func=common_methods_invocations.sample_inputs_scatter_add,
        skips=(
            DecorateInfo(unittest.skip("skipped!"), 'TestOps', 'test_correctness', 
            dtypes=[torch.float16, torch.float32]),
            DecorateInfo(unittest.skip("skipped!"), 'TestOps', 'test_variant_consistency_eager', 
            dtypes=[torch.float32]),
            DecorateInfo(unittest.skip("skipped!"), 'TestOps', 'test_out', 
            dtypes=[torch.float32]),
        ),
    ),  
    OpInfo(
        'searchsorted',
        dtypes=_dispatch_dtypes((torch.int32, torch.int64, torch.float32, )),
        dtypesIfNPU=_dispatch_dtypes((torch.int32, torch.int64, torch.float32)),
        sample_inputs_func=common_methods_invocations.sample_inputs_searchsorted,
        supports_autograd=False,
        skips=(
            DecorateInfo(unittest.skip("skipped!"), 'TestOps', 'test_correctness', 
            dtypes=[torch.int32, torch.int64, torch.float32]),
            DecorateInfo(unittest.skip("skipped!"), 'TestOps', 'test_variant_consistency_eager', 
            dtypes=[torch.float32, ]),
        ),
    ),  
    UnaryUfuncInfo(
        'sigmoid',
        aliases=('special.expit', 'nn.functional.sigmoid'),
        dtypes=_dispatch_dtypes((torch.float32, )),
        dtypesIfNPU=_dispatch_dtypes((torch.float16, torch.float32)),
        skipSample={
            'test_correctness' : (0, 2, ),
        },
    ), 
    UnaryUfuncInfo(
        'sign',
        dtypes=_dispatch_dtypes((torch.float32, )),
        dtypesIfNPU=_dispatch_dtypes((torch.float16, torch.float32)),
    ),
    UnaryUfuncInfo(
        'nn.functional.selu',
        dtypes=_dispatch_dtypes((torch.float32, )),
        dtypesIfNPU=_dispatch_dtypes((torch.float16, torch.float32)),
        supports_forward_ad=True,  # depends on 'elu'
        supports_fwgrad_bwgrad=False,  # Needs: elu_backward
        supports_autograd=True,
        assert_autodiffed=False,
        supports_gradgrad=True,
        supports_out=False,
        inplace_variant=lambda x: torch.nn.functional.selu(x, inplace=True),
    ),
    UnaryUfuncInfo(
        'nn.functional.silu',
        dtypes=_dispatch_dtypes((torch.float32, )),
        dtypesIfNPU=_dispatch_dtypes((torch.float16, torch.float32)),
        supports_out=False,
        skips=(
            DecorateInfo(unittest.skip("skipped!"), 'TestOps', 'test_correctness', 
            dtypes=[torch.float16, torch.float32]),
        ),
    ), 
    UnaryUfuncInfo(
        'sin',
        dtypes=_dispatch_dtypes((torch.float32, )),
        dtypesIfNPU=_dispatch_dtypes((torch.float16, torch.float32)),
    ), 
    UnaryUfuncInfo(
        'sinh',
        dtypes=_dispatch_dtypes((torch.float32, )),
        dtypesIfNPU=_dispatch_dtypes((torch.float16, torch.float32)),
    ), 
    OpInfo(
        'softmax',
        aliases=('special.softmax', 'nn.functional.softmax',),
        dtypes=_dispatch_dtypes((torch.float32, )),
        dtypesIfNPU=_dispatch_dtypes((torch.float16, torch.float32)),
        sample_inputs_func=common_methods_invocations.sample_inputs_softmax_variant,
        formats=(0, 3, 29),
        supports_out=False,
        skips=(
            DecorateInfo(unittest.skip("skipped!"), 'TestOps', 'test_correctness', 
            dtypes=[torch.float16, torch.float32]),
        ),
    ), 
    OpInfo(
        'nn.functional.softshrink',
        dtypes=_dispatch_dtypes((torch.float32, )),
        dtypesIfNPU=_dispatch_dtypes((torch.float16, torch.float32)),
        sample_inputs_func=common_methods_invocations.sample_inputs_softshrink,
        supports_out=False,
    ), 
    OpInfo(
        'sort',
        dtypes=_dispatch_dtypes((torch.float32, )),
        dtypesIfNPU=_dispatch_dtypes((torch.float16, torch.float32)),
        sample_inputs_func=common_methods_invocations.sample_inputs_sort,
        skips=(
            DecorateInfo(unittest.skip("skipped!"), 'TestOps', 'test_correctness', 
            dtypes=[torch.float16, torch.float32]),
            DecorateInfo(unittest.skip("skipped!"), 'TestOps', 'test_variant_consistency_eager', 
            dtypes=[torch.float32]),
        ),
    ), 
    UnaryUfuncInfo(
        'sqrt',
        dtypes=_dispatch_dtypes((torch.float32, )),
        dtypesIfNPU=_dispatch_dtypes((torch.float16, torch.float32)),
        formats=(0, 3),
        skipSample={
            'test_correctness' : (0, 2, ),
        },
    ), 
    OpInfo(
        'stack',
        dtypes=_dispatch_dtypes((torch.float32, )),
        dtypesIfNPU=_dispatch_dtypes((torch.float16, torch.float32)),
        sample_inputs_func=common_methods_invocations.sample_inputs_stack,
    ), 
    OpInfo(
        'std_mean',
        dtypes=_dispatch_dtypes((torch.float32, )),
        dtypesIfNPU=_dispatch_dtypes((torch.float16, torch.float32)),
        sample_inputs_func=partial(common_methods_invocations.sample_inputs_reduction, 
        supports_multiple_dims=False),
        supports_out=False,
        skips=(
            DecorateInfo(unittest.skip("skipped!"), 'TestOps', 'test_correctness', 
            dtypes=[torch.float16, torch.float32]),
        ),
    ), 
    BinaryUfuncInfo(
        'sub',
        aliases=('subtract',),
        dtypes=_dispatch_dtypes((torch.int32, torch.float32)),
        dtypesIfNPU=_dispatch_dtypes((torch.int32, torch.float16, torch.float32)),
        sample_inputs_func=partial(common_methods_invocations.sample_inputs_add_sub, 
        alpha=2, other_scalar=True),
        formats=(2, ),
        supports_inplace_autograd=False,
        skipSample={
            'test_variant_consistency_eager' : (8, 18),
        },
    ),  
    # sum reductioninfo
    OpInfo(
        'symeig',
        dtypes=_dispatch_dtypes((torch.float32, )),
        dtypesIfNPU=_dispatch_dtypes((torch.float32, )),
        sample_inputs_func=common_methods_invocations.sample_inputs_symeig,
        skips=(
            DecorateInfo(unittest.skip("skipped!"), 'TestOps', 'test_correctness', 
            dtypes=[torch.float32]),
            DecorateInfo(unittest.skip("skipped!"), 'TestOps', 'test_variant_consistency_eager', 
            dtypes=[torch.float32]),
        ),
    ),    
    OpInfo(
        'take',
        dtypes=_dispatch_dtypes((torch.int8, torch.int16, torch.int64, torch.float32, )),
        dtypesIfNPU=_dispatch_dtypes((torch.int8, torch.int16, torch.int64, torch.float16, 
        torch.float32, )),
        sample_inputs_func=common_methods_invocations.sample_inputs_take,
        skips=(
            DecorateInfo(unittest.skip("skipped!"), 'TestOps', 'test_correctness', 
            dtypes=[torch.int8, torch.int16, torch.int64, torch.float16, torch.float32]),
            DecorateInfo(unittest.skip("skipped!"), 'TestOps', 'test_variant_consistency_eager', 
            dtypes=[torch.float32]),
        ),
    ),   
    UnaryUfuncInfo(
        'tan',
        dtypes=_dispatch_dtypes((torch.float32, )),
        dtypesIfNPU=_dispatch_dtypes((torch.float32, )),
    ), 
    UnaryUfuncInfo(
        'tanh',
        dtypes=_dispatch_dtypes((torch.float32, )),
        dtypesIfNPU=_dispatch_dtypes((torch.float32, )),
        skipSample={
            'test_correctness' : (0, 2, ),
        },
    ), 
    OpInfo(
        'threshold',
        dtypes=_dispatch_dtypes((torch.int32, torch.float32)),
        dtypesIfNPU=_dispatch_dtypes((torch.int32, torch.float32)),
        sample_inputs_func=common_methods_invocations.sample_inputs_threshold,
    ),   
    OpInfo(
        'topk',
        dtypes=_dispatch_dtypes((torch.float32, )),
        dtypesIfNPU=_dispatch_dtypes((torch.float16, torch.float32)),
        sample_inputs_func=common_methods_invocations.sample_inputs_topk,
        formats=(0, 3, 4, 29),
        skips=(
            DecorateInfo(unittest.skip("skipped!"), 'TestOps', 'test_correctness', 
            dtypes=[torch.float16, torch.float32]),
            DecorateInfo(unittest.skip("skipped!"), 'TestOps', 'test_variant_consistency_eager', 
            dtypes=[torch.float32]),
        ),
    ),    
    OpInfo(
        'transpose',
        aliases=('swapdims', 'swapaxes'),
        dtypes=_dispatch_dtypes((torch.float16, torch.float32)),
        dtypesIfNPU=_dispatch_dtypes((torch.float16, torch.float32)),
        sample_inputs_func=common_methods_invocations.sample_inputs_transpose_swapdims,
        supports_out=False,
    ), 
    OpInfo(
        'triangular_solve',
        dtypes=_dispatch_dtypes((torch.float32, )),
        dtypesIfNPU=_dispatch_dtypes((torch.float32, )),
        sample_inputs_func=common_methods_invocations.sample_inputs_legacy_solve,
        skips=(
            DecorateInfo(unittest.skip("skipped!"), 'TestOps', 'test_correctness', 
            dtypes=[torch.float32]),
            DecorateInfo(unittest.skip("skipped!"), 'TestOps', 'test_variant_consistency_eager', 
            dtypes=[torch.float32]),
        ),
    ),  
    OpInfo(
        'tril',
        dtypes=_dispatch_dtypes((torch.float16, torch.float32, )),
        dtypesIfNPU=_dispatch_dtypes((torch.float16, torch.float32, )),
        sample_inputs_func=common_methods_invocations.sample_inputs_tril_triu,
        skips=(
            DecorateInfo(unittest.skip("skipped!"), 'TestOps', 'test_correctness', 
            dtypes=[torch.float16, torch.float32]),
            DecorateInfo(unittest.skip("skipped!"), 'TestOps', 'test_variant_consistency_eager', 
            dtypes=[torch.float32]),
        ),
    ),  
    OpInfo(
        'triu',
        dtypes=_dispatch_dtypes((torch.float16, torch.float32, )),
        dtypesIfNPU=_dispatch_dtypes((torch.float16, torch.float32, )),
        sample_inputs_func=common_methods_invocations.sample_inputs_tril_triu,
    ),  
    BinaryUfuncInfo(
        'true_divide',
        dtypes=_dispatch_dtypes((torch.bool, torch.int32, torch.float16, torch.float32)),
        dtypesIfNPU=_dispatch_dtypes((torch.bool, torch.int32, torch.float16, torch.float32)),
        skips=(
            DecorateInfo(unittest.skip("skipped!"), 'TestOps', 'test_correctness', 
            dtypes=[torch.bool, torch.int32, torch.float16, torch.float32]),
            DecorateInfo(unittest.skip("skipped!"), 'TestOps', 'test_variant_consistency_eager', 
            dtypes=[torch.float32]),
        ),
    ),  
    UnaryUfuncInfo(
        'trunc',
        aliases=('fix', ),
        dtypes=_dispatch_dtypes((torch.float32, )),
        dtypesIfNPU=_dispatch_dtypes((torch.float16, torch.float32)),
    ), 
    OpInfo(
        'where',
        dtypes=_dispatch_dtypes((torch.float32, )),
        dtypesIfNPU=_dispatch_dtypes((torch.float16, torch.float32)),
        sample_inputs_func=common_methods_invocations.sample_inputs_where,
        skips=(
            DecorateInfo(unittest.skip("skipped!"), 'TestOps', 'test_correctness', 
            dtypes=[torch.float16, torch.float32]),
            DecorateInfo(unittest.skip("skipped!"), 'TestOps', 'test_variant_consistency_eager', 
            dtypes=[torch.float32]),
            DecorateInfo(unittest.skip("skipped!"), 'TestOps', 'test_out', dtypes=[torch.float32]),
        ),
    ), 
    BinaryUfuncInfo(
        'xlogy',
        aliases=('special.xlogy', ),
        dtypes=_dispatch_dtypes((torch.float32, )),
        dtypesIfNPU=_dispatch_dtypes((torch.float32, )),
        skips=(
            DecorateInfo(unittest.skip("skipped!"), 'TestOps', 'test_correctness', 
            dtypes=[torch.float32]),
            DecorateInfo(unittest.skip("skipped!"), 'TestOps', 'test_variant_consistency_eager', 
            dtypes=[torch.float32]),
        ),
    ),  
    OpInfo(
        'zeros_like',
        dtypes=_dispatch_dtypes((torch.float32, )),
        dtypesIfNPU=_dispatch_dtypes((torch.float16, torch.float32)),
        sample_inputs_func=common_methods_invocations.sample_inputs_like_fns,
        supports_autograd=False,
        formats=(0, 3, 29),
        supports_out=False,
        skips=(
            DecorateInfo(unittest.skip("skipped!"), 'TestOps', 'test_variant_consistency_eager', 
            dtypes=[torch.float32]),
        ),
    ), 
]
