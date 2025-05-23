#!/usr/bin/env python
# -*- coding: utf-8 -*-
#  Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.


from torch._inductor import sizevars
from torch._inductor.codegen.simd import SIMDKernel
from torch._inductor.codegen.triton import TritonKernel
from torch._inductor.codegen.triton import TritonScheduling
from torch._inductor.ir import Reduction, LoopBody
from torch_npu._inductor.codegen._sizevars import simplify
from torch_npu._inductor.codegen.ir import (num_splits, loopbody__call__, transform_dims_in_indexing,
                                            substituted_dims_in_indexing)
from torch_npu._inductor.codegen.schduling import create_tiling
from torch_npu._inductor.codegen.triton import group_fn, select_index_dtype
from torch_npu._inductor.codegen.triton import is_compatible

from ..config import log as npulog


Reduction.num_splits = num_splits
setattr(LoopBody, 'transform_dims_in_indexing', transform_dims_in_indexing)
setattr(LoopBody, 'substituted_dims_in_indexing', substituted_dims_in_indexing)

LoopBody.__call__ = loopbody__call__
# need to enable this to speedup attn_cp_test
# triton scheduling
TritonScheduling.group_fn = group_fn
TritonScheduling.select_index_dtype = select_index_dtype
TritonScheduling.create_tiling = create_tiling
# triton kernel
setattr(SIMDKernel, 'is_compatible', is_compatible)

# util
sizevars.SizeVarAllocator.simplify = simplify
