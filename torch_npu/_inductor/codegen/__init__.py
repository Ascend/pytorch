#!/usr/bin/env python
# -*- coding: utf-8 -*-
#  Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.


from torch._inductor import sizevars
from torch._inductor.codegen.simd import SIMDKernel
from torch._inductor.codegen.triton import TritonKernel
from torch._inductor.codegen.triton import TritonScheduling, Scheduler
from torch._inductor.ir import Reduction, LoopBody
from torch._inductor.loop_body import CaptureIndexing
from torch_npu._inductor.codegen._sizevars import simplify
from torch_npu._inductor.codegen.ir import (num_splits, loopbody__call__, transform_dims_in_indexing,
                                            substituted_dims_in_indexing, generate_indirect_replacements,
                                            substitube_indirect_index,
                                            loop_body_block_index_select, simplify_indexing_index_select,
                                            loop_body_block_gather_template,
                                            simplify_indexing_gather_template, loop_body_block_indexput_template,
                                            simplify_indexing_indexput_template,
                                            loop_body_block_scatter_template,
                                            simplify_indexing_scatter_template,
                                            simplify_indexing_cat_store, loop_body_block_cat_store)
from torch_npu._inductor.codegen.scheduling import create_tiling, are_long_distant_nodes
from torch_npu._inductor.codegen.triton import group_fn, select_index_dtype
from torch_npu._inductor.codegen.triton import is_compatible
from torch_npu.npu._backends import get_soc_version

from ..config import log as npulog, Ascend910_9391


Reduction.num_splits = num_splits
setattr(LoopBody, 'transform_dims_in_indexing', transform_dims_in_indexing)
setattr(LoopBody, 'substituted_dims_in_indexing', substituted_dims_in_indexing)
setattr(LoopBody, 'generate_indirect_replacements', generate_indirect_replacements)
setattr(LoopBody, 'substitube_indirect_index', substitube_indirect_index)

LoopBody.__call__ = loopbody__call__
# need to enable this to speedup attn_cp_test
# triton scheduling
TritonScheduling.group_fn = group_fn
TritonScheduling.select_index_dtype = select_index_dtype
TritonScheduling.create_tiling = create_tiling
if get_soc_version() >= Ascend910_9391:
    Scheduler.are_long_distant_nodes = are_long_distant_nodes
# triton kernel
setattr(SIMDKernel, 'is_compatible', is_compatible)

# util
sizevars.SizeVarAllocator.simplify = simplify

setattr(CaptureIndexing, 'index_select', loop_body_block_index_select)
setattr(sizevars.SimplifyIndexing, 'index_select', simplify_indexing_index_select)
setattr(CaptureIndexing, 'gather_template', loop_body_block_gather_template)
setattr(sizevars.SimplifyIndexing, 'gather_template', simplify_indexing_gather_template)
setattr(CaptureIndexing, 'indexput_template', loop_body_block_indexput_template)
setattr(sizevars.SimplifyIndexing, 'indexput_template', simplify_indexing_indexput_template)
setattr(CaptureIndexing, 'scatter_template', loop_body_block_scatter_template)
setattr(sizevars.SimplifyIndexing, 'scatter_template', simplify_indexing_scatter_template)
setattr(CaptureIndexing, 'cat_store', loop_body_block_cat_store)
setattr(sizevars.SimplifyIndexing, 'cat_store', simplify_indexing_cat_store)