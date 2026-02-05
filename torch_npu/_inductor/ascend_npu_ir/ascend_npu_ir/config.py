import os
import sys
import time

import torch
from os.path import abspath, dirname
from typing import (
    Any, Callable, Dict, Optional, List,
    Set, Type, TYPE_CHECKING, Union,
    )

from torch._inductor import config, inductor_prims

aten = torch.ops.aten
tr_c10d = torch.ops.tr_c10d
prims = torch.ops.prims
npu = torch.ops.npu

always_compile = False
enable_graph_trace = True
acc_comp_mode = True
disable_any_pbr = True
autotune_fx_fallback = False
cache_named_op = False

traced_graph_cache = os.environ.get("ANIR_TRACED_GRAPH_CACHE", None)
torch_mlir_dump_path = os.environ.get("ANIR_TORCH_MLIR_DUMP", None)

online_acc_comp = os.environ.get("ANIR_ONLINE_ACC_COMP", "0") == "1"
runtime_error_dump = os.environ.get("ANIR_RUNTIME_ERROR_DUMP", "0") == "1"
fallback_dump = os.environ.get("ANIR_FALLBACK_DUMP", "0") == "1"
acc_comp_tol = {
    torch.float32: {'rtol': 1.3e-6, 'atol': 1e-5},
    torch.float16: {'rtol': 1e-3, 'atol': 1e-5},
    torch.bfloat16: {'rtol': 1.6e-2, 'atol': 1e-5},
    "default": {'rtol': 1.3e-6, 'atol': 1e-5},
}
tune_error_time = 99999.9

autotune = os.environ.get("AUTOTUNE", "1") == "1"
multiprocess_compile = autotune and os.environ.get("DISABLE_MP_COMPILE", "0") == "0"

mode = os.getenv('ANIR_MODE', 'O1')
if mode not in ["O0", "O1"]:
    raise ValueError(f"Invalid MODE value: {mode}. Allowed values are 'O0' and 'O1'.")

'''
Add extra command for bisheng compile. Some useful commands:
"-mlir-print-ir-before-all": print the entire IR before each pass.
"-mlir-print-ir-after-all": print the entire IR after each pass.
Add extra commands before model excute, code like:

from torch_npu._inductor.ascend_npu_ir.ascend_npu_ir import config as anir_config
anir_config.extra_command += [
      "-mlir-print-ir-before-all",
      "-mlir-print-ir-after-all"
]
'''
extra_command = []

debug = os.environ.get("ANIR_DEBUG", "0") == "1"
fallback_warning = os.environ.get("ANIR_FALLBACK_WARNING", "0") == "1"

'''
set force_fallback_kernel_names while your faces runtime errors, and want to skip the kernel by kernel name.
set force_fallback_kernel_paths while your faces runtime errors, and want to skip the kernel by kernel cache paths.
examples:
force_fallback_kernel_names = {'mlir_fused_add_1', 'mlir_fused_add_2'}
force_fallback_kernel_paths = {'/path/to/mlir_fused_add_1_0_True_True.o', '/path/to/mlir_fused_add_2_0_True_True.o'}
'''
force_fallback_kernel_names = {}
force_fallback_kernel_paths = {}


if debug:
    debug_dir = os.environ.get("ANIR_DEBUG_DIR", f"{os.environ['PWD']}/anir_debug")
    os.makedirs(debug_dir, exist_ok=True)

fx_graph_dump_path: str = None

# dump fx subgraph for debugging
fx_subgraph_dump_path: str = os.environ.get("FX_SUBGRAPH_DUMP_PATH", None)
"""
compile_mode introductions:
"default" refers to the mode of fully compiling with MLIR. Currently, it is not fully supported, but it will be set as the default once the capability matures.
"complete_fallback" refers to completely falling back to the eager execution mode of the FX graph, without performing any MLIR compilation. It is primarily used for debugging.
"auto_fallback" refers to automatically falling back to the fx_graph_backend when compilation fails. 
auto_fallback mechanism is designed to provide a fallback strategy when the primary compilation process encounters an issue. It works in conjunction with the fx_graph_backend configuration, allowing for the fallback approach:
Fallback to fx_graph_backend: If the first fallback attempt fails, the system falls back to the fx_graph_backend.
If you need further clarification or have other questions, please let me know!
"""
compile_mode: str = 'auto_fallback'
def _get_compile_mode():
    m = sys.modules[__name__]
    if isinstance(m.compile_mode, str):
        mode = m.compile_mode
        if mode not in ['default', 'complete_fallback', 'auto_fallback']:
            raise ValueError(f"Invalid mode {mode=}")
    else:
        raise ValueError(f"Please use the *str* type to set the compile mode, current type is {type(compile_mode)=}")

    return mode

block_dim = 48

"""
support {"off", "include", "exclude"}, to 
"off": No fallback at all.
"include": At compile-time, Aten IR included in FALLBACK_LIST will fall back to aten.
"exclude": At compile-time, Aten IR excluded from GENERATE_LIST will fall back to aten.
"""
fallback_to_aten_mode: str = "exclude"

REDUCTION_OPS = [
    aten.sum, 
    prims.sum,
    aten.prod,
    aten.any,
    aten.max,
    aten.min,
    prims.xor_sum,
    aten.amax,
    aten.amin,
    aten.argmax,
    aten.argmin,
    aten.mean,
    aten.var, 
    prims.var,
    aten.var_mean,
]

# fall back to aten exclude GENERATE_LIST, all aten IR except 
GENERATE_LIST = [
    aten.mul,
    aten.add,
    aten.sub,
    aten.div,
    aten.exp,
    aten.pow,
    aten.rsqrt,
    aten.neg,
    aten.lt,
    aten.gt,
    aten.ge,
    aten.le,
    aten.eq,
    aten.sigmoid,
    prims.convert_element_type,
    torch.ops.npu.npu_dtype_cast,
    torch.ops.npu.npu_dtype_cast_backward,
    torch.ops.npu._npu_dtype_cast,
    torch.ops.npu._npu_dtype_cast_backward,

    aten.squeeze,
    aten.unsqueeze,
    aten.expand,
    aten.repeat,
    aten.clone,
    aten.reshape,
    aten.sin,
	aten.cos,
    aten.var_mean,
    aten.sum, 
    aten.mean,
    aten.full,
    aten.slice,
    aten.split,
    aten.split_with_sizes,
    aten.reciprocal,
    aten.select,
    # prims.iota,
    aten.relu,
    aten.copy_,
    aten.where,
    aten.log,
    aten.scalar_tensor,
    aten.permute,
    aten.cat,
    aten.constant_pad_nd,
    aten.amax,
    aten.slice_scatter,
    aten.sqrt,
    aten.copy,
    aten.clamp_min,
    aten.clamp_max,
    aten.bitwise_not,
    aten.tanh,
    aten.unbind,
    aten.lift_fresh_copy,
]


FALLBACK_LIST = [
    aten.mm,
    aten.bmm,
    aten.addmm,
    aten.convolution,
    aten.convolution_backward,
    aten._adaptive_avg_pool2d,
    aten.max_pool2d_with_indices,
    aten.max_pool2d_with_indices_backward,
    aten.avg_pool2d,
    aten.avg_pool2d_backward,
    inductor_prims.lookup_seed,
    inductor_prims.random,
    prims.device_put,
    aten.upsample_nearest2d,
    aten.upsample_nearest2d_backward,
    aten.embedding,
    # aten.cat,
    # aten.permute,
    aten.constant_pad_nd,
    aten.abs,
    aten.max,
    aten.amax,
    aten.amin,
    aten.slice_scatter,
    aten.select_scatter,
    aten.gather,
    aten.scatter,
    npu._npu_dropout,
    aten.empty,
    aten.index,
    aten.copy_
]


decomps_to_exclude_npu = [
    aten.gelu.default,
    aten.gelu_backward.default,
    aten.embedding,
    aten.embedding_backward,
    aten.embedding_dense_backward,
    aten.upsample_nearest2d,
    aten.upsample_nearest2d_backward,
    aten.upsample_nearest1d,
    aten.upsample_nearest1d_backward,
    aten.upsample_nearest3d,
    aten.upsample_nearest3d_backward,
    aten.upsample_bilinear2d,
    aten.upsample_bilinear2d_backward,
    aten.nll_loss2d_forward,
    aten.nll_loss2d_backward,
    aten.nll_loss_backward,
    aten.nll_loss_forward,
    aten.triu,
    aten.convolution_backward,
    aten._softmax_backward_data.default,
    aten.max_pool2d_with_indices,
    aten.max_pool2d_with_indices_backward,
    aten.slice.Tensor,
    aten.reflection_pad2d_backward,
    aten.reflection_pad2d,
    aten.grid_sampler_2d,
    aten.grid_sampler_2d_backward,
]