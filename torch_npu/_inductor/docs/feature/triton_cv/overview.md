# TORCHINDUCTOR_MAX_AUTOTUNE_GEMM_BACKENDS （同社区）

## 功能描述

确认max autotune可尝试的后端有哪些，该环境变量与社区max autotune设置可尝试后端的环境变量一致。

若想尝试Catlass的后端，请在该环境变量中配置上"CATLASS"。

默认配置为TORCHINDUCTOR_MAX_AUTOTUNE_GEMM_BACKENDS="ATEN,TRITON,CPP", 此默认值为社区的默认配置。

## 配置示例

尝试在max autotune中使用ATEN和CATLASS的后端。

```shell
export TORCHINDUCTOR_MAX_AUTOTUNE_GEMM_BACKENDS="TRITON"
```

## 使用约束

无

## 支持的型号

- <term>Atlas A5 系列产品</term>

# TORCHINDUCTOR_MAX_AUTOTUNE （同社区）

## 功能描述

该环境变量用于控制是否开启max autotune功能，与社区一致。

"0"为关闭，"1"为开启

默认配置为TORCHINDUCTOR_MAX_AUTOTUNE="0"

## 配置示例

开启max autotune功能

```shell
export TORCHINDUCTOR_MAX_AUTOTUNE=1
```

关闭max autotune功能

```shell
export TORCHINDUCTOR_MAX_AUTOTUNE=0
```

## 使用约束

无

## 支持的设备

- <term>Atlas A5 系列产品</term>

# TORCHINDUCTOR_PROFILE_WITH_DO_BENCH_USING_PROFILING （同社区）

## 功能描述

该环境变量与inductor整体的profiling环境变量保持一致，用于管理autotune过程中是否使用profiling。

"0"为不使用profiling，"1"为使用profiling。

默认配置为TORCHINDUCTOR_PROFILE_WITH_DO_BENCH_USING_PROFILING="0"。

## 配置示例

开启autotune过程中的profiling

```shell
export TORCHINDUCTOR_PROFILE_WITH_DO_BENCH_USING_PROFILING="1"
```

关闭autotune过程中的profiling

```shell
export TORCHINDUCTOR_PROFILE_WITH_DO_BENCH_USING_PROFILING="0"
```

## 使用约束

无

## 支持的型号

- <term>Atlas A5 系列产品</term>

# Triton cv特性介绍

## 特性简介

在深度学习领域，矩阵乘法MM类算子是计算量最大的核心操作。传统的 PyTorch eager 模式下，矩阵乘法和随后pointwise（逐点的）算子（如激活函数 ReLU（修正线性单元）、SiLU（Sigmoid Linear Unit）、Bias Add（偏置加法） 等）通常作为独立的Kernel依次执行。这种方式会导致频繁的Kernel启动开销和大量的全局内存数据传输（I/O 瓶颈），严重影响性能。

另外，社区PyTorch Inductor通过Triton后端可以支持cv融合，因此torch_npu的inductor中也要对标社区功能，为cv融合提供相应的triton后端。

## Triton cv特性使用方法

### Triton cv使用示例

```python
import torch
import torch.nn.functional as F
from torch._dynamo.testing import rand_strided
from testutils import TestUtils
from torch.testing._internal.common_utils import run_tests
import torch_npu

MM_SHAPES = [
(200, 256, 256), ]

_MAX_AUTOTUNE = os.environ.get("TORCHINDUCTOR_MAX_AUTOTUNE", "0") == "1"
MODE_TAG = "maxautotune" if _MAX_AUTOTUNE else "noautotune"

def forward_mm(self, a, b):
return torch.mm(a, b)

#@unittest.skip("addmm triton template not yet supported in max_autotune")
def test_mm(self):
    for M, N, K in MM_SHAPES:
        with self.subTest(M=M, N=N, K=K):
            a = _t((M, K), stride=(K, 1))
            b = _t((K, N), stride=(1, K))

            eager_result = self.forward_mm(a, b)
            self._run_both_modes(
                self.forward_mm, (a, b),
                profile_name=f"cv_mm_{M}x{N}x{K}",
                eager_result=eager_result,
            )
```

### Triton cv应用效果

#### 编译进行中

在编译执行的过程中，会看到Triton cv在进行autotune过程的日志，以上述的示例为例，会有类似如下的日志输出

```shell
AUTOTUNE mm(200x256, 256x256)
strides: [256, 1], [1, 256]
dtypes: torch.float16, torch.float16
triton_npu_persistent_mm_164 0.0034 ms 100.0% ACC_TYPE='tl.float32', ALLOW_TF32='False', BLOCK_K=128, BLOCK_M=128, BLOCK_N=64, EVEN_K=True, GROUP_M=2, NUM_BLOCKS=8, NUM_BLOCKS_M=2, NUM_BLOCKS_N=4, NUM_SMS=8, NUM_TILES_PER_PROGRAM=1, WIDTH=8, num_stages=2, num_warps=4
triton_npu_triton_mm_14 0.0034 ms 99.8% ACC_TYPE='tl.float32', ALLOW_TF32='False', BLOCK_K=128, BLOCK_M=128, BLOCK_N=64, EVEN_K=True, GROUP_M=1, num_stages=2, num_warps=4
triton_npu_persistent_mm_177 0.0034 ms 99.7% ACC_TYPE='tl.float32', ALLOW_TF32='False', BLOCK_K=128, BLOCK_M=128, BLOCK_N=64, EVEN_K=True, GROUP_M=2, NUM_BLOCKS=8, NUM_BLOCKS_M=2, NUM_BLOCKS_N=4, NUM_SMS=8, NUM_TILES_PER_PROGRAM=1, WIDTH=8, num_stages=2, num_warps=8
triton_npu_persistent_mm_167 0.0034 ms 99.6% ACC_TYPE='tl.float32', ALLOW_TF32='False', BLOCK_K=128, BLOCK_M=128, BLOCK_N=64, EVEN_K=True, GROUP_M=2, NUM_BLOCKS=8, NUM_BLOCKS_M=2, NUM_BLOCKS_N=4, NUM_SMS=8, NUM_TILES_PER_PROGRAM=1, WIDTH=8, num_stages=3, num_warps=8
triton_npu_persistent_mm_176 0.0034 ms 99.6% ACC_TYPE='tl.float32', ALLOW_TF32='False', BLOCK_K=128, BLOCK_M=128, BLOCK_N=64, EVEN_K=True, GROUP_M=2, NUM_BLOCKS=8, NUM_BLOCKS_M=2, NUM_BLOCKS_N=4, NUM_SMS=8, NUM_TILES_PER_PROGRAM=1, WIDTH=8, num_stages=2, num_warps=4
triton_npu_triton_mm_22 0.0035 ms 99.5% ACC_TYPE='tl.float32', ALLOW_TF32='False', BLOCK_K=128, BLOCK_M=64, BLOCK_N=128, EVEN_K=True, GROUP_M=4, num_stages=3, num_warps=4
triton_npu_persistent_mm_170 0.0035 ms 99.5% ACC_TYPE='tl.float32', ALLOW_TF32='False', BLOCK_K=128, BLOCK_M=128, BLOCK_N=64, EVEN_K=True, GROUP_M=2, NUM_BLOCKS=8, NUM_BLOCKS_M=2, NUM_BLOCKS_N=4, NUM_SMS=8, NUM_TILES_PER_PROGRAM=1, WIDTH=8, num_stages=2, num_warps=4
triton_npu_persistent_mm_168 0.0035 ms 99.2% ACC_TYPE='tl.float32', ALLOW_TF32='False', BLOCK_K=128, BLOCK_M=128, BLOCK_N=64, EVEN_K=True, GROUP_M=2, NUM_BLOCKS=8, NUM_BLOCKS_M=2, NUM_BLOCKS_N=4, NUM_SMS=8, NUM_TILES_PER_PROGRAM=1, WIDTH=8, num_stages=4, num_warps=4
triton_npu_persistent_mm_162 0.0035 ms 99.2% ACC_TYPE='tl.float32', ALLOW_TF32='False', BLOCK_K=128, BLOCK_M=128, BLOCK_N=64, EVEN_K=True, GROUP_M=2, NUM_BLOCKS=8, NUM_BLOCKS_M=2, NUM_BLOCKS_N=4, NUM_SMS=8, NUM_TILES_PER_PROGRAM=1, WIDTH=8, num_stages=4, num_warps=4
triton_npu_persistent_mm_179 0.0035 ms 99.1% ACC_TYPE='tl.float32', ALLOW_TF32='False', BLOCK_K=128, BLOCK_M=128, BLOCK_N=64, EVEN_K=True, GROUP_M=2, NUM_BLOCKS=8, NUM_BLOCKS_M=2, NUM_BLOCKS_N=4, NUM_SMS=8, NUM_TILES_PER_PROGRAM=1, WIDTH=8, num_stages=3, num_warps=8
SingleProcess AUTOTUNE benchmarking takes 48.6676 seconds and 349.0871 seconds precompiling for 462 choices
Wxxx.708000 2317919 site-packages/torch/_inductor/debug.py:518] [0/0] model__0_inference_0 debug trace: xxx
[xxx] [WARNING] [2317919] profiler.py: Incorrect schedule: Stop profiler while current state is RECORD which may result in incomplete parsed data.
[xxx] [INFO] [2331382] profiler.py: Start parsing profiling data in sync mode at: xxx
[xxx] [INFO] [2331391] profiler.py: CANN profiling data parsed in a total time of 0:00:02.020406
[xxx] [INFO] [2331382] profiler.py: All profiling data parsed in a total time of 0:00:03.191598
```

如上所示的示例中，表明了inductor正在对 (512, 256) x (256, 1024)的matmul操作进行autotune的过程

#### 运行结果

在autotune的过程中，被选为最优的算子时，我们在torch_compile_debug的output_code.py中，即可看到相应的算子

以上述示例为例，我们最终会在output_code.py中看到如下的triton cv融合算子

```python
# AOT ID: ['0_inference']
from ctypes import c_void_p, c_long, c_int
import torch
import math
import random
import os
import tempfile
from math import inf, nan
from cmath import nanj
from torch._inductor.hooks import run_intermediate_hooks
from torch._inductor.utils import maybe_profile
from torch._inductor.codegen.memory_planning import _align as align
from torch import device, empty_strided
from torch._inductor.async_compile import AsyncCompile
from torch._inductor.select_algorithm import extern_kernels
from torch_npu._C import _npu_getCurrentRawStream as get_raw_stream
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import start_graph, end_graph
from torch_npu._C import _npu_getCurrentRawStream as get_raw_stream
import torch_npu
torch_npu.npu._initialized = torch_npu.npu.is_initialized()
has_initialized = False
import torch_npu._inductor.runtime.triton_heuristics as triton_heuristics

aten = torch.ops.aten
inductor_ops = torch.ops.inductor
_quantized = torch.ops._quantized
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
assert_alignment = torch._C._dynamo.guards.assert_alignment
empty_strided_cpu = torch._C._dynamo.guards._empty_strided_cpu
empty_strided_cpu_pinned = torch._C._dynamo.guards._empty_strided_cpu_pinned
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
empty_strided_xpu = torch._C._dynamo.guards._empty_strided_xpu
empty_strided_mtia = torch._C._dynamo.guards._empty_strided_mtia
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor
alloc_from_pool = torch.ops.inductor._alloc_from_pool
async_compile = AsyncCompile()
empty_strided_p2p = torch._C._distributed_c10d._SymmetricMemory.empty_strided_p2p



# Topologically Sorted Source Nodes: [mm], Original ATen: [aten.mm]
# Source node to ATen node mapping:
#   mm => mm
# Graph fragment:
#   %arg0_1 : Tensor "f16[200, 256][256, 1]npu:0" = PlaceHolder[target=arg0_1]
#   %arg1_1 : Tensor "f16[256, 256][1, 256]npu:0" = PlaceHolder[target=arg1_1]
#   %mm : Tensor "f16[200, 256][256, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mm.default](args = (%arg0_1, %arg1_1), kwargs = {})
#   return %mm
# SchedulerNodes: [SchedulerNode(name='op0')]

triton_npu_persistent_mm = async_compile.triton('triton_npu_persistent_mm', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties

import torch
import torch_npu
if not torch_npu.npu.is_initialized() and torch_npu.npu._is_in_bad_fork():
    torch_npu.npu._initialized = True
from torch_npu._inductor.runtime import triton_heuristics as triton_heuristics
from torch_npu._inductor.runtime import triton_helpers
from torch_npu._inductor.runtime.triton_helpers import libdevice, extension, math as tl_math

@triton_heuristics.template(
    num_stages=2,
    num_warps=4,
    triton_meta={'signature': {'arg_A': '*fp16', 'arg_B': '*fp16', 'out_ptr0': '*fp16'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=56, cc='Ascend950PR_9579', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=None, warp_size=None), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'kernel_name': 'triton_npu_persistent_mm', 'backend_hash': '5E441DC75500F0E6F24AA8AF77F3CC84D6023DAB25962F4AF0DC791A6526EC37', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'grid_type': 'FixedGrid', 'fixed_grid': ['_grid_0', '_grid_1', '_grid_2'], 'extra_launcher_args': ['_grid_0', '_grid_1', '_grid_2'], 'BLOCK_M': 128, 'BLOCK_N': 64},
)
@triton.jit
def triton_npu_persistent_mm(arg_A, arg_B, out_ptr0):
    BLOCK_M : tl.constexpr = 128
    BLOCK_N : tl.constexpr = 64
    BLOCK_K : tl.constexpr = 128
    GROUP_M : tl.constexpr = 2
    NUM_BLOCKS_M : tl.constexpr = 2
    NUM_BLOCKS_N : tl.constexpr = 4
    NUM_BLOCKS : tl.constexpr = 8
    WIDTH : tl.constexpr = 8
    NUM_SMS : tl.constexpr = 8
    NUM_TILES_PER_PROGRAM : tl.constexpr = 1
    ALLOW_TF32 : tl.constexpr = False
    ACC_TYPE : tl.constexpr = tl.float32
    EVEN_K : tl.constexpr = True
    INDEX_DTYPE : tl.constexpr = tl.int32
    A = arg_A
    B = arg_B

    M = 200
    N = 256
    K = 256
    if M * N == 0:
        # early exit due to zero-size input(s)
        return
    stride_am = 256
    stride_ak = 1
    stride_bk = 1
    stride_bn = 256

    start_pid = tl.program_id(0).to(INDEX_DTYPE)

    # Offsets shared across all tiles handled by this program
    rk_init = tl.arange(0, BLOCK_K)

    # Iterate over the tiles assigned to this program.
    # NUM_TILES_PER_PROGRAM is a compile-time constant (= ceil(NUM_BLOCKS / NUM_SMS)).
    for tile_iter in range(NUM_TILES_PER_PROGRAM):
        tile_id = start_pid + tile_iter * NUM_SMS
        if tile_id < NUM_BLOCKS:
            # ---- Super-grouping tile reordering ----
            # Map linear tile_id → (pid_m, pid_n) with diagonal traversal.
            # WIDTH = GROUP_M * NUM_BLOCKS_N (compile-time constant).
            group_id = tile_id // WIDTH
            pid_m = group_id * GROUP_M + (tile_id % GROUP_M)
            pid_n = (tile_id % WIDTH) // GROUP_M

            rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
            rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

            # ---- K-loop: accumulate matmul ----
            acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)
            for k_start in range(0, K, BLOCK_K):
                offs_k = k_start + rk_init

                a = tl.load(A + (rm[:, None] * stride_am + offs_k[None, :] * stride_ak))
                b = tl.load(B + (offs_k[:, None] * stride_bk + rn[None, :] * stride_bn))

                acc += tl.dot(a, b, allow_tf32=ALLOW_TF32, out_dtype=ACC_TYPE)

            # ---- Store output (inductor generates epilogue suffix) ----
            rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
            rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
            idx_m = rm[:, None]
            idx_n = rn[None, :]
            mask = (idx_m < M) & (idx_n < N)
            xindex = idx_n + 256*idx_m
            tl.store(out_ptr0 + (tl.broadcast_to(xindex, [BLOCK_M, BLOCK_N])), acc, mask)
''', device_str='npu')


async_compile.wait(globals())
del async_compile

class Runner:
    def __init__(self, partitions):
        self.partitions = partitions

    def recursively_apply_fns(self, fns):
        new_callables = []
        for fn, c in zip(fns, self.partitions):
            new_callables.append(fn(c))
        self.partitions = new_callables

    def call(self, args):
        arg0_1, arg1_1 = args
        args.clear()
        with torch.npu.utils.device(0):
            torch.npu.set_device(0)
            buf0 = empty_strided((200, 256), (256, 1), device='npu', dtype=torch.float16)
            # Topologically Sorted Source Nodes: [mm], Original ATen: [aten.mm]
            stream0 = get_raw_stream(0)
            triton_npu_persistent_mm.run(arg0_1, arg1_1, buf0, 8, 1, 1, stream=stream0)
            del arg0_1
            del arg1_1
        return (buf0, )

runner = Runner(partitions=[])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((200, 256), (256, 1), device='npu:0', dtype=torch.float16)
    arg1_1 = rand_strided((256, 256), (1, 256), device='npu:0', dtype=torch.float16)
    fn = lambda: call([arg0_1, arg1_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
```
