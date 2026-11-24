# Triton cv特性介绍

环境变量配置请参见[Triton CV配置项](./triton_cv.md)。

## 特性简介

在深度学习领域，矩阵乘法MM类算子是计算量最大的核心操作。传统的 PyTorch eager 模式下，矩阵乘法和随后pointwise（逐点的）算子（如激活函数 ReLU（修正线性单元）、SiLU（Sigmoid Linear Unit）、Bias Add（偏置加法）等）通常作为独立的Kernel依次执行。这种方式会导致频繁的Kernel启动开销和大量的全局内存数据传输（I/O 瓶颈），严重影响性能。

另外，社区PyTorch Inductor通过Triton后端可以支持cv融合，因此torch_npu的inductor中也要对标社区功能，为cv融合提供相应的Triton后端。

## Triton cv特性使用方法

### Triton cv使用示例

```python
import os
os.environ["TORCHINDUCTOR_MAX_AUTOTUNE"] = "1"
os.environ["TORCHINDUCTOR_MAX_AUTOTUNE_GEMM_BACKENDS"] = "TRITON,ATEN"
os.environ["TORCH_COMPILE_DEBUG"] = "1"

import torch
import torch_npu


def forward_mm(a, b):
    return torch.relu(torch.mm(a, b))


a = torch.randn((200, 256), device="npu", dtype=torch.float16)
b = torch.randn((256, 256), device="npu", dtype=torch.float16).transpose(0, 1)
compiled_mm = torch.compile(forward_mm)
result = compiled_mm(a, b)
torch.npu.synchronize()
torch.testing.assert_close(result, forward_mm(a, b), rtol=1e-2, atol=1e-2)
print(result.shape)
```

### Triton cv应用效果

#### 编译进行中

在编译执行的过程中，会看到Triton cv在进行autotune过程的日志，以上述的示例为例，会有类似如下的日志输出：

```output
AUTOTUNE mm(200x256, 256x256)
strides: [256, 1], [1, 256]
dtypes: torch.float16, torch.float16
triton_npu_triton_mm_25 0.0031 ms 100.0% ACC_TYPE='tl.float32', ALLOW_HF32=False, ALLOW_TF32='False', BLOCK_K=128, BLOCK_M=100, BLOCK_N=64, GROUP_M=2, MASKED_K=False, MASKED_M=False, MASKED_N=False, num_stages=3, num_warps=4
triton_npu_triton_mm_22 0.0032 ms 98.1% ACC_TYPE='tl.float32', ALLOW_HF32=False, ALLOW_TF32='False', BLOCK_K=128, BLOCK_M=100, BLOCK_N=128, GROUP_M=2, MASKED_K=False, MASKED_M=False, MASKED_N=False, num_stages=3, num_warps=4
triton_npu_persistent_mm_70 0.0032 ms 97.3% ACC_TYPE='tl.float32', ALLOW_HF32=False, ALLOW_TF32='False', BLOCK_K=128, BLOCK_M=100, BLOCK_N=64, GROUP_M=2, MASKED_K=False, MASKED_M=False, MASKED_N=False, NUM_BLOCKS=8, NUM_BLOCKS_M=2, NUM_BLOCKS_N=4, NUM_SMS=8, NUM_TILES_PER_PROGRAM=1, WIDTH=8, num_stages=3, num_warps=4
triton_npu_triton_mm_23 0.0032 ms 97.1% ACC_TYPE='tl.float32', ALLOW_HF32=False, ALLOW_TF32='False', BLOCK_K=128, BLOCK_M=100, BLOCK_N=128, GROUP_M=2, MASKED_K=False, MASKED_M=False, MASKED_N=False, num_stages=4, num_warps=4
triton_npu_persistent_mm_76 0.0032 ms 96.9% ACC_TYPE='tl.float32', ALLOW_HF32=False, ALLOW_TF32='False', BLOCK_K=128, BLOCK_M=50, BLOCK_N=128, GROUP_M=4, MASKED_K=False, MASKED_M=False, MASKED_N=False, NUM_BLOCKS=8, NUM_BLOCKS_M=4, NUM_BLOCKS_N=2, NUM_SMS=8, NUM_TILES_PER_PROGRAM=1, WIDTH=8, num_stages=3, num_warps=4
triton_npu_persistent_mm_71 0.0032 ms 96.4% ACC_TYPE='tl.float32', ALLOW_HF32=False, ALLOW_TF32='False', BLOCK_K=128, BLOCK_M=100, BLOCK_N=64, GROUP_M=2, MASKED_K=False, MASKED_M=False, MASKED_N=False, NUM_BLOCKS=8, NUM_BLOCKS_M=2, NUM_BLOCKS_N=4, NUM_SMS=8, NUM_TILES_PER_PROGRAM=1, WIDTH=8, num_stages=4, num_warps=4
triton_npu_persistent_mm_68 0.0032 ms 96.3% ACC_TYPE='tl.float32', ALLOW_HF32=False, ALLOW_TF32='False', BLOCK_K=128, BLOCK_M=100, BLOCK_N=128, GROUP_M=2, MASKED_K=False, MASKED_M=False, MASKED_N=False, NUM_BLOCKS=4, NUM_BLOCKS_M=2, NUM_BLOCKS_N=2, NUM_SMS=4, NUM_TILES_PER_PROGRAM=1, WIDTH=4, num_stages=4, num_warps=4
triton_npu_persistent_mm_67 0.0032 ms 96.2% ACC_TYPE='tl.float32', ALLOW_HF32=False, ALLOW_TF32='False', BLOCK_K=128, BLOCK_M=100, BLOCK_N=128, GROUP_M=2, MASKED_K=False, MASKED_M=False, MASKED_N=False, NUM_BLOCKS=4, NUM_BLOCKS_M=2, NUM_BLOCKS_N=2, NUM_SMS=4, NUM_TILES_PER_PROGRAM=1, WIDTH=4, num_stages=3, num_warps=4
triton_npu_persistent_mm_77 0.0032 ms 95.9% ACC_TYPE='tl.float32', ALLOW_HF32=False, ALLOW_TF32='False', BLOCK_K=128, BLOCK_M=50, BLOCK_N=128, GROUP_M=4, MASKED_K=False, MASKED_M=False, MASKED_N=False, NUM_BLOCKS=8, NUM_BLOCKS_M=4, NUM_BLOCKS_N=2, NUM_SMS=8, NUM_TILES_PER_PROGRAM=1, WIDTH=8, num_stages=4, num_warps=4
triton_npu_triton_mm_32 0.0032 ms 95.7% ACC_TYPE='tl.float32', ALLOW_HF32=False, ALLOW_TF32='False', BLOCK_K=128, BLOCK_M=50, BLOCK_N=128, GROUP_M=4, MASKED_K=False, MASKED_M=False, MASKED_N=False, num_stages=4, num_warps=4
SingleProcess AUTOTUNE benchmarking takes 1.2920 seconds and 2.5798 seconds precompiling for 91 choices
Wxxx.708000 2317919 site-packages/torch/_inductor/debug.py:518] [0/0] model__0_inference_0 debug trace: xxx
[xxx] [WARNING] [2725544] profiler.py: Incorrect schedule: Stop profiler while current state is RECORD which may result in incomplete parsed data.
[xxx] [INFO] [2725544] profiler.py: Start parsing profiling data: xxx
[xxx] [INFO] [2725544] profiler.py: SimpleProfilingAnalyzer parsed in a total time of 0:00:00.186119
```

日志显示，Inductor正在对形状为`(200, 256)`和`(256, 256)`的两个矩阵的乘法进行自动调优。

#### 运行结果

在autotune的过程中，被选为最优的算子时，我们在torch_compile_debug的output_code.py中，即可看到相应的算子

当调优选中 Triton 矩阵乘模板且 ReLU 成功融合时，可在 `torch_compile_debug` 目录下的 `output_code.py` 中看到如下融合 kernel。

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
from torch._C._dynamo.guards import copy_if_misaligned
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
import torch_npu
empty_strided_npu = torch_npu._C._empty_strided_npu


# Topologically Sorted Source Nodes: [mm, relu], Original ATen: [aten.mm, aten.relu]
# Source node to ATen node mapping:
#   mm => mm
#   relu => relu
# Graph fragment:
#   %arg0_1 : Tensor "f16[200, 256][256, 1]npu:0" = PlaceHolder[target=arg0_1]
#   %arg1_1 : Tensor "f16[256, 256][1, 256]npu:0" = PlaceHolder[target=arg1_1]
#   %mm : Tensor "f16[200, 256][256, 1]npu:0" = PlaceHolder[target=mm]
#   %mm : Tensor "f16[200, 256][256, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mm.default](args = (%arg0_1, %arg1_1), kwargs = {})
#   %relu : Tensor "f16[200, 256][256, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%mm,), kwargs = {})
#   return %mm,%relu
# SchedulerNodes: [SchedulerNode(name='op0'), SchedulerNode(name='op1')]

triton_tem_fused_mm_relu_0 = async_compile.triton('triton_tem_fused_mm_relu_0', '''
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
    num_stages=3,
    num_warps=4,
    triton_meta={'signature': {'arg_A': '*fp16', 'arg_B': '*fp16', 'out_ptr1': '*fp16'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=56, cc='Ascend950PR_9579', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=None, warp_size=None), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'kernel_name': 'triton_tem_fused_mm_relu_0', 'backend_hash': 'ACD81A46BCFF346FDD2860106A525F1D988E397114CE6CCE2A7D252E0034D2FD', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False, 'grid_type': 'FixedGrid', 'fixed_grid': ['_grid_0', '_grid_1', '_grid_2'], 'extra_launcher_args': ['_grid_0', '_grid_1', '_grid_2'], 'BLOCK_M': 100, 'BLOCK_N': 64},
)
@triton.jit
def triton_tem_fused_mm_relu_0(arg_A, arg_B, out_ptr1):
    BLOCK_M : tl.constexpr = 100
    BLOCK_N : tl.constexpr = 64
    BLOCK_K : tl.constexpr = 128
    GROUP_M : tl.constexpr = 2
    ALLOW_HF32 : tl.constexpr = False
    ACC_TYPE : tl.constexpr = tl.float32
    MASKED_M : tl.constexpr = False
    MASKED_N : tl.constexpr = False
    MASKED_K : tl.constexpr = False
    ALLOW_TF32 : tl.constexpr = False
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

    # based on triton.ops.matmul
    pid = tl.program_id(0).to(INDEX_DTYPE)
    grid_m = (M + BLOCK_M - 1) // BLOCK_M
    grid_n = (N + BLOCK_N - 1) // BLOCK_N

    # re-order program ID for better L2 performance
    # Use super-grouping (GROUP_M) for improved L2 cache hit rate
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // group_size

    tl.assume(pid_m >= 0)
    tl.assume(pid_n >= 0)

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)




    offs_k = tl.arange(0, BLOCK_K)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)

    for k_idx in range(0, tl.cdiv(K, BLOCK_K)):
        offs_k = k_idx * BLOCK_K + tl.arange(0, BLOCK_K)


        a = tl.load(A + (rm[:, None] * stride_am + offs_k[None, :] * stride_ak))


        b = tl.load(B + (offs_k[:, None] * stride_bk + rn[None, :] * stride_bn))

        acc += tl.dot(a, b, out_dtype=ACC_TYPE)

    # rm, rn, and m_mask are reused from above (no rematerialization).
    idx_m = rm[:, None]
    idx_n = rn[None, :]
    mask = (idx_m < M) & (idx_n < N)

    # inductor generates a suffix
    xindex = idx_n + 256*idx_m
    tmp0 = tl.full([1], 0, tl.int32)
    tmp1 = triton_helpers.maximum(tmp0, acc)
    tl.store(out_ptr1 + (tl.broadcast_to(xindex, [BLOCK_M, BLOCK_N])), tmp1, mask)
    tl.store(out_ptr1 + (tl.broadcast_to(xindex, [BLOCK_M, BLOCK_N])), tmp1, mask)
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
            arg0_1 = copy_if_misaligned(arg0_1)
            arg1_1 = copy_if_misaligned(arg1_1)
            buf1 = empty_strided_npu((200, 256), (256, 1), torch.float16)
            # Topologically Sorted Source Nodes: [mm, relu], Original ATen: [aten.mm, aten.relu]
            raw_stream0 = get_raw_stream(0)
            triton_tem_fused_mm_relu_0.run(arg0_1, arg1_1, buf1, 8, 1, 1, stream=raw_stream0)
            del arg0_1
            del arg1_1
        return (buf1, )

runner = Runner(partitions=[])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns


def get_args():
    from torch._dynamo.testing import rand_strided
    arg0_1 = rand_strided((200, 256), (256, 1), device='npu:0', dtype=torch.float16)
    arg1_1 = rand_strided((256, 256), (1, 256), device='npu:0', dtype=torch.float16)
    return [arg0_1, arg1_1]


def benchmark_compiled_module(args, times=10, repeat=10):
    from torch._inductor.utils import print_performance
    fn = lambda: call(list(args))
    return print_performance(fn, times=times, repeat=repeat, device='npu')


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    args = get_args()
    compiled_module_main('None', lambda times, repeat: benchmark_compiled_module(args, times=times, repeat=repeat))

```
