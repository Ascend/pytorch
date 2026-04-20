# Catlass特性介绍

## 特性简介

在深度学习领域，矩阵乘法（GEMM: 通用矩阵乘发，MM: 矩阵乘法，BMM: 批量矩阵乘法，MMADD: 矩阵乘加）是计算量最大的核心操作。传统的 PyTorch eager 模式下，矩阵乘法和随后pointwise（逐点的）算子（如激活函数 ReLU（修正线性单元）、SiLU（Sigmoid Linear Unit）、Bias Add（偏置加法） 等）通常作为独立的Kernel依次执行。这种方式会导致频繁的Kernel启动开销和大量的全局内存数据传输（I/O 瓶颈），严重影响性能。
NVIDIA的CUTLASS库提供了高度优化的GEMM Kernel，并支持在Epilogue阶段（Kernel 的最后阶段）融合自定义的后处理操作。通过将部分支持的pointwise类算子直接融合到 GEMM Kernel 中，可以显著减少内存访问和Kernel启动次数，提升整体性能。

在昇腾领域，Catlass对标NVIDIA Cutlass。因此，为了在PyTorch编译模式下，提供对等的方案，我们在inductor-ascend中也引入Catlass；在此基础上，也支持部分pointwise类进行epilogue融合。

## Catlass特性使用方法

### Catlass使用示例

```python
import os

os.environ["TORCHINDUCTOR_MAX_AUTOTUNE"] = "1"  # 开启max autotune
os.environ["TORCHINDUCTOR_MAX_AUTOTUNE_GEMM_BACKENDS"] = "CATLASS,ATen"  # 对matmul类算子尝试使用CATLASS和ATen后端调优
os.environ["TORCHINDUCTOR_NPU_CATLASS_DIR"] = "/path/to/catlass/dir"  # 设置环境中Catlass库所在的路径
os.environ["CATLASS_EPILOGUE_FUSION"] = "1"  # 开启Catlass epilogue的CV融合功能
os.environ["TORCHINDUCTOR_CATLASS_ENABLED_OPS"] = "mm,addmm,bmm,grouped_mm" # 可对mm,addmm,bmm和grouped_mm类的matmul算子进行CATLASS模版调优尝试

import torch
import torch.nn.functional as F
from torch._dynamo.testing import rand_strided
import torch_npu


def forward_mm(a, b):
    x = torch.mm(a, b)
    x = F.silu(x)
    return x


torch_npu.npu.matmul.allow_hf32 = True

shapeA, strideA = (512, 256), (256, 1)
shapeB, strideB = (256, 1024), (1024, 1)
a = rand_strided(shapeA, strideA, device="npu", dtype=torch.float32)
b = rand_strided(shapeB, strideB, device="npu", dtype=torch.float32)

forward_mm = torch.compile(forward_mm, backend="inductor")  # 使用PyTorch编译模式，选择后端backend="inductor"
compile_result = forward_mm(a, b)  # 获取运行结果
```

### Catlass应用效果

#### 编译进行中

在编译执行的过程中，会看到Catlass在进行autotune过程的日志，以上述的示例为例，会有类似如下的日志输出

```shell
[xxx] [INFO] [10987] profiler.py: CANN profiling data parsed in a total time of 0:00:02.025342
[xxx] [INFO] [10978] profiler.py: All profiling data parsed in a total time of 0:00:02.994265
AUTOTUNE mm(512x256, 256x1024)
  catlass_gemm_3 0.0083 ms 100.0% AtlasA5_BasicMatmulTla_MmadPingpong_GemmIdentityBlockSwizzle_3_0_80_128_256_80_128_64
  catlass_gemm_0 0.0085 ms 98.1% AtlasA5_BasicMatmulTla_MmadPingpong_GemmIdentityBlockSwizzle_3_0_128_160_128_128_160_48
  mm 0.0086 ms 96.9% 
  catlass_gemm_1 0.0088 ms 94.2% AtlasA5_BasicMatmulTla_MmadPingpong_GemmIdentityBlockSwizzle_3_0_112_96_256_112_96_64
  catlass_gemm_2 0.0107 ms 77.7% AtlasA5_BasicMatmulTla_MmadPingpong_GemmIdentityBlockSwizzle_3_0_128_80_256_128_80_64
SingleProcess AUTOTUNE benchmarking takes 3.5552 seconds and 2.3107 seconds precompiling for 5 choices
```

如上所示的示例中，表明了inductor正在对 (512, 256) x (256, 1024)的matmul操作进行autotune的过程，标注了‘Catlass_xxx’的算子即为不同config下的Catlass算子，
而'mm'的算子则表示为aclnn对应的matmul算子

#### 运行结果

当Catlass在autotune的过程中，被选为最优的算子时，我们在torch_compile_debug的output_code.py中，即可看到相应的Catlass算子

以上述示例为例，因为上述示例的尾块vector算子silu还可以与Catlass进行CV融合，我们最终会在output_code.py中看到如下的Catlass cv融合算子

```python
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
from torch._inductor.codegen.multi_kernel import MultiKernelCall
import torch_npu
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import start_graph, end_graph
import torch_npu
has_initialized = False
from torch_npu._C import _npu_getCurrentRawStream as get_raw_stream

aten = torch.ops.aten
inductor_ops = torch.ops.inductor
_quantized = torch.ops._quantized
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cpu = torch._C._dynamo.guards._empty_strided_cpu
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
empty_strided_xpu = torch._C._dynamo.guards._empty_strided_xpu
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor
alloc_from_pool = torch.ops.inductor._alloc_from_pool
async_compile = AsyncCompile()
empty_strided_p2p = torch._C._distributed_c10d._SymmetricMemory.empty_strided_p2p


# kernel path: /tmp/torchinductor_root/xx/xxx.py
# Topologically Sorted Source Nodes: [x_1], Original ATen: [aten.silu]
# Source node to ATen node mapping:
#   x_1 => mul, sigmoid
# Graph fragment:
#   %sigmoid : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%mm,), kwargs = {})
#   %mul : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mm, %sigmoid), kwargs = {})
catlass_fused_silu_0 = async_compile.catlass(r'''
#include <exception>
#include <iostream>
#include <memory>
#include <random>
#include <vector>

#include <acl/acl.h>
#include <runtime/rt_ffts.h>
#include <tiling/platform/platform_ascendc.h>
#include <catlass/catlass.hpp>
#include <catlass/arch/arch.hpp>
#include <catlass/layout/layout.hpp>
#include <catlass/status.hpp>
#include <catlass/gemm/block/block_mmad.hpp>
#include <catlass/gemm/block/block_swizzle.hpp>
#include <catlass/gemm/dispatch_policy.hpp>
#include <catlass/gemm/gemm_type.hpp>
#include <catlass/gemm/device/device_gemm.hpp>
#include <catlass/gemm_coord.hpp>
#include <catlass/matrix_coord.hpp>
#include <catlass/epilogue/block/block_epilogue.hpp>
#include <catlass/epilogue/fusion/fusion.hpp>
#include <catlass/gemm/kernel/basic_matmul_tla_visitor.hpp>


// We compile all models with -fvisibility=hidden. Any symbols that need to be
// exposed in the final shared library must be declared with PT_EXPORT to make
// them visible.
#ifdef __GNUC__ // Applies to any compiler with GNU extensions (clang and g++)
#define PT_EXPORT __attribute__((__visibility__("default")))
#else
#ifdef _WIN32
#define PT_EXPORT __declspec(dllexport)
#else
#define PT_EXPORT
#endif
#endif

#define ACL_CHECK(status)                                                                    \
    do {                                                                                     \
        aclError error = status;                                                             \
        if (error != ACL_ERROR_NONE) {                                                       \
            std::cerr << __FILE__ << ":" << __LINE__ << " aclError:" << error << std::endl;  \
        }                                                                                    \
    } while (0)

// Macro function for unwinding rt errors.
#define RT_CHECK(status)                                                                     \
    do {                                                                                     \
        rtError_t error = status;                                                            \
        if (error != RT_ERROR_NONE) {                                                        \
            std::cerr << __FILE__ << ":" << __LINE__ << " rtError:" << error << std::endl;   \
        }                                                                                    \
    } while (0)

using namespace Catlass;
using namespace tla;

// Macro function for unwinding Catlass errors.
#define CATLASS_CHECK(status)                                                                \
    do {                                                                                     \
        Catlass::Status error = status;                                                      \
        if (error != Catlass::Status::kSuccess) {                                            \
            std::cerr << __FILE__ << ":" << __LINE__ << " raise catlassError" << std::endl;  \
        }                                                                                    \
    } while (0)

bool IsNeedPadding(layout::RowMajor layout, uint32_t align)
{
    // If the stride is greater than 65536, padding is required to reduce the stride.
    if (layout.stride(0) < 65536) {
        return layout.stride(0) % align != 0;
    } else {
        return true;
    }
}

bool IsNeedPadding(layout::ColumnMajor layout, uint32_t align)
{
    // If the stride is greater than 65536, padding is required to reduce the stride.
    if (layout.stride(1) < 65536) {
        return layout.stride(1) % align != 0;
    } else {
        return true;
    }
}

bool IsNeedPadding(layout::zN layout, uint32_t align)
{
    return false;
}

bool IsNeedPadding(layout::nZ layout, uint32_t align)
{
    return false;
}

// When workspace_size is not a nullptr, populates requested workspace_size and returns.
// Otherwise, computes the Gemm kernel using the given workspace ptr.

extern "C" {
PT_EXPORT int catlass_fused_silu_0(const float* X, const float* W, const float* Y, const int M, const int N, const int K, size_t* workspace_size, uint8_t* workspace, aclrtStream stream) {
    try {
    uint8_t* deviceA = (uint8_t*)(X);
    uint8_t* deviceB = (uint8_t*)(W);
    uint8_t* deviceBias = nullptr;
    uint8_t* deviceC = (uint8_t*)(Y);


        uint32_t m = M;
    uint32_t k = K;
    uint32_t n = N;





    using EpilogueDispatchPolicy = Epilogue::EpilogueVisitor<false>;

using Accum = Catlass::Epilogue::Fusion::VisitorAccLoad<float>;

using Compute0 = Catlass::Epilogue::Fusion::VisitorCompute<
    Catlass::Epilogue::Fusion::Sigmoid, float
>;

using Compute1 = Catlass::Epilogue::Fusion::VisitorCompute<
    Catlass::Epilogue::Fusion::Mul, float
>;

using TvCompute1 = Catlass::Epilogue::Fusion::TopologicalVisitor<
    tla::tuple<
        tla::seq<>,
        tla::seq<0>,
        tla::seq<0, 1>
    >,
    Accum,
    Compute0,
    Compute1
>;

using LayoutTagD = Catlass::layout::RowMajor;
LayoutTagD tagD{512, 1024};
auto layoutD = tla::MakeLayoutFromTag(tagD);

using D = Catlass::Epilogue::Fusion::VisitorAuxStore<
    float, decltype(layoutD)
>;

using EVGD = Catlass::Epilogue::Fusion::TreeVisitor<
    D,
    TvCompute1
>;

typename EVGD::Arguments evg_args{
    {
            {},
        {},
        {}
    },
    {deviceC, layoutD}
};

constexpr uint32_t computeLength = 216 * 1024 / ((3 * sizeof(float))) / 2 / 32 * 32;



        using ArchTag = Arch::AtlasA5;
    constexpr bool enableUnitFlag = false;
    constexpr bool useHf32Mode = true;
    constexpr uint32_t l0cStages = 1;
    constexpr bool enableL1Resident = false;
    constexpr uint32_t l1aStages = 2;
    constexpr uint32_t l1bStages = 2;
    constexpr uint32_t l0aStages = 2;
    constexpr uint32_t l0bStages = 2;
    using DispatchPolicy = Gemm::MmadPingpong<ArchTag, enableUnitFlag, useHf32Mode, l0cStages, enableL1Resident, l1aStages, l1bStages, l0aStages, l0bStages>;
    using L1TileShape = Shape<Int<80>, Int<128>, Int<256>>;
    using L0TileShape = Shape<Int<80>, Int<128>, Int<64>>;

    using ElementA = float;
    using ElementB = float;
    using ElementC = float;
    using LayoutTagA = layout::RowMajor;
    using LayoutTagB = layout::RowMajor;
    using LayoutTagC = layout::RowMajor;

    using TileCopy = Gemm::Tile::PackedTileCopyTla<ArchTag, ElementA, LayoutTagA, ElementB, LayoutTagB, ElementC, LayoutTagC>;
    using BlockMmad = Gemm::Block::BlockMmadTla<DispatchPolicy, L1TileShape, L0TileShape, ElementA, ElementB, ElementC, void, TileCopy>;
    using BlockEpilogue = Epilogue::Block::BlockEpilogue<
        EpilogueDispatchPolicy,
        ArchTag,
        Int<computeLength>,
        EVGD,
        ElementC
    >;

    using BlockScheduler = typename Gemm::Block::GemmIdentityBlockSwizzle<3, 0>;
    using GemmKernel = Gemm::Kernel::BasicMatmulTlaVisitor<BlockMmad, BlockEpilogue, BlockScheduler>;


        GemmCoord problemShape{m, n, k};
    // Define the layout of each matrix
    LayoutTagA tagA{m, k};
    LayoutTagB tagB{k, n};
    LayoutTagC tagC{m, n};
    auto layoutA = tla::MakeLayoutFromTag(tagA);
    auto layoutB = tla::MakeLayoutFromTag(tagB);
    auto layoutC = tla::MakeLayoutFromTag(tagC);


    using GemmAdapter = Gemm::Device::DeviceGemm<GemmKernel>;

    auto aicCoreNum = platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAic();

    typename GemmKernel::Arguments arguments{
        problemShape, deviceA, layoutA, deviceB, layoutB, nullptr, {}, nullptr, evg_args
    };
    GemmAdapter gemm_op;

    if (workspace_size) {
        *workspace_size = gemm_op.GetWorkspaceSize(arguments);
        return 0;
    }

    {
        auto status = gemm_op.CanImplement(arguments);
        CATLASS_CHECK(status);
    }
    {
        auto status = gemm_op.Initialize(arguments, workspace);
        CATLASS_CHECK(status);
    }
    {
        auto status = gemm_op(stream, aicCoreNum);
        CATLASS_CHECK(status);
    }

    }
    catch (std::exception& e) {
        std::cerr << "Runtime error: " << e.what() << std::endl;
        return -1;
    }
    catch (...) {
        return -1;
    }
    return 0;
}
}
''', 'so', aot_compile=False, is_mix=True)


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1 = args
    args.clear()
    buf1 = empty_strided((512, 1024), (1024, 1), device='npu', dtype=torch.float32)
    with torch.npu.utils.device(0):
        torch.npu.set_device(0)
        workspace_0 = empty_strided((2097152, ), (1, ), device='npu', dtype=torch.uint8)
        stream0 = get_raw_stream(0)
        catlass_fused_silu_0.catlass_fused_silu_0(c_void_p(arg1_1.data_ptr()), c_void_p(arg0_1.data_ptr()), c_void_p(buf1.data_ptr()), 512, 1024, 256, None, c_void_p(workspace_0.data_ptr()), c_void_p(stream0))
        del workspace_0
        del arg0_1
        del arg1_1
    return (buf1, )
```