# 自动Tiling优化

## 概述

面向A5+PyTorch-v2.7.1，在Inductor-Ascend模块中提供对融合算子自动生成候选tiling集合的能力。Inductor中“VV（Vector‑Vector）融合算子”灵活多变，包括pointwise、规约、以及离散访存等不同类别，这些动态生成的融合算子在昇腾设备上需要寻找到最优tiling（包括编译选项）才能完全发挥其计算性能。同时由于和GPU微架构的区别，为了减少发射的逻辑核数量过多造成的硬件调度开销，Inductor-Ascend中会对一个维度进行两次切分：
1. 核间切分：控制每个核处理的数据总量（等价于控制发射的逻辑核数）；
2. 核内切分：控制单次计算搬运的数据量，通常称其为tiling大小。
这导致NPU上tiling调优相较于GPU会复杂很多，表现在可选择的候选tiling配置数量往往十分庞大（100k+），如果对这些全量候选tiling进行寻优会导致寻优时间开销不可接受，因此如何设计一个自动tiling生成算法，使得候选tiling配置数量适中，同时尽可能保证性能最优的tiling在候选集中，变得十分重要。

对生成的融合算子通过自动tiling生成算法产生了tiling候选集合后，需要在此集合中找到性能最优的tiling配置，由于一个模型中可能存在成百上千个融合算子，一个融合算子可能有成百上千个tiling配置，如果寻优过程完全串行执行会使得整体寻优开销时间非常长，因此需要考虑如何对autotune寻优进行并行加速，同时对于一个融合算子的最优配置需要具有缓存功能，避免后续多次运行的重复寻优开销。

## 关键概念解释

### Tiling（平铺）
Tiling 是计算优化中的核心概念，指将大循环/数组计算分解成小块（tiles 或 blocks），以匹配硬件缓存（cache）或共享内存大小，减少内存访问开销，提高局部性（locality）。在 Inductor 中，tiling 主要应用于 Triton 内核（GPU），将高维张量操作拆分成 block-level 计算。
关键子概念： 
+ Tile Size：每个块的大小（e.g., [32, 32]），需根据矩阵尺寸、寄存器压力和共享内存调优。
+ Block：Triton 中每个核的执行单元。
+ SubBlock： 每次循环的执行单元。

### Autotune（自动调优）
Autotune 是 Inductor 的参数搜索机制，通过编译时测试多种配置（tile sizes、thread blocks、融合策略等），选择性能最佳的版本。Inductor 支持多种 autotune 级别，如 max-autotune。
关键子概念：
+ Heuristics：预设规则快速缩小搜索空间（e.g., 根据矩阵大小选 tile）。
+ Profiling：实际运行小基准测试不同配置的性能。

## 作用
### Tiling 的作用：
+ 优化内存访问：大矩阵/卷积拆分成小块，减少全局内存读写，提升缓存命中率（locality）。
+ 支持融合：tiling 后易融合相邻操作（e.g., matmul + add + relu → fused kernel）。
+ 处理动态形状：通过 SymPy 符号化索引，支持 variable size 的 tiling。
+ 整体提升：训练/推理速度提高 10-50%（视操作而定），尤其 GEMM、conv、reduction。

### Autotune 的作用：
+ 自动选优：避免手动调参，针对硬件/输入自动找最佳 tile size、block config 等。
+ 平衡编译时间与运行性能：max-autotune 接受长编译时间，换取更高运行时速度。
+ 适应后端：Triton、CUTLASS、C++ template 等后端间 autotune，提升跨硬件兼容性。
+ 可使用多进程多kernel并发编译以及单kernel内部多线程并发编译每个tiling config进行precompile阶段的加速。

Tiling + Autotune 结合作用：Autotune 自动测试多种 tiling 配置，选择最佳 tile size/block，实现性能自适应。


## 原理

### Tiling 原理：
+ Loop Decomposition：Inductor 的 Loop-level IR 将高维循环拆分成 tiled blocks（e.g., [XBLOCK, YBLOCK]）。每个 block 内计算小片数据，利用 shared memory / register。
+ Symbolic Representation：用 SymPy 符号（如 i0 = tl.program_id(0) * XBLOCK + tl.arange(0, XBLOCK)）表示索引，支持动态形状和 masking（offsets < size）。
+ 融合兼容：tiling 后易做 horizontal/vertical fusion（e.g., matmul + relu 在同一 tiled kernel 内）。
+ 卡上（NPU/GPU） 实现：@triton.jit 内核中，tiling 通过 tl.arange 和 mask 实现；CPU 用 C++ 循环 nesting。
+ 性能原理：减少内存 bandwidth 消耗，提升 arithmetic intensity（计算/内存访问比）。

### Autotune 原理：
+ Search Space：定义 heuristics（预设规则）缩小候选（如 tile sizes [16,32,64]、thread blocks [128,256,512]）。
+ Profiling：编译时运行小基准（benchmark），测量每个配置的运行时间，选择最快。
+ Multi-Backend：autotune 跨后端（如 Triton vs CUTLASS vs C++ template）选择最佳。
+ Compile-Time Overhead：max-autotune 模式下，autotune 发生在首次编译，增加时间，但后续运行更快。
