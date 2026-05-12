# 概述

本文档介绍 Ascend NPU 的静态图优化能力，涵盖 NPUGraph 图捕获技术和 PyTorch `torch.compile()` 编译接口两大模块。两者都旨在通过静态图优化减少运行时开销、提升训练和推理性能，但作用层级和使用方式不同。

## NPUGraph vs Inductor：两种优化路径

| 维度 | NPUGraph | Inductor |
|------|----------|----------|
| **核心机制** | 图捕获与重放（Capture-Replay） | 算子融合与代码生成 |
| **对标** | CUDA Graphs | PyTorch Inductor（原生） |
| **优化目标** | 消除 CPU 端 kernel 启动开销 | 减少内存访问、提升 NPU 利用率 |
| **工作方式** | 将已有的 NPU kernel 序列打包为静态图，一次捕获、多次复跑 | 将多个算子融合为更大的 kernel，减少 kernel 数量和内存读写 |
| **输入形状** | 必须固定 | 支持动态（但会触发重新编译） |
| **算子要求** | 仅支持 aclnn 算子 | 支持更广泛的算子，不支持的会 fallback |
| **适用场景** | kernel 调用频繁、CPU 瓶颈、高迭代次数 | 单步计算量大、可通过算子融合获益 |

**简单理解：**

- NPUGraph 不改变已有的 kernel，只是把它们"打包"成一批次提交，减少 CPU 逐个启动的间隙
- Inductor 通过算子融合把多个小 kernel 合并为一个大 kernel，从根源上减少 kernel 数量和内存访问

## NPUGraph vs torch.compile：API 层级不同

| 维度 | NPUGraph 底层 API | torch.compile() |
|------|-------------------|-----------------|
| **层级** | 手动图捕获（L4 API） | 自动图编译（JIT 编译器） |
| **使用方式** | 手动调用 `capture_begin()` / `capture_end()` 或使用 `graph()` 上下文管理器 | 一行 `torch.compile(model, backend=...)` 自动完成 |
| **控制粒度** | 精细，可手动管理 Stream、分区域捕获 | 自动化，由 Dynamo 前端决定图断裂点 |
| **学习成本** | 较高，需理解捕获-重放-更新机制 | 较低，与原生训练代码完全兼容 |

`torch.compile()` 是一个更上层的编译接口，其 `backend="npugraphs"` 后端本质上就是自动化的 NPUGraph 图捕获。

## 我该用什么？

**使用 NPUGraph 底层 API（`NPUGraph` / `graph()` / `make_graphed_callables`）当：**

- 你需要精细控制哪些代码被图化、哪些保持 eager 执行
- 模型中包含动态控制流（if/for 依赖张量值），需要手动划分安全子图
- 需要多 Stream 管理、分区域捕获等底层操作

**使用 `torch.compile(backend="inductor")` 当：**

- 模型计算量较大，可通过算子融合获益
- 输入形状可能有变化（支持动态形状）
- 不确定该用什么后端时，优先尝试 Inductor（默认后端）

**使用 `torch.compile(backend="npugraphs")` 当：**

- 输入形状完全固定
- kernel 调用频繁，存在 CPU 瓶颈
- 高迭代次数的训练或推理任务，需要彻底消除启动开销

**使用 `torch.compile(mode="reduce-overhead")` 当：**

- 需要 NPUGraph 的优化收益，但同时希望支持一定程度的动态形状
- 内部会管理多个 NPUGraph 子图，根据输入形状自动路由

## 文档结构

本模块包含以下文档：

- **[NPUGraph](./pytorch_npugraph_desc.md)** — NPUGraph 图捕获技术详解（Capture-Replay-Update 机制、底层 API 使用）
- **[PyTorch 图模式（torch.compile）](./pytorch_graph_mode.md)** — torch.compile 编译接口概述和各后端说明
