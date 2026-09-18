# 概述

`torch.compile()`是PyTorch 2.0推出的核心编译接口，通过“动态图捕获+静态图优化+高效代码生成”的方式显著加速模型训练和推理任务。TorchNPU 7.3.0版本开始支持`torch.compile()`功能，仅需一行代码即可自动实现前端图捕获和后端优化，适用于全自动编译的场景。

torch.compile包含如下核心组件：

| 组件              | 作用       |
|-------------------|-----------|
| Dynamo前端        |Dynamo能够JIT（即时）将用户的eager（动态图）代码编译为FX Graph（PyTorch的中间表示）。 |
| AOT Autograd      |提前捕获反向传播图，使前向和反向传播都可以由后端进行优化。|
| 编译后端          |对FX Graph进行优化并生成最终可执行的代码。|

通用用法和核心概念请参见本文的[参考文档](#参考文档)章节。

## 接口说明

### 接口原型

```python
torch.compile(model, *, fullgraph=False, dynamic=None, backend="inductor",
              mode=None, options=None, disable=False)
```

### 参数说明

| 参数 | 数据类型 | 默认值 | 说明 |
|------|------|--------|------|
| model | nn.Module | 必填 | 待编译的模型 |
| fullgraph | bool | False | 是否强制整图编译 |
| dynamic | bool | None | 是否启用动态shape编译 |
| backend | str/Callable | `"inductor"` | 编译后端：`inductor`、`npugraphs`、`npugraph_ex`、`aot_eager`、`TorchAir-GE后端(Callable)` |
| mode | str | None | 编译模式：`None`或`"reduce-overhead"` （仅`inductor`后端支持）|
| options | dict | None | 编译选项|
| disable | bool | False | 关闭torch.compile |

更多参数详情可参见[torch.compile](https://docs.pytorch.org/docs/stable/generated/torch.compile.html)。

**编译后端说明**

| 后端 | 开启方式 | 核心机制 | 适用场景 |
|------|----------|---------|---------|
| Inductor（默认） | `backend="inductor"` | 算子融合 + 代码生成（Triton/MLIR/DVM/Ascend C） | 大多数场景，不确定时首选 |
| NPUGraphs | `backend="npugraphs"` | ACLGraph图下沉，一次捕获多次重放，消除kernel启动开销 | kernel调用频繁、CPU调度密集 |
| NPUGraph_EX | `backend="npugraph_ex"` | ACLGraph图下沉 + FX图优化 + 编译缓存复用 | 大模型推理部署 |
| AOT_Eager | `backend="aot_eager"` | 不做优化，仅验证图捕获正确性 | 调试、基线性能对比 |
| TorchAir-GE  | `backend=torchair.get_npu_backend(...)` | 将PyTorch的FX图转换为计算图，并通过GE图引擎实现计算图编译和运行 | 大模型推理部署 |

## 参考文档

下表依据特性与使用场景，整理了PyTorch官方文档中关于[torch.compile](https://docs.pytorch.org/docs/2.14/user_guide/torch_compiler/torch.compiler.html)的相关章节，涵盖通用用法、工作机制详解及常见问题排查指南，便于快速检索。

| 参考文档 | 简介 |
| --- | --- |
| [快速入门](https://docs.pytorch.org/docs/2.14/user_guide/torch_compiler/torch.compiler_get_started.html) | 介绍torch.compile的基本用法，结合简单函数和预训练模型示例，展示如何查看编译后代码，从而深入理解算子融合原理。 |
| [Dynamo概述](https://docs.pytorch.org/docs/2.14/user_guide/torch_compiler/torch.compiler_dynamo_overview.html) | 解析Dynamo从Python字节码提取FX图并调用后端编译的流程，结合示例说明Guard校验机制及编译产物的查看方法。 |
| [torch.compile编程模型](https://docs.pytorch.org/docs/2.14/user_guide/torch_compiler/compile/programming_model.html) | 概述编译行为与细粒度控制策略，提供了图中断处理、非严格追踪、重编译优化及调试方法等核心文档的入口。 |
| [Dynamo核心概念](https://docs.pytorch.org/docs/2.14/user_guide/torch_compiler/compile/programming_model.dynamo_core_concepts.html) | 围绕Dynamo追踪、图中断、Guard及重编译阐释核心机制，结合示例说明动态形状在减少重编译中的作用。 |
| [处理图中断](https://docs.pytorch.org/docs/2.14/user_guide/torch_compiler/compile/programming_model.graph_breaks_index.html) | 介绍图中断时的编译、执行及追踪恢复流程，分析其对性能的影响，并汇总不同编译模式下的处理策略。 |
| [使用fullgraph=True识别并消除图中断](https://docs.pytorch.org/docs/2.14/user_guide/torch_compiler/compile/programming_model.fullgraph_true.html) | 介绍如何利用 fullgraph=True 定位图中断，以及通过代码重构、启用非严格追踪、自定义算子或调整编译区域来消除图中断的具体方法。 |
| [常见图中断](https://docs.pytorch.org/docs/2.14/user_guide/torch_compiler/compile/programming_model.common_graph_breaks.html) | 通过示例分析由代码错误、数据依赖操作及日志打印引发的图中断，提供相应的排查步骤与规避方案。 |
| [使用nonstrict_trace](https://docs.pytorch.org/docs/2.14/user_guide/torch_compiler/compile/programming_model.dynamo_nonstrict_trace.html) | 介绍在编译区域内使用nonstrict_trace追踪函数的方法，说明输入输出类型约束及非输入值的常量处理规则。 |
| [自定义算子](https://docs.pytorch.org/docs/2.14/user_guide/torch_compiler/compile/programming_model.custom_ops.html) | 介绍通过自定义算子封装难以追踪的函数，使编译器保留调用而不追踪内部实现，并提供Python和C++自定义算子的使用指导入口。 |
| [使用fullgraph=False](https://docs.pytorch.org/docs/2.14/user_guide/torch_compiler/compile/programming_model.fullgraph_false.html) | 介绍fullgraph=False模式下的编译策略，包括选择编译入口、禁用不适合编译的函数，以及如何通过日志排查影响性能的关键图中断。 |
| [torch.compile的应用位置](https://docs.pytorch.org/docs/2.14/user_guide/torch_compiler/compile/programming_model.where_to_apply_compile.html) | 提供在训练步骤、顶层模块及子模块上应用编译的最佳实践，说明`model.compile()`的使用方式及分布式包装模块的处理建议。 |
| [禁用编译和抑制错误](https://docs.pytorch.org/docs/2.14/user_guide/torch_compiler/compile/programming_model.compiler_disable.html) | 介绍通过torch.compiler.disable禁用局部编译的方法，说明递归调用的控制策略，以及抑制编译错误的行为与局限性。 |
| [切换error_on_graph_break](https://docs.pytorch.org/docs/2.14/user_guide/torch_compiler/compile/programming_model.error_on_graph_break.html) | 说明在fullgraph=False模式下按代码区域设置图中断报错行为的方法，涵盖嵌套设置规则及其与fullgraph=True的关系。 |
| [嵌套图中断](https://docs.pytorch.org/docs/2.14/user_guide/torch_compiler/compile/programming_model.nested_graph_breaks.html) | 结合嵌套函数示例梳理图中断的处理与追踪恢复流程，分析重复追踪的原因及嵌套调用带来的额外开销。 |
| [跳过的函数](https://docs.pytorch.org/docs/2.14/user_guide/torch_compiler/compile/programming_model.skipped_functions.html) | 介绍循环、上下文管理器和 try 语句块中导致函数跳过编译的原因，并提供修复图中断或隔离问题代码的方法。 |
| [非严格追踪编程模型](https://docs.pytorch.org/docs/2.14/user_guide/torch_compiler/compile/programming_model.non_strict_tracing_model.html) | 介绍非严格追踪的工作机制，说明纯函数约束、数据指针操作限制及常量特化行为，结合示例分析追踪结果与原函数不一致的原因。 |
| [处理重编译](https://docs.pytorch.org/docs/2.14/user_guide/torch_compiler/compile/programming_model.recompilation.html) | 介绍重编译原因的定位方法，以及通过动态形状优化、标量张量化、模块整数属性配置和缓存限制调整等优化策略，帮助减少重编译开销。 |
| [降低Guard开销](https://docs.pytorch.org/docs/2.14/user_guide/torch_compiler/compile/programming_model.reducing_guard_overhead.html) | 分析Guard校验与计算图执行前的字节码开销，给出优化方法，并说明Guard过滤和跳过校验的适用条件与正确性风险。 |
| [tlparse / TORCH_TRACE](https://docs.pytorch.org/docs/2.14/user_guide/torch_compiler/compile/programming_model.observability.html) | 讲解TORCH_TRACE日志采集、tlparse编译报告解析及TORCH_LOGS日志配置方法，帮助定位图中断、Guard校验与重编译等问题。 |
| [报告问题](https://docs.pytorch.org/docs/2.14/user_guide/torch_compiler/compile/programming_model.reporting_issues.html) | 梳理编译问题的定位与反馈流程，涵盖后端对比、版本二分排查及独立复现示例构建，帮助整理便于分析的问题报告。 |
| [PyTorch 2.0 NNModule支持](https://docs.pytorch.org/docs/2.14/user_guide/torch_compiler/torch.compiler_nn_module.html) | 说明torch.compile对nn.Module的特殊处理机制，重点列出前向、反向及state_dict相关Hook的支持情况、使用限制与变更检测配置。 |
| [torch.compile的Autograd语义差异](https://docs.pytorch.org/docs/2.14/user_guide/torch_compiler/torch.compiler_backward.html) | 介绍编译模式与动态图模式下的Autograd语义差异，重点介绍反向传播中的autocast假设及backward_pass_autocast的配置方法。 |
