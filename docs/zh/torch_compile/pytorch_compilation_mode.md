# 概述

`torch.compile()`是PyTorch 2.0推出的核心编译接口，通过“动态图捕获+静态图优化+高效代码生成”的方式显著加速模型训练和推理任务。Ascend Extension for PyTorch 7.3.0版本开始支持`torch.compile()`功能，仅需一行代码即可自动实现前端图捕获和后端优化，适用于全自动编译的场景。

torch.compile包含如下核心组件：

**表 1**  核心组件

| 组件              | 作用       |
|-------------------|-----------|
| Dynamo前端        |Dynamo能够JIT（即时）将用户的eager（动态图）代码编译为FX Graph（PyTorch的中间表示）。 |
| 编译后端          |对FX Graph进行优化并生成最终可执行的代码。|

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
| dynamic | bool | None | 是否启用动态 shape 编译 |
| backend | str/Callable | `"inductor"` | 编译后端：`inductor`、`npugraphs`、`npugraph_ex`、`aot_eager`、`TorchAir-GE后端(Callable)` |
| mode | str | None | 编译模式：`None` 或 `"reduce-overhead"` （仅`inductor`后端支持）|
| options | dict | None | 编译选项|
| disable | bool | False | 关闭 torch.compile |

更多参数详情可参见 [torch.compile](https://docs.pytorch.org/docs/stable/generated/torch.compile.html)。

**编译后端说明**

| 后端 | 使能方式 | 核心机制 | 适用场景 |
|------|----------|---------|---------|
| Inductor（默认） | `backend="inductor"` | 算子融合 + 代码生成（Triton/MLIR/DVM） | 大多数场景，不确定时首选 |
| NPUGraphs | `backend="npugraphs"` | ACLGraph图下沉，一次捕获多次重放，消除kernel启动开销 | kernel调用频繁、CPU调度密集 |
| NPUGraph_EX | `backend="npugraph_ex"` | ACLGraph图下沉 + FX图优化 + 编译缓存复用 | 大模型推理部署 |
| AOT_Eager | `backend="aot_eager"` | 不做优化，仅验证图捕获正确性 | 调试、基线性能对比 |
| TorchAir-GE  | `backend=torchair.get_npu_backend(...)` | 将PyTorch的FX图转换为计算图，并通过GE图引擎实现计算图编译和运行 | 大模型推理部署 |
