# NPUGraphs

## 简介

`npugraphs`是`torch.compile()`的图捕获后端，基于NPUGraph（ACLGraph）技术实现。NPUGraphs捕获一系列NPU操作（如kernel调用、内存拷贝）组成静态图缓存在NPU设备上；一次捕获、多次复跑，避免重复的kernel启动开销（kernel launch overhead）。

使用该后端时，Dynamo将FX Graph交由NPUGraph自动下沉为ACLGraph静态图重放优化，无需手动管理Stream和捕获流程。NPUGraphs是NPUGraph API的高级封装，自动完成图捕获和重放流程。如需更精细的控制（如手动管理Stream、分区域捕获、动态控制流下的安全子图捕获等），请参考[torch_npu.npu.NPUGraph](../../../developer_notes/npugraph.md)文档中的`NPUGraph`类、`graph()`上下文管理器和`make_graphed_callables` API。

## 使用场景

- 输入形状固定、kernel调用频繁的模型。
- 需要彻底消除启动开销的高迭代训练/推理任务。

## 开启方法

```python
compiled_model = torch.compile(model, backend="npugraphs")
```

NPUGraphs后端没有专用的`options`配置项，传入的后端选项会被忽略。`guard_filter_fn`等前端选项由Dynamo处理。

## 调用样例

以两层MLP的推理为例。输入为32条128维样本，输出形状为`(32, 10)`。

```python
import torch
import torch.nn as nn
import torch_npu

model = nn.Sequential(nn.Linear(128, 256), nn.ReLU(), nn.Linear(256, 10))
model = model.eval().to("npu")
compiled_model = torch.compile(model, backend="npugraphs")
fixed_input = torch.randn(32, 128, device="npu")

with torch.no_grad():
    expected = model(fixed_input)
    for _ in range(4):
        # 每次前向开始新一轮推理，允许图树复用上一轮输出所占用的内存。
        torch.compiler.cudagraph_mark_step_begin()
        output = compiled_model(fixed_input)
        torch.testing.assert_close(output, expected)
        del output
```

## 约束说明

- 已捕获的图按捕获时的输入形状回放。输入形状变化时，后端可能为新形状编译或捕获新图，也可能回退执行。尽量减少重复出现的形状种类。
- 只有支持图捕获的NPU算子才能入图。可用`TORCH_LOGS=cudagraphs`查看预热、捕获和回放情况。
- 使用Inductor的`mode="reduce-overhead"`前，按 [Inductor 后端](inductor.md) 中的对应编译器模式准备依赖。该模式同样需要为不同输入形状管理捕获的图。
