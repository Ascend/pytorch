# 概述

torch.compile()是PyTorch 2.0+推出的核心优化接口，通过"动态图捕获+静态图优化+高效代码生成"的方式显著加速模型训练和推理任务。Ascend Extension for PyTorch在2.6.0以上版本已支持该特性。

`torch.compile()`的工作流程是：TorchDynamo（前端编译器）将用户的eager（动态图）代码JIT编译为FX Graph中间表示，然后交由选定的后端编译器（如Inductor、NPUGraph等）进行图优化和代码生成，最终输出优化后的底层机器代码。

## 后端选择

torch.compile()支持多种编译后端，可根据场景需求选择：

| 后端 | 使能方式 | 说明 | 适用场景 |
|------|----------|------|----------|
| **Inductor**（默认） | `backend="inductor"` | 具备Triton/MLIR/DVM三种算子编译器模式，支持算子融合等图优化策略 | 迭代次数多、单步计算量中等 |
| **NPUGraphs** | `backend="npugraphs"` | 基于ACLGraph的硬件级下沉优化，一次捕获、多次复跑，彻底消除启动开销 | 输入形状固定、kernel调用频繁 |
| **NPUGraph_EX** | `backend="npugraph_ex"` | 融合ACLGraph图下沉+FX图优化+编译缓存复用，进一步加速大模型编译运行 | 大模型推理加速，与服务化框架对接 |

此外，`mode="reduce-overhead"`会启用**NPUGraph Tree**逻辑——管理多个有关联的NPUGraphs，让优化收益覆盖动态形状场景，优化段图场景多个子图的内存使用。

## 接口说明

> [!NOTICE]
>
> Inductor后端需安装最新版本的Triton-Ascend依赖包，具体可参考[Triton-Ascend说明文档](https://gitcode.com/Ascend/triton-ascend/blob/master/docs/sources/getting-started/installation.md)。<br>
> Inductor后端使用MLIR模式时需额外安装Torch-MLIR依赖包，可以在[Torch-MLIR归档地址](https://repo.oepkgs.net/ascend/pytorch/vllm/torch/)下载。

**接口原型：**

```python
def compile(model, *, fullgraph=False, dynamic=None, backend="inductor", mode=None, options=None, disable=False)
```

**参数说明：**

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| model | nn.Module | 必填 | 待编译的模型 |
| fullgraph | bool | False | 是否强制整图编译 |
| dynamic | bool | None | 是否启用动态shape编译 |
| backend | str | "inductor" | 编译后端，支持`inductor`、`npugraphs`、`npugraph_ex` |
| mode | str | None | 编译模式，支持`None`和`"reduce-overhead"` |
| options | dict | None | 编译选项，详见各后端文档 |
| disable | bool | False | 是否关闭torch.compile能力 |

该接口详情可参考原生[torch.compile](https://docs.pytorch.org/docs/stable/generated/torch.compile.html)。

## 使用样例

- Inductor后端`torch.compile(backend="inductor")`示例：

    ```Python
    import torch
    import torch_npu
    import torch.nn as nn

    # 1. 定义简单模型（如MLP）
    class SimpleMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(128, 256)
            self.fc2 = nn.Linear(256, 10)

        def forward(self, x):
            x = torch.relu(self.fc1(x))
            return self.fc2(x)

    # 2. 编译模型（核心一行）
    model = SimpleMLP().npu()
    compiled_model = torch.compile(
        model,
        backend="inductor",  # 指定后端为Inductor
        mode="reduce-overhead"  # 优化策略：降低开销
    )
    # 如果需要指定算子编译器，加上选项options={"npu_backend":"mlir"/"dvm"}

    # 3. 正常训练/推理（使用方式与原始模型完全一致）
    optimizer = torch.optim.Adam(compiled_model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    x = torch.randn(32, 128).npu()  # 输入张量（batch_size=32）
    y = torch.randint(0, 10, (32,)).npu()


    for _ in range(100):  # 迭代训练

        output = compiled_model(x)
        loss = criterion(output, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    ```

- NPUGraph后端`torch.compile(backend="npugraphs")`示例：

    ```Python
    import torch
    import torch.nn as nn

    # 1. 定义模型（同模式1，需确保输入形状固定）
    class SimpleMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(128, 256)
            self.fc2 = nn.Linear(256, 10)

        def forward(self, x):
            x = torch.relu(self.fc1(x))
            return self.fc2(x)

    # 2. 编译模型（指定npugraphs后端）
    model = SimpleMLP().npu()
    compiled_model = torch.compile(
        model,
        backend="npugraphs"  # 核心：启用NPUGraph优化
    )

    # 3. 训练/推理（必须保证输入形状固定！）
    optimizer = torch.optim.Adam(compiled_model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    # 注意：npugraphs要求每次输入形状、步长等完全一致
    fixed_input = torch.randn(32, 128).npu()  # 固定形状（32, 128）
    fixed_target = torch.randint(0, 10, (32,)).npu()

    for _ in range(1000):  # 高迭代次数场景（复跑收益更明显）
        output = compiled_model(fixed_input)
        loss = criterion(output, fixed_target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    ```

- NPUGraph_EX后端`torch.compile(backend="npugraph_ex")`示例：

    ```Python
    import torch
    import torch.nn as nn

    # 1. 定义模型
    class SimpleMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(128, 256)
            self.fc2 = nn.Linear(256, 10)

        def forward(self, x):
            x = torch.relu(self.fc1(x))
            return self.fc2(x)

    # 2. 编译模型（指定npugraph_ex后端）
    model = SimpleMLP().npu()
    compiled_model = torch.compile(
        model,
        backend="npugraph_ex"  # 核心：启用NPUGraph_EX优化
    )

    # 3. 训练/推理
    input_data = torch.randn(32, 128).npu()

    for _ in range(1000):  # 高迭代次数场景（复跑收益更明显）
        output = compiled_model(input_data)
    ```

## 约束说明

1. 优化器（optimizer）通常不入图，优化器的step()包含Python侧动态逻辑（如学习率调度、梯度累积、自适应更新规则），难以被静态图捕获。
2. torch.compile(backend="npugraphs")必须固定输入形状（捕获后无法修改batch_size、序列长度等）。torch.compile(backend="inductor")支持动态形状，但会触发重新编译（增加开销），建议尽量固定形状。
3. 使用NPUGraph(ACLGraph)时需要判断算子在replay时是否需要更新，如需更新，启用update机制。
4. 仅算子全部为NN算子时，方可使用NPUGraph(ACLGraph)。
