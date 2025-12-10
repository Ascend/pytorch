# PyTorch编译模式（torch.compile）

## 简介

torch.compile\(\)是PyTorch 2.0+推出的核心优化接口，通过"动态图捕获+静态图优化+高效代码生成"的方式显著加速模型训练和推理任务。Ascend Extension for PyTorch在2.6.0以上版本已支持该特性，为用户提供两种常用的backend配置选项，分别是torch.compile\(backend="inductor"\)和torch.compile\(backend="npugraphs"\)。

torch.compile\(\)包含如下核心组件：

**表 1**  核心组件

|组件|定位|作用|
|--|--|--|
|Dynamo|前端编译器（代码转换器）|TorchDynamo能够JIT（即时）将用户的eager（动态图）代码编译为FX Graph，进而交给其他lowering编译器（如Inductor）进行编译，最终生成优化过后的底层机器代码，达到加速效果。|
|Inductor|后端编译器（高效代码生成器）|具备基于triton-ascend自动生成高性能算子的能力，能够显著减少开发者手动设计Tiling、管理内存等工作量。支持算子融合等图优化策略，通过减少内存访问次数来提升性能。|
|NPUGraph（aclgraph）|硬件级下沉优化（NPU操作录屏）|捕获一系列NPU操作（如 kernel 调用、内存拷贝）组成静态图缓存在NPU device设备上；一次捕获、多次复跑，避免重复的 kernel 启动开销（kernel launch overhead）。|
|NPUGraph Tree|动态形状路由与子图管理|管理多个有关联的 NPUGraphs，让 NPUGraph的优化收益能覆盖动态形状场景，而非仅局限于固定形状，优化段图场景多个子图的内存使用。|


## 使用场景

Inductor后端：通过torch.compile(backend="inductor")使能，以降低Python开销和kernel启动开销为核心，通过Dynamo+Inductor协同，在不改变模型逻辑的前提下，自动进行算子融合和生成，提升训练或推理的吞吐量，尤其适合迭代次数多、单步计算量中等的场景。

NPUGraph后端：通过torch.compile(backend="npugraphs")使能，利用NPUGraphs技术，彻底消除NPU任务的启动开销和CPU至NPU同步开销，适合eager模式存在host bound且kernel调用频繁但输入形状固定的场景，整体功能与backend="cudagraphs"一致。

## 使用指导

> [!NOTICE]  
> Inductor后端需安装最新版本的triton-ascend，具体可参考[LINK](https://gitcode.com/Ascend/triton-ascend/blob/master/docs/sources/getting-started/installation.md)。


接口原型：

```
def compile(model, *, fullgraph = False, dynamic = None, backend = "inductor", mode = None, options = None, disable = False)
```

参数说明：

-   **model**: 必选参数，要编译的模型或者参数。
-   **fullgraph**：可选参数，是否强制整图编译，默认值为False。
-   **dynamic**：可选参数，是否需要动态shape编译，默认值为None。
-   **backend**：可选参数，编译后端，支持inductor和npugraphs，默认值为inductor。
-   **mode**：可选参数，编译模式，目前支持None（默认值）和"reduce-overhead"。
-   **options**：可选参数，编译选项。目前支持以下：
    -   triton.cudagraphs
    -   trace.enabled
    -   enable\_shape\_handling

-   **disable**：可选参数，是否关闭torch.compile能力，默认值为False。

该接口详情可参考原生[torch.compile](https://docs.pytorch.org/docs/stable/generated/torch.compile.html)。

## 使用样例

-   Inductor后端torch.compile\(backend="inductor"\)示例：

    ```Python
    import torch
    import torch_npu
    import torch.nn as nn
    
    # 1. 定义简单模型（如 MLP）
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
        backend="inductor",  # 指定后端为 Inductor
        mode="reduce-overhead"  # 优化策略：降低开销
    )
    
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

-   NPUGraph后端torch.compile\(backend="npugraphs"\)示例：

    ```Python
    # 运行
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
    
    # 2. 编译模型（指定 npugraphs 后端）
    model = SimpleMLP().npu()
    compiled_model = torch.compile(
        model,
        backend="npugraphs"  # 核心：启用 NPUGraph 优化
    )
    
    # 3. 训练/推理（必须保证输入形状固定！）
    optimizer = torch.optim.Adam(compiled_model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    
    # 注意：npugraphs 要求每次输入形状、步长等完全一致
    fixed_input = torch.randn(32, 128).npu()  # 固定形状（32, 128）
    fixed_target = torch.randint(0, 10, (32,)).npu()
    
    for _ in range(1000):  # 高迭代次数场景（复跑收益更明显）
        output = compiled_model(fixed_input)
        loss = criterion(output, fixed_target)
    
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    ```

## 约束说明

1.  优化器（optimizer）通常不入图，优化器的step\(\)包含Python侧动态逻辑（如学习率调度、梯度累积、自适应更新规则），难以被静态图捕获。
2.  torch.compile\(backend="npugraphs"\)必须固定输入形状（捕获后无法修改 batch\_size、序列长度等）。torch.compile\(backend="inductor"\)支持动态形状，但会触发重新编译（增加开销），建议尽量固定形状。
3.  使用NPUGraph（aclgraph）时需要判断算子在replay时是否需要更新，如需更新，启用update机制。

