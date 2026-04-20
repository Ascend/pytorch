# torch.compile api详解

## 使用指导

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
    -   [enable_shape_handling](../feature/dynamicshape/overview.md) (形状处理特性开关，详见特性介绍页面)

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
