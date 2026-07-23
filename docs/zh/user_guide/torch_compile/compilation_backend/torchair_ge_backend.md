# TorchAir-GE后端

## 简介

TorchAir-GE后端将PyTorch的FX计算图转换为中间表示（IR, Intermediate Representation），并通过GE（Graph Engine, 图引擎）实现计算图的编译和执行。

TorchAir-GE后端的核心优势：

- **FX图优化**：在PyTorch FX图层面进行亲和NPU的图优化，减少冗余计算和内存访问。
- **编译缓存复用**：支持编译结果的缓存复用，避免重复编译带来的开销。

## 使用场景

TorchAir-GE后端适用于大模型推理场景，通过图优化和缓存复用进一步加速编译运行，并与主流服务化框架快速对接。

## 开启方法

```python
config = torchair.CompilerConfig()
npu_backend = torchair.get_npu_backend(compiler_config=config)
compiled_model = torch.compile(model, backend=npu_backend)
```

TorchAir-GE后端支持的编译选项（`compiler_config`参数）和详细使用指导请参考《TorchAir》中的 [GE图模式](https://gitcode.com/Ascend/torchair/blob/26.1.0/docs/zh/ascend_ir/quick_start.md)。

## 调用样例

```Python
import torch
import torch_npu
import torchair

# 1. 定义模型
class SimpleMLP(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(128, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

# 2. 创建TorchAir-GE 后端
config = torchair.CompilerConfig()
npu_backend = torchair.get_npu_backend(compiler_config=config)
    
# 3. 编译模型（使用 TorchAir-GE 后端）
model = SimpleMLP().npu()
compiled_model = torch.compile(model, backend=npu_backend)

# 4. 推理
input_data = torch.randn(32, 128).npu()

for _ in range(1000):
    output = compiled_model(input_data)
```

## 约束说明

1. 使用TorchAir-GE后端前，确保模型已在昇腾NPU单算子模式（Eager）下正确执行。
2. 脚本中须先`import torch_npu`，再`import torchair`才能正常使用。
