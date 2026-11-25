# FXRT

## 简介

FXRT是TorchNPU提供的推理图运行时。Dynamo捕获FX图后，可以由Inductor编译算子，再把图交给FXRT执行。也可以通过`backend="fxrt"`让FXRT直接执行捕获的FX图。

使用FXRT需要安装包含该组件的`torch_npu`。从源码构建时请保留默认的FXRT构建选项。

推荐把FXRT用作Inductor的加速运行时。自定义后端只负责执行Dynamo捕获的FX Graph，不再走Inductor的算子融合和代码生成。

## 使用场景

- **Inductor加速运行时（推荐）**：Inductor完成算子融合和代码生成后，由FXRT执行host图。
- **自定义后端**：只执行Dynamo捕获的FX Graph，不进行Inductor算子融合和代码生成。

两种使用场景均支持以下执行模式：

| 模式 | 配置 | 说明 |
| - | - | - |
| 默认执行模式 | 不设置，或`TORCH_NPU_FXRT_EXECUTOR=default` | 使用FXRT默认执行器。 |
| GERT执行模式 | `TORCH_NPU_FXRT_EXECUTOR=GERT` | 使用`GERT`执行器，要求CANN版本不低于9.3.0。 |

环境变量在图编译时读取，需要在编译对象首次执行前设置。取值说明见[TORCH_NPU_FXRT_EXECUTOR](../../../api/environment_variable/inductor/TORCH_NPU_FXRT_EXECUTOR.md)。

## 开启方法

- **作为Inductor的加速运行时（推荐）**：`options={"fxrt": True}`只作用于这一次`torch.compile()`：

  ```python
  compiled_model = torch.compile(model, backend="inductor", options={"fxrt": True})
  ```

- **在作用域内启用**：退出`with`后配置恢复，作用域内已经创建的编译对象可以在作用域外首次执行：

  ```python
  import torch_npu._inductor  # 注册FXRT配置项
  from torch._inductor import config

  with config.patch({"fxrt": True}):
      compiled_model = torch.compile(model, backend="inductor")
  ```

- **为后续编译设置默认值**：当前上下文里之后创建的Inductor编译对象都会读取该值，直到再次修改配置：

  ```python
  import torch_npu._inductor  # 注册FXRT配置项
  from torch._inductor import config

  config.fxrt = True
  compiled_model = torch.compile(model, backend="inductor")
  ```

- **作为`torch.compile()`自定义后端**：`backend="fxrt"`把Dynamo捕获的FX Graph直接交给FXRT。该后端在首次被选中时加载，`import torch_npu`本身不会加载FXRT：

  ```python
  compiled_model = torch.compile(model, backend="fxrt")
  ```

## 调用样例

**样例一：作为Inductor的加速运行时**

```python
import torch
import torch.nn as nn
import torch_npu

class SimpleMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(128, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))

model = SimpleMLP().npu()
compiled_model = torch.compile(model, backend="inductor", options={"fxrt": True})

x = torch.randn(32, 128).npu()
output = compiled_model(x)
```

**样例二：作为`torch.compile()`自定义后端**

```python
import torch
import torch_npu


def model(x, y):
    return torch.relu((x + y).view(4, -1))


x = torch.randn(2, 8).npu()
y = torch.randn(2, 8).npu()

compiled_model = torch.compile(model, backend="fxrt")
with torch.no_grad():
    output = compiled_model(x, y)
print(output.shape)
```

## 约束说明

- `torch.compile()`创建编译对象时会记下当时的`fxrt`配置，之后修改配置不影响已经创建的编译对象。
- 配置为`GERT`时，要求CANN版本不低于9.3.0。合法取值只有`default`和`GERT`，其他取值会报错。
