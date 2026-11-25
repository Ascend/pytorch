# NPUGraph_EX

## 简介

NPUGraph_EX是轻量化高性能图后端，融合了ACLGraph的图下沉调度能力，在PyTorch FX图上叠加亲和NPU的图优化和编译缓存复用等能力，进一步加速大模型在NPU上编译运行。

NPUGraph_EX后端的核心优势：

- **图下沉调度**：基于ACLGraph实现NPU操作的静态图捕获和复跑，消除kernel launch开销。
- **FX图优化**：在PyTorch FX图层面进行亲和NPU的图优化，减少冗余计算和内存访问。
- **编译缓存复用**：支持编译结果的缓存复用，避免重复编译带来的开销。
- **服务化框架对接**：与主流服务化框架快速、无缝地对接，便于大模型推理部署。

> [!NOTICE]
>
> - 使用NPUGraph_EX后端前，需安装TorchNPU插件及对应Python版本的packaging库（版本需大于等于21.3）。NPUGraph_EX的部分Python代码依赖packaging库进行版本比较，若环境中缺少该库，首次编译时会报错“ModuleNotFoundError: No module named 'packaging'”，并被PyTorch封装为“BackendCompilerFailed”。
>   请在运行模型的Python环境中执行 `python -m pip install "packaging>=21.3"`。

## 使用场景

NPUGraph_EX后端适用于大模型推理场景，通过图优化和缓存复用进一步加速编译运行，并与主流服务化框架快速对接。

## 开启方法

```python
compiled_model = torch.compile(model, backend="npugraph_ex")
```

NPUGraph_EX支持的编译选项（`options`参数）和详细使用指导请参考《TorchAir》中的 [npugraph_ex后端](https://gitcode.com/Ascend/torchair/blob/26.2.0/docs/zh/npugraph_ex/npugraph_ex.md)。

## 调用样例

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

# 2. 编译模型（指定 npugraph_ex 后端）
model = SimpleMLP().npu()
compiled_model = torch.compile(
    model,
    backend="npugraph_ex"  # 核心：启用 NPUGraph_EX 优化
)

# 3. 训练/推理
input_data = torch.randn(32, 128).npu()

for _ in range(1000):  # 高迭代次数场景（复跑收益更明显）
    output = compiled_model(input_data)
```

## 约束说明

- 优化器（optimizer）通常不入图，优化器的`step()`包含Python侧动态逻辑（如学习率调度、梯度累积、自适应更新规则），难以被静态图捕获。
- 使用NPUGraph_EX时需要判断算子在replay时是否需要更新，如需更新，启用update机制。
- 主要面向在线推理场景，暂不支持反向流程capture成图、随机数算子capture。
- 与torch.cuda.CUDAGraph原生接口的约束保持一致，例如不支持stream sync、动态控制流等。
- 建议将`fullgraph`参数设置为True。若编译器遇到无法追踪到单一计算图中的代码（即“图中断”），则会引发错误。
- 昇腾NPU暂不支持`mode`参数，功能配置请通过`options`参数传入。
- 建议先执行`torch.compile(backend="aot_eager", fullgraph=True)`验证脚本可正常入图。若该步骤失败，说明脚本或算子本身不满足`backend="npugraph_ex"`的使用条件。

## 参考文档

本页仅提供NPUGraph_EX后端在torch.compile视角下的最小示例与通用约束。`options`支持的完整功能项、各功能的专属约束与配置方式、问题定界方法及API参考，由《TorchAir》中的 "[npugraph_ex](https://gitcode.com/Ascend/torchair/blob/26.2.0/docs/zh/npugraph_ex/quick_start.md)" 章节维护，请查阅以下内容。

- [npugraph_ex快速上手](https://gitcode.com/Ascend/torchair/blob/26.2.0/docs/zh/npugraph_ex/quick_start.md)：使用约束、torch.compile参数在aclgraph模式下的说明、基础/进阶/DFX功能清单、问题定界与功能拓展。
- [基础功能](https://gitcode.com/Ascend/torchair/blob/26.2.0/docs/zh/npugraph_ex/basic/basic.md)：force_eager、force_recapture、FX图优化Pass、内存复用、静态Kernel编译、固定权重类输入地址、集合通信入图等功能及各自的约束。
- [进阶功能](https://gitcode.com/Ascend/torchair/blob/26.2.0/docs/zh/npugraph_ex/advanced/advanced.md)：模型编译缓存、多流表达、AI Core和Vector Core限核、自定义FX图优化Pass、SuperKernel等功能及各自的约束。
- [DFX功能](https://gitcode.com/Ascend/torchair/blob/26.2.0/docs/zh/npugraph_ex/dfx/dfx.md)：图编译Debug信息保存、算子Data Dump、多流并发死锁检测。
- [API参考](https://gitcode.com/Ascend/torchair/blob/26.2.0/docs/zh/npugraph_ex/api/api.md)：torch.npu.npugraph_ex、compile_fx、CompilerConfig、register_replacement及scope接口。
