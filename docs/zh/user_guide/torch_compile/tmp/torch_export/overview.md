# torch.export

torch.export 用于把 PyTorch 模型捕获为可执行、可序列化的 `ExportedProgram`。导出后可以执行导出模块、保存为 `.pt2` 文件，也可以交给 Inductor、TorchAir 或 AOTInductor 继续编译和部署。

TorchNPU 支持 torch.export 的导出、执行、保存和加载能力。使用时导入 `torch_npu`，将模型和输入放到 NPU，API 使用方式与原生保持一致。

## 导出与编程模型

torch.export 在捕获阶段确定输入输出结构、动态维度约束和可执行的图。

- **torch.export.export**：导出模型并得到 `ExportedProgram`。TorchNPU 提供与原生 PyTorch 一致的导出能力，API 和参数与原生一致。可以阅读 [torch.export](https://docs.pytorch.org/docs/2.15/user_guide/torch_compiler/export.html) 和[torch.export 编程模型](https://docs.pytorch.org/docs/2.15/user_guide/torch_compiler/export/programming_model.html)了解用法。

下面是在 NPU 上导出并执行导出模块的最小示例：

```python
import torch
import torch_npu  # 注册 NPU 后端


class Model(torch.nn.Module):
    def forward(self, x):
        return torch.relu(x) + 1


model = Model().to("npu:0")
example_args = (torch.randn(8, 4, device="npu:0"),)
ep = torch.export.export(model, example_args)

# 导出模块可以直接在同一个 NPU 设备上执行。
output = ep.module()(*example_args)
torch.npu.synchronize()
print(output)
```

`torch_npu` 导入后会注册 NPU 后端；模型、示例输入和导出模块的输入必须位于 NPU 上。上例使用 `npu:0`，也可以按实际设备编号替换。

- **动态维度声明**：用 `Dim`、`ShapesCollection` 等接口声明维度约束；`ShapeVar`、`TensorSpec`、`ShapesSpec` 等写法需要 PyTorch 2.14 及以上版本。写法与原生一致，可以阅读[动态性说明](https://docs.pytorch.org/docs/2.15/user_guide/torch_compiler/export.html#expressing-dynamism)和 [Dim API](https://docs.pytorch.org/docs/2.15/user_guide/torch_compiler/export/api_reference.html#torch.export.dynamic_shapes.Dim)了解用法。

- **strict 与 non-strict tracing**：两种捕获方式对 Python 语义的支持范围不同。可以阅读[torch.export 编程模型](https://docs.pytorch.org/docs/2.15/user_guide/torch_compiler/export/programming_model.html#strict-vs-non-strict-tracing)了解区别和选择方式。

- **API 与 IR**：查询导出、反分解和序列化接口时使用 API 参考；理解导出图约束和 IR 层级时阅读 IR 规范。入口见[torch.export API Reference](https://docs.pytorch.org/docs/2.15/user_guide/torch_compiler/export/api_reference.html)和[torch.export IR Specification](https://docs.pytorch.org/docs/2.15/user_guide/torch_compiler/export/ir_spec.html)。

- **Draft Export**：在模型尚未完全满足导出约束时得到导出结果和问题报告，用于定位导出问题。可以阅读 [Draft Export](https://docs.pytorch.org/docs/2.15/user_guide/torch_compiler/export/draft_export.html) 进行了解。

- **Joint with descriptors**：处理联合图与描述符的场景。可以阅读 [Joint with descriptors](https://docs.pytorch.org/docs/2.15/user_guide/torch_compiler/export/joint_with_descriptors.html) 进行了解。

## 保存、加载与 PT2 Archive

`ExportedProgram` 可以保存为 `.pt2` 文件并在后续进程加载，产物内部结构由 PT2 Archive 规范定义。

- **torch.export.save 与 torch.export.load**：保存和恢复 `ExportedProgram`，API 使用方式与原生一致。可以阅读[torch.export 主文档](https://docs.pytorch.org/docs/2.15/user_guide/torch_compiler/export.html)和 [API 参考](https://docs.pytorch.org/docs/2.15/user_guide/torch_compiler/export/api_reference.html)了解接口。

- **PT2 Archive**：说明 `.pt2` 文件的目录结构、版本信息与产物组成，`torch.export.save` 生成的 ExportedProgram 包使用该格式，这类 ExportedProgram 的保存和加载在 TorchNPU 上可直接使用。AOTInductor 编译产物同样使用 PT2 Archive，但 NPU 上 AOTI 编译包的可用性非常有限，大部分场景不支持，不推荐使用；只有特定版本组合完成过 Python 加载验证，详见 [AOTInductor 与部署](#aotinductor-与部署)。可以阅读 [PT2 Archive Spec](https://docs.pytorch.org/docs/2.15/user_guide/torch_compiler/export/pt2_archive.html) 了解格式。

- **在同一 NPU 上导出、执行、保存并加载**：导入 `torch_npu` 并在同一个设备编号（例如 `npu:0`）完成上述操作即可。不同设备生成的 `.pt2` 产物不保证通用，请在目标设备上导出和加载。写法如下：

```python
# producer.py：导出、执行并保存
import torch
import torch_npu  # 注册 NPU 后端


class M(torch.nn.Module):
    def forward(self, x):
        return x * 2


device = torch.device("npu:0")
torch.npu.set_device(device)
model = M().to(device)
x = (torch.arange(64, dtype=torch.float32).reshape(8, 8) / 64).to(device)

ep = torch.export.export(model, (x,))
actual = ep.module()(x)
torch.npu.synchronize()
torch.testing.assert_close(actual.cpu(), (x * 2).cpu())
torch.export.save(ep, "model.pt2")
```

```python
# consumer.py：在新的 Python 进程中加载并执行
import torch
import torch_npu  # 注册 NPU 后端

loaded = torch.export.load("model.pt2")
x = (torch.arange(64, dtype=torch.float32).reshape(8, 8) / 64).to("npu:0")
y = loaded.module()(x)
torch.npu.synchronize()
assert y.device == x.device
assert torch.equal(y.cpu(), x.cpu() * 2)
```

## 控制流

torch.export 用结构化算子表达条件、循环和扫描，导出后以 higher-order operator 的形式保留在图中。

!!! note "支持范围"
    TorchNPU 已验证这些控制流算子可以完成 NPU eager 执行、`torch.export` 捕获、导出模块执行和 Inductor 图编译；`torch.switch` 需要 PyTorch 2.14 及以上版本。使用其他图模式后端时，需另行确认其是否支持这些高阶算子。

- **控制流算子总览**：介绍 `cond`、`switch`、`while_loop`、`scan`、`associative_scan` 和 `map` 的语义、不变量和使用限制。可以阅读 [Control Flow Operators](https://docs.pytorch.org/docs/2.15/higher_order_ops/index.html) 进行了解。

- **torch.cond**：条件分支，API 和参数与原生一致。可以阅读 [Control Flow - Cond](https://docs.pytorch.org/docs/2.15/higher_order_ops/cond.html) 了解用法。

下面是在 NPU 上导出含条件分支模型的最小示例：

```python
import torch
import torch_npu  # 注册 NPU 后端


def true_fn(x):
    return x.cos()


def false_fn(x):
    return x.sin()


class CondModel(torch.nn.Module):
    def forward(self, x):
        return torch.cond(x.shape[0] > 4, true_fn, false_fn, (x,))


model = CondModel().to("npu:0")
args = (torch.randn(4, 3, device="npu:0"),)
ep = torch.export.export(
    model, args, dynamic_shapes={"x": {0: torch.export.Dim.DYNAMIC}}
)
output = ep.module()(*args)
print(output)
```

- **torch.switch**：多分支选择，需要 PyTorch 2.14 及以上版本，使用方式与原生一致。可以阅读 [Control Flow - Switch](https://docs.pytorch.org/docs/2.15/higher_order_ops/switch.html) 了解用法。

- **torch.while_loop**：按条件循环执行，API 和参数与原生一致。可以阅读 [Control Flow - While Loop](https://docs.pytorch.org/docs/2.15/higher_order_ops/while_loop.html) 了解用法。

- **torch.scan**：带 carry 的扫描，API 和参数与原生一致。可以阅读 [Control Flow - Scan](https://docs.pytorch.org/docs/2.15/higher_order_ops/scan.html) 了解用法。

- **torch.associative_scan**：对满足结合律的函数执行扫描。`combine_mode="pointwise"` 需要 torch_npu 2.14 及以上版本。`combine_mode="generic"` 按社区 decomposition 处理。可以阅读 [Control Flow - Associative Scan](https://docs.pytorch.org/docs/2.15/higher_order_ops/associative_scan.html) 了解用法。

```python
import torch
import torch_npu  # 注册 NPU 后端
from torch._higher_order_ops.associative_scan import associative_scan


class AssociativeScanModule(torch.nn.Module):
    def forward(self, xs: torch.Tensor) -> torch.Tensor:
        def combine_fn(x, y):
            return x + y

        # pointwise 需要 torch_npu 2.14 及以上版本。
        return associative_scan(combine_fn, xs, dim=0, combine_mode="pointwise")


mod = AssociativeScanModule()
inp = torch.randn(5, 3, device="npu:0")
dim_seq = torch.export.Dim("seq", min=2)
ep = torch.export.export(mod, (inp,), dynamic_shapes={"xs": {0: dim_seq}})
print(ep)
```

- **torch.map**：沿指定维度逐切片映射，API 和参数与原生一致。可以阅读 [Control Flow - Map](https://docs.pytorch.org/docs/2.15/higher_order_ops/map.html) 了解用法。

- **TorchAir 图后端**：高阶控制流当前建议优先使用上述 Inductor 路径。选择 TorchAir 时，需要按具体算子和版本检查后端创建及 converter 支持情况。基础后端说明见[Torch.compile 概述](https://gitcode.com/Ascend/pytorch/blob/v2.14.0-26.2.0/docs/zh/user_guide/torch_compile/overview.md)。

## 图分析与转换

!!! note "支持范围"
    ExportDB、FakeTensor 和 ATen IR 图转换属于图分析与 IR 层，TorchNPU 上 API 和参数与原生一致，不需要额外替换设备参数。ExportDB 的 supported/unsupported 描述的是导出能力，不代表 NPU 后端执行支持；涉及实际 NPU 执行时，按控制流或 AOTInductor 章节确认后端和版本。

- **ExportDB**：按导出特性查找支持和不支持的示例及改写方式，用于理解原生导出行为。可以阅读 [ExportDB](https://docs.pytorch.org/docs/2.15/generated/exportdb/index.html) 了解示例。

- **FakeTensor**：在没有真实数据分配的情况下分析形状、设备和图行为。可以阅读 [FakeTensor](https://docs.pytorch.org/docs/2.15/user_guide/torch_compiler/torch.compiler_fake_tensor.html) 进行了解。

- **ATen IR 图转换**：在导出后的 ATen IR 上编写图变换。可以阅读 [Writing Graph Transformations on ATen IR](https://docs.pytorch.org/docs/2.15/user_guide/torch_compiler/torch.compiler_transformations.html) 了解写法。

## AOTInductor 与部署

AOTInductor 把导出的模型提前编译成共享库和产物包，用于后续部署。通用流程见 [AOTInductor](https://docs.pytorch.org/docs/2.15/user_guide/torch_compiler/torch.compiler_aot_inductor.html)。

!!! warning "不推荐使用 AOTInductor"
    当前不推荐使用 AOTInductor，NPU 目前对 AOTInductor 的支持有限。目前仅在 Ascend 950DT＋torch_npu 2.13.0＋triton-ascend 3.6.0＋CANN 9.2.x 的特定环境上完成验证，其他构建方式可能会出现各类错误，因此不建议用于生产或通用部署。

- **编译、打包和加载（仅上述环境）**：下面是该特定环境组合中测试通过的 Python 路径，其他环境请勿直接套用。先导入 `torch_npu._inductor` 注册 NPU AOTI override，再用同一 CPU 模型生成参考值：

```python
import copy
import importlib
import torch
import torch_npu  # 注册 NPU 后端

importlib.import_module("torch_npu._inductor")
from torch._inductor.codegen.common import device_op_overrides_dict
assert "npu" in device_op_overrides_dict


class SmallModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(4, 3)
        self.fc2 = torch.nn.Linear(3, 2)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))


torch.manual_seed(20261114)
torch.npu.manual_seed_all(20261114)
cpu_model = SmallModel().eval()
cpu_input = torch.tensor([[-1., .5, 2., 1.], [.25, -.75, 1.5, 3.]])

with torch.no_grad():
    cpu_reference = cpu_model(cpu_input)
    npu_model = copy.deepcopy(cpu_model).to("npu:0")
    exported = torch.export.export(npu_model, (cpu_input.to("npu:0"),))
    package = torch._inductor.aoti_compile_and_package(
        exported,
        package_path="npu-model.pt2",
        inductor_configs={"max_autotune": False},
    )

from torch.export.pt2_archive._package import load_pt2

torch.npu.set_device(0)
contents = load_pt2(package, device_index=-1)
name = sorted(contents.aoti_runners)[0]
runner = contents.aoti_runners[name]
actual = runner(cpu_input.to("npu:0"))
torch.npu.synchronize()
torch.testing.assert_close(actual.cpu(), cpu_reference, rtol=2e-4, atol=2e-5)
```

- **加载时指定设备**：加载 AOTI 产物时建议显式指定 `device_index`。使用默认值 `-1` 且当前设备不是 0 号卡时，可能报 `The aclrtStream is not managed by torch_npu`。指定 1 号卡时写法如下：

```python
torch.npu.set_device(1)
contents = load_pt2(package, device_index=1)
name = sorted(contents.aoti_runners)[0]
runner = contents.aoti_runners[name]
actual = runner(cpu_input.to("npu:1"))
torch.npu.synchronize()
torch.testing.assert_close(actual.cpu(), cpu_reference, rtol=2e-4, atol=2e-5)
```

## 日志与调试

- **torch._logging 与 torch._logging.set_logs**：控制编译日志输出。TorchNPU 的日志类型与原生基本一致，可以先按社区文档选择需要的日志项。可以阅读 [torch._logging](https://docs.pytorch.org/docs/2.15/logging.html) 和[torch._logging.set_logs](https://docs.pytorch.org/docs/2.15/generated/torch._logging.set_logs.html)了解参数。

- **cudagraphs 日志（NPU Graph）**：TorchNPU 的 `cudagraphs` 日志项对应 NPU Graph（ACL Graph）的捕获与回放。开启后，使用 `torch.npu.graph()` 捕获计算时会输出带 `NPUGRAPH` 的 Capture/Replay 生命周期日志；这里记录的是 NPU Graph，不是 CUDA Graph。示例：

```python
import torch
import torch_npu  # 注册 NPU 后端
from torch._logging import set_logs

# 打开 cudagraphs 日志项。
set_logs(cudagraphs=True)


@torch.compile(backend="inductor")
def f(x):
    return x.sin()


x = torch.randn(4, 4, device="npu:0")
# 使用 NPU Graph 捕获并回放，日志中会出现 NPUGRAPH Capture/Replay 事件。
with torch.npu.graph():
    y = f(x)
    torch.npu.synchronize()
print(y)
```

如果使用环境变量，可以执行 `TORCH_LOGS=cudagraphs python your_script.py`。查看日志时搜索 `NPUGRAPH`，Capture 阶段表示正在捕获图，Replay 阶段表示回放已捕获的执行序列。

- **AOTInductor Minifier**：把触发错误的模型缩小成最小复现。当前仅 torch_npu 2.13.0 发布版支持；2.14 及以上版本的 repro 运行时接口尚未适配。使用时需要在父进程和 launcher、repro 的子进程中导入 `torch_npu._inductor`，生成的 `repro.py` 重现原来的错误才算完成。通用流程见[AOTInductor Minifier](https://docs.pytorch.org/docs/2.15/user_guide/torch_compiler/torch.compiler_aot_inductor_minifier.html)。

子进程可通过独立的 `sitecustomize.py` 注入后端，不需要修改全局 Python 安装：

```python
# child-site/sitecustomize.py
import os

if os.environ.get("V7_MINIFIER_NPU") == "1":
    import torch_npu._inductor
```

```bash
V7_MINIFIER_NPU=1 PYTHONPATH=child-site python minifier_launcher.py
# launcher 生成 repro.py 后，继续运行并检查是否重现同一错误：
V7_MINIFIER_NPU=1 PYTHONPATH=child-site python repro.py
```

- **NPU 调试手段**：定位编译或加载问题时，可以用 `ASCEND_LAUNCH_BLOCKING=1` 让 kernel 同步执行，把报错定位到实际出错的 kernel；在 torch.compile 路径中，可以用 `TORCHINDUCTOR_NAN_ASSERTS=1` 在 kernel 前后检查 NaN；AOTI 输入检查可以设置 `AOTI_RUNTIME_CHECK_INPUTS=1`。

开启同步执行，让异步报错出现在实际出错的 kernel 处：

```bash
ASCEND_LAUNCH_BLOCKING=1 python train.py
```

下面的最小示例在 kernel 中产生 NaN，并通过 `TORCHINDUCTOR_NAN_ASSERTS` 触发断言。环境变量必须在导入 `torch` 之前设置：

```python
import os

os.environ["TORCHINDUCTOR_NAN_ASSERTS"] = "1"

import torch
import torch_npu  # 注册 NPU 后端


@torch.compile(backend="inductor")
def f(x):
    return torch.log(x)


x = torch.tensor([1.0, -1.0], device="npu:0")  # log(-1) 产生 NaN
try:
    f(x)
except AssertionError as error:
    print("检测到 NaN：", error)
```

执行已有 AOTI 脚本时，可以通过输入检查确认传入的 shape、dtype 和 stride 满足编译约束；不满足时会抛出 `RuntimeError`：

```bash
AOTI_RUNTIME_CHECK_INPUTS=1 python run_aoti.py
```

通用方法见[AOTInductor 调试指南](https://docs.pytorch.org/docs/2.15/user_guide/torch_compiler/torch.compiler_aot_inductor_debugging_guide.html)。

## 如何选择使用方式

1. **只需要捕获并执行模型图**：用 `torch.export.export` 导出后直接执行 `ep.module()`。
2. **需要保存后在其他进程使用**：用 `torch.export.save` 和 `torch.export.load`，并保持同一 NPU 设备编号。
3. **需要进一步编译加速**：交给 Inductor 编译；使用控制流时按具体算子选择版本：`pointwise` 需要 torch_npu 2.14 及以上版本，`torch.switch` 需要 PyTorch 2.14 及以上版本。
4. **需要提前编译和打包**：按 NPU 使用条件选择 AOTInductor 的 Python `load_pt2` 路径，并检查加载数值。
5. **导出或加载失败**：先用 Draft Export 定位导出问题，再用日志和 Minifier 缩小复现。
