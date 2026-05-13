# NPUGraph

## 简介

NPUGraph是一种静态图捕获技术，核心思想与[CUDAGraphs](https://pytorch.org/blog/accelerating-pytorch-with-cuda-graphs/)一致——将一系列NPU内核定义并封装为一个单元（即操作图），通过单一CPU操作启动多个NPU操作，从而减少启动开销。其工作流程分为**捕获（Capture）、重放（Replay）、更新（Update）**三个阶段，详见[核心机制](#核心机制捕获重放与更新)章节。

### 核心优势

图重放机制通过牺牲动态图执行的灵活性，换取显著的CPU开销降低：

1. **固定计算图结构**：参数与内核在捕获阶段确定后不再变更，重放时无需重复进行参数校验、内核选择等操作。
2. **高效执行流程**：重放时直接调用底层启动接口，将整个图任务批量提交至NPU，跳过CPU端多层调度开销。

NPUGraph的优势可以通过下图展示：

![NPUGraph优势](../figures/npugraph.png)

> [!NOTE]
>
> CPU逐个启动一列短核时，CPU启动开销会在核之间造成显著间隙。而使用NPUGraph替代这串核序列，最初需要花更多时间构建图并一次性启动整个图，但后续执行会非常快，因为核间的间隙非常小。当同一操作序列重复多次（例如训练步数非常多）时，差异更为明显。构建和启动图的初始成本将在整个训练迭代次数中摊销。

## 适用场景

**推荐使用NPUGraph的场景：**

- 网络结构完全或部分为静态图（图安全）
- 存在CPU瓶颈（特别是短内核密集型任务）
- 小批量训练场景（NPU利用率低）
- 输入形状固定的推理或训练任务
- 高迭代次数的重复计算任务
- 仅支持aclnn算子入图

**NPUGraph不适用的场景：**

- 动态形状输入（每批次尺寸变化）
- 动态控制流（条件分支、循环结构可变）
- 需要频繁CPU-NPU同步的操作

## 核心机制：捕获、重放与更新

### 捕获（Capture）

将NPU流置于捕获模式后，内核调用被记录为计算图结构。NPUGraph提供两种捕获方式：

**方式一：使用NPUGraph类（底层API）**

`torch_npu.npu.NPUGraph`是底层原始类，提供对捕获流程的精细控制。使用时需手动管理Stream、调用`capture_begin()`和`capture_end()`。

```python
import torch
import torch_npu

def graph_capture_simple():
    s = torch_npu.npu.Stream()

    with torch_npu.npu.stream(s):
        a = torch.full((1000,), 1, device="npu")
        g = torch_npu.npu.NPUGraph()
        torch_npu.npu.empty_cache()
        g.capture_begin()
        b = a
        for _ in range(10):
            b = b + 1
        g.capture_end()
    torch_npu.npu.current_stream().wait_stream(s)

    g.replay()

    print(f"b.sum().item() == {b.sum().item()}.")

graph_capture_simple()
```

**方式二：使用graph上下文管理器**

`torch_npu.npu.graph`是简单通用的上下文管理器，可在其作用域内捕获NPU操作。相比手动调用`capture_begin()`和`capture_end()`，该方式更加简洁，自动处理Stream同步和缓存清理。

```python
import torch
import torch_npu

def graph_simple():
    a = torch.full((1000,), 1, device="npu")
    g = torch_npu.npu.NPUGraph()
    with torch_npu.npu.graph(g):
        b = a
        for _ in range(10):
            b = b + 1

    g.replay()

    print(f"b.sum().item() == {b.sum().item()}.")

graph_simple()
```

### 重放（Replay）

捕获完成后，通过`g.replay()`多次重放相同计算逻辑。每次重放的kernel序列、执行顺序与捕获时完全一致，无需重复的图构建和kernel启动准备。

重放时的内存管理是核心要点：捕获阶段分配的张量内存地址在重放时保持不变。若需要在不同step之间使用新数据，必须通过`copy_()`将新数据写入捕获时占用的内存地址，而不能直接对张量重新赋值（如`tensor = new_data`），否则变量将指向新内存地址，重放仍在旧地址上操作导致数据失效。该机制在`make_graphed_callables`内部实现中自动处理——对于前向和反向传播，当检测到输入张量的`data_ptr()`与捕获时不同时，自动执行`copy_()`更新。

### 更新（Update）

对于需要动态参数（如序列长度）的算子（如FlashAttention），NPUGraph通过专用的update机制在重放前刷新参数值，无需重新捕获整张图。该机制由`_npugraph_handlers`中的算子Handler自动管理，用户通过`auto_dispatch_capture=True`启用。

## make_graphed_callables：安全子图捕获

以上`NPUGraph`类和`graph()`上下文管理器属于底层API，适用于捕获整段连续计算图。当网络中存在不可捕获部分（如动态控制流、动态网络拓扑、CPU同步或关键的CPU端逻辑）时，可使用`torch_npu.npu.make_graphed_callables`高级API——与CUDA Graphs社区一致，自动处理图捕获细节和输入数据的`copy_()`更新。该API将安全部分封装为图化可调用对象，其余部分保持eager执行。

```python
import torch
import torch_npu
from torch_npu.contrib import transfer_to_npu
import torch.nn as nn
import torch.optim as optim
from itertools import chain

def main():
    # 1. 环境检查：验证NPU可用性
    if not torch.npu.is_available():
        print("# NPU不可用，请在支持NPU的环境中运行本示例")
        return
    print(f"# PyTorch版本：{torch.__version__}")
    print(f"# torch_npu版本：{torch_npu.__version__}")
    print(f"# NPU设备数量：{torch.npu.device_count()}")

    # 2. 可复现性设置
    torch.manual_seed(42)
    torch.npu.manual_seed(42)

    # 3. 模型定义与NPU迁移
    N, D_in, H, D_out = 640, 4096, 2048, 1024
    module1 = nn.Linear(D_in, H).npu()
    module2 = nn.Linear(H, D_out).npu()
    module3 = nn.Linear(H, D_out).npu()

    loss_fn = nn.MSELoss().npu()
    optimizer = optim.SGD(
        chain(module1.parameters(), module2.parameters(), module3.parameters()),
        lr=0.1
    )

    # 4. 准备捕获用静态张量（requires_grad状态需与实际输入匹配）
    x = torch.randn(N, D_in, device='npu')  # module1输入无需梯度
    h = torch.randn(N, H, device='npu', requires_grad=True)  # module2/3输入需梯度

    # 5. 使用make_graphed_callables捕获子图
    print("# 捕获NPUGraph子图")
    module1 = torch_npu.npu.make_graphed_callables(module1, (x,))
    module2 = torch_npu.npu.make_graphed_callables(module2, (h,))
    module3 = torch_npu.npu.make_graphed_callables(module3, (h,))
    print("# NPUGraph子图捕获完成")

    # 6. 准备真实训练数据
    real_inputs = [torch.randn_like(x) for _ in range(10)]
    real_targets = [torch.randn(N, D_out, device='npu') for _ in range(10)]

    # 7. 执行训练迭代（含动态分支）
    print("# 开始10次迭代（使用NPUGraphed Callables）")
    for i, (data, target) in enumerate(zip(real_inputs, real_targets)):
        optimizer.zero_grad(set_to_none=True)

        # 前向：module1无条件执行
        tmp = module1(data)  # 图化前向

        # 动态分支：根据中间结果选择module2或module3
        # ⚠️ 注意：NPUGraph要求分支结构在捕获时确定；此处分支仅影响复用哪个图，
        #        各分支内部计算图保持静态，因此安全可用
        if tmp.sum().item() > 0:
            tmp = module2(tmp)  # 图化前向
        else:
            tmp = module3(tmp)  # 图化前向

        loss = loss_fn(tmp, target)
        loss.backward()  # 对应选中模块的图化反向 + module1反向
        optimizer.step()

        if i == 0 or i == 9:
            param_sum = sum(p.sum().item() for p in chain(
                module1.parameters(), module2.parameters(), module3.parameters()))
            print(f"# 迭代{i+1}: 模型参数总和={param_sum:.6f}, 损失={loss.item():.6f}")

    print("# 所有迭代完成")
    print("# NPUGraphed Callables功能验证成功")

if __name__ == "__main__":
    main()
```
