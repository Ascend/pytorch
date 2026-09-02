# Host级TaskQueue流水线并行

## 简介

在模型规模较大、算子调用密集的场景下（如Transformer类大模型推理/训练），Host端算子下发耗时显著，导致NPU设备利用率下降。TaskQueue是TorchNPU为加速算子下发而设计的软件机制，通过将算子下发流程拆分为主线程与算子下发线程，利用队列进行数据传递，实现算子准备工作与aclnn host API调用并行执行，从而抵消部分下发耗时并降低CPU icache miss。在实际业务场景中，开启TaskQueue均可带来性能提升（典型场景下发延迟降低5%-20%）。采用[场景三算子异步下发流程](#场景三算子异步下发流程task_queue_enable2)可进一步提升下发性能，但需关注独立内存池导致的峰值内存上涨。

默认情况下，同一设备内所有线程和Stream共享一个队列；不同设备或不同进程的队列相互独立。TaskQueue主要用于以下场景：

- **训练/推理时**：Host端算子下发对耗时敏感，需抵消aclnn host API调用耗时。
- **算子密集时**：模型中算子数量大、单算子耗时短（如大模型逐element-wise算子），Host端易成为性能瓶颈。
- **缓存优化时**：需要降低CPU icache miss带来的性能损耗。

## TaskQueue使用场景

### 场景一：算子串行下发流程（TASK_QUEUE_ENABLE=0）

![经典算子下发流程](taskqueue1.png)

该场景下，算子下发流程通过PyTorch函数式调用逐层执行并返回，所有调用均在一个线程内完成，行为与GPU类似。该线程通常为：

- **推理应用**：Python业务进程的主线程，或用户手动创建的Python线程。
- **训练应用**：正向阶段为Python业务进程的主线程，反向阶段为PyTorch内部创建的反向线程。

### 场景二：算子异步下发流程（TASK_QUEUE_ENABLE=1，默认）

![场景二算子下发流程](taskqueue2.png)

将部分负载（主要是aclnn的Host API调用，含runtime kernel的实际launch）迁移至算子下发线程。通过队列在两个线程间传递数据，以流水线方式实现时间重叠。

> [!NOTE]
> 
> workspace内存的申请仍在主线程，使用当前流的内存池，与算子输入/输出内存池相同。

### 场景三：算子异步下发流程（TASK_QUEUE_ENABLE=2）

![场景三算子下发流程](taskqueue3.png)

在场景二的基础上，将workspace内存的计算与申请下放至算子下发线程。

由于算子输入/输出内存在主线程申请，而workspace内存在算子下发线程申请，若共用同一内存池，workspace可能错误复用同一算子的输入或已释放内存，导致内存冲突。因此该模式需采用独立的workspace内存池设计，非TorchNPU内置的自定义算子不应使用此模式。

---

## TaskQueue使用指导

模式1：用户可直接使用默认设置，或者手动设置开启。

```shell
# 开启模式1
export TASK_QUEUE_ENABLE=1
```

该模式相比关闭（`TASK_QUEUE_ENABLE=0`），几乎所有实际业务场景均可获得性能提升。该设置默认对所有内置算子生效；非内置算子需参考[算子开发人员接入指南](taskqueue_op_developer.md)完成适配，适配后同样受`TASK_QUEUE_ENABLE`控制。

模式2：算子下发对延迟/耗时等敏感时，可手动设置开启。

```shell
# 开启模式2
export TASK_QUEUE_ENABLE=2
```

该模式相比模式1可进一步提升性能，由于采用独立的内存池设计，若部分算子使用较大workspace内存，模式2相比模式1可能出现峰值内存上涨的情况。建议仅在模型稳定训练后、剩余内存充裕的场景下尝试。

> [!NOTE]
> 
> 模式2与NPUGraph（aclgraph）不兼容，无法同时开启。

## 常见问题排查

### 调用栈不准确

TaskQueue开启模式1或模式2时，异步操作会导致报错发生位置与主线程的调用点解耦：当Host API在算子下发线程执行时发生报错，主线程获取的Python调用栈可能无法准确反映错误发生位置，导致调用栈信息不准确的情况。

| 错误发生位置 | GPU报错堆栈是否准确 | NPU报错堆栈是否准确（模式1） |
|------|------|------|
| Python、Aten | 是 | 是 |
| Host API | 是 | 否 |
| device kernel | 否 | 否 |

#### 排查方法

- **Host API层报错**（典型为`aclnnXxx`执行报错）：关闭TaskQueue以获取准确调用栈。

```shell
# 关闭TaskQueue
export TASK_QUEUE_ENABLE=0
```

- **device kernel层报错**：开启强制同步模式（关闭TaskQueue+device synchronize）以明确调用栈。

```shell
# 开启强制同步模式
export ASCEND_LAUNCH_BLOCKING=1
```

### 队列清空

队列清空是指CPU侧将缓存在队列中的所有算子完成下发的动作（不一定完成device执行），属于高消耗操作，对Host性能影响较大。

常见Python接口按是否触发队列清空分为两类，完整列表见[队列清空接口列表](queue_clear_interface_list.md)。简要归类如下：

- **触发清空**：设备/流同步、流对象隐式转换（`npu_stream`、`__hash__`、`__repr__`等）、Event时间统计、内存池配置、`tensor.item`、`empty_cache`、dump结束、算子超时设置、aclgraph图捕获开始等。
- **未触发清空**：设备查询与设置、流设置、`current_stream`/`default_stream`获取NPUStream对象、`Event.record`、`Event.query`、`Event.synchronize`、aclgraph图捕获结束等。

> [!NOTE]
>
> 对延迟敏感的关键路径，应避免频繁调用会触发队列清空的接口。若必须获取`aclrtStream`，应使用`torch.npu._C._npu_getCurrentRawStreamNoWait`或参考[流与处理队列](taskqueue_op_developer.md#流与处理队列)中的`stream(false)`方案。
