# 报告问题

如果提供的解决方法仍不足以让`torch.compile`正常工作，可以考虑向PyTorch报告问题。不过，您可以先执行以下操作，这将显著降低我们的排查难度。

## 消融分析

使用`torch.compile`的`backend=`选项检查`torch.compile`技术栈中导致问题的组件。具体而言，请尝试：

- `torch.compile(fn, backend="eager")`：只运行TorchDynamo，即`torch.compile`的图捕获组件。
- `torch.compile(fn, backend="aot_eager")`：运行TorchDynamo和AOTAutograd，后者还会在编译期间生成反向图。
- `torch.compile(fn, backend="aot_eager_decomp_partition")`：运行TorchDynamo和带有算子分解/分区的AOTAutograd。
- `torch.compile(fn, backend="inductor")`：运行TorchDynamo、AOTAutograd和TorchInductor；TorchInductor是生成已编译kernel的后端ML编译器。

如果只有Inductor后端失败，还可以测试多种Inductor模式：

- `torch.compile(fn, backend="inductor", mode="default")`
- `torch.compile(fn, backend="inductor", mode="reduce-overhead")`
- `torch.compile(fn, backend="inductor", mode="max-autotune")`

还可以检查动态形状是否会导致任一后端出现问题：

- `torch.compile(fn, dynamic=True)`（始终使用动态形状）
- `torch.compile(fn, dynamic=False)`（从不使用动态形状）
- `torch.compile(fn, dynamic=None)`（自动动态形状）

## 二分定位

是否尝试过最新的nightly版本？某项功能过去可以正常工作，但现在不再工作了吗？能否通过二分定位确定最早出现问题的nightly版本？对于性能、精度或编译时间回退等无法立即判断问题来源的情况，二分定位尤其有用。

## 创建复现用例

创建复现用例需要大量工作，如果没有时间也完全可以理解。但是，如果您是一位积极的用户，却不熟悉`torch.compile`的内部实现，创建独立复现用例将极大提高我们修复缺陷的能力。如果没有复现用例，缺陷报告必须包含足够的信息，使我们能够确定问题根因并从头编写复现用例。

以下按优先级从高到低列出了有用的复现用例：

1. **独立的小型复现用例：**不含外部依赖、代码不超过100行且运行后能够复现问题的脚本。
2. **独立的大型复现用例：**即使代码很长，能够独立运行也是一项巨大优势！
3. **依赖可控的非独立复现用例：**例如，如果执行`pip install transformers`后运行脚本即可复现问题，这种依赖是可控的。我们很可能可以运行并调查该用例。
4. **需要大量设置的非独立复现用例：**这可能涉及下载数据集、多个环境设置步骤，或需要Docker镜像的特定系统库版本。设置越复杂，我们重建环境的难度越大。

> [!NOTE]
>
> Docker简化了设置，却增加了更改环境的难度，因此并非完美的解决方案；不过必要时我们仍会使用它。

如有可能，请尽量让复现用例采用单进程，因为单进程复现用例比多进程复现用例更易调试。

此外，下面列出了一些应在问题中检查并尝试在复现用例中重现的方面，但并非详尽列表：

- **Autograd**。Tensor输入是否设置了`requires_grad=True`？是否对输出调用了`backward()`？
- **动态形状**。是否设置了`dynamic=True`？或者是否使用不同形状多次运行测试代码？
- **自定义算子**。实际工作流中是否涉及自定义算子？能否使用Python自定义算子API重现它的一些重要特征？
- **配置**。是否设置了完全相同的配置？这包括`torch._dynamo.config`和`torch._inductor.config`设置，以及`backend`/`mode`等`torch.compile`参数。
- **上下文管理器**。是否重现了所有处于活动状态的上下文管理器？例如`torch.no_grad`、自动混合精度、`TorchFunctionMode`/`TorchDispatchMode`、激活检查点和编译Autograd等。
- **Tensor子类**。是否涉及Tensor子类？
