# `torch.compile`的Autograd语义差异

将torch.compile应用于模型前向传播中的函数时，它会自动为已编译函数生成反向传播。在编译期间，它会追踪出一个反向传播图，并在每次调用Autograd时使用该图。torch.compile中负责此工作的组件称为AOTDispatcher（有时也称为AOTAutograd）。

因此，在前向传播函数的编译期间，`torch.compile`会将计算细节固化到追踪得到的反向图中。而在eager模式PyTorch中，反向计算是动态的：在前向传播之外，可以使用上下文管理器包装`tensor.backward()`或`torch.autograd.grad(...)`调用，从而改变其行为。

本文介绍torch.compile与PyTorch Eager模式在Autograd语义上的差异，并提供了相应的解决方案。

## `Autocast`行为

torch.compile会假设反向传播是否处于外围Autocast上下文管理器中。可通过torch._functorch.config.backward_pass_autocast配置该假设；若假设与实际不符，可能导致隐蔽的计算错误。

> [!NOTICE]
>
> AMP建议仅使用[`torch.autocast`](https://docs.pytorch.org/docs/2.13/amp.html#torch.autocast)包装前向传播和损失计算。不建议在Autocast下执行反向传播；请参考[Autocasting](https://docs.pytorch.org/docs/2.13/amp.html#autocasting)中的[`torch.autocast`](https://docs.pytorch.org/docs/2.13/amp.html#torch.autocast)指南。编译器默认设置`"same_as_forward"`会假设编译后的反向传播与编译后的前向传播在相同的Autocast上下文中运行，从而有意保留现有`torch.compile`行为。如果代码遵循AMP建议，在Autocast之外执行反向传播，请为编译区域将`torch._functorch.config.backward_pass_autocast`设置为`"off"`。

可选值如下：

- `"same_as_forward"`（默认值）。假设`torch.compile`编译区域的反向传播在与该区域运行时相同的Autocast上下文管理器中运行（如果存在）。适用于以下代码：

  ```python
  with torch.amp.autocast(...):
      y = torch.compile(region)(x)
      ...
      # 反向传播与编译区域在相同的Autocast上下文中运行
      z.backward()
  ```

- `"off"`。假设torch.compile编译区域的反向传播不在任何Autocast上下文管理器中运行。适用于以下代码：

  ```python
  with torch.amp.autocast(...):
      y = torch.compile(region)(x)
      ...
  # 反向传播不在Autocast下运行。
  z.backward()
  ```

- 还有第三个选项。如果将`torch._functorch.config.backward_pass_autocast`设置为kwargs列表，则假设反向传播在由这些kwargs构造的Autocast上下文中运行。

  例如，如果代码如下：

  ```python
  y = torch.compile(region)(x)
  ...
  # 反向传播在特殊上下文管理器中运行
  with torch.amp.autocast(**kwargs):
      z.backward()
  ```

  则设置`torch._functorch.config.backward_pass_autocast = kwargs`。

使用`patch`将选项应用于特定的`torch.compile`调用：

```python
with torch.amp.autocast(...):
    with torch._functorch.config.patch(backward_pass_autocast="same_as_forward")
    y = torch.compile(region)(x)
    ...
    # 反向传播与编译区域在相同的Autocast上下文中运行
    z.backward()
```
