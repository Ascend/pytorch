# 常见图断裂

下面列出了一些常见图断裂及其解决方法。

## 错误代码

代码本身可能存在错误（即使不使用`torch.compile`也无法执行）。以下示例中的`torch.sin`调用因多传入一个参数而存在拼写错误。**请始终禁用`torch.compile`，检查代码能否正确运行。**

```python
@torch.compile
def fn(x):
    y = torch.sin(x, x)
    return y

try:
    fn(torch.ones(3, 3))
except Exception as e:
    pass
```

Dynamo会尽力提示图断裂是否由代码本身导致。但有时仍难以从日志判断图断裂究竟源于代码错误、更复杂的图断裂，还是`torch.compile`缺陷。为加以区分，建议在不使用`torch.compile`的情况下运行代码，检查图断裂报告的错误是否仍然出现。

还可以使用`torch.compiler.set_stance("force_eager")`快速禁用`torch.compile`，无需修改`torch.compile`调用：

```python
@torch.compile
def fn(x):
    y = torch.sin(x, x)
    return y

try:
    with torch.compiler.set_stance("force_eager"):
        fn(torch.ones(3, 3))
except Exception as e:
    print(e)
```

有关使用`set_stance`进行调试的更多示例，请参考https://docs.pytorch.org/tutorials/recipes/torch_compiler_set_stance_tutorial.html#crashing-sooner。

## 依赖数据的操作

遇到依赖数据的操作时，`torch.compile`会发生图断裂，例如依赖数据的控制流（if语句、包含Tensor的循环）和直接访问Tensor数据（`.item`、`.data_ptr`）。

```python
@torch.compile
def fn(x):
    y = x.sum()
    if y > 0:
        return x + y.item()
    return x - y.item()

print(fn(torch.ones(3, 3)))
```

此类图断裂的一般解决方法是避免执行依赖数据的操作。具体方法包括：

- 如果控制流实际上并不依赖数据值，请考虑修改代码，基于常量执行控制流。

```python
# 旧代码
x = torch.randn(3, 3)
@torch.compile
def fn(y):
    if x.sum() > 0:
        return y + x
    else:
        return y - x

print(fn(torch.ones(3, 3)))
```

```python
# 新代码
x = torch.randn(3, 3)
cond = (x.sum() > 0).item()
@torch.compile
def fn(y):
    if cond:
        return y + x
    else:
        return y - x

print(fn(torch.ones(3, 3)))
```

- 使用[`torch.cond`](https://docs.pytorch.org/docs/2.13/generated/torch.cond.html)等高阶算子替代依赖数据的控制流。

```python
# 旧代码
@torch.compile
def fn(x):
    if x.sum() > 0:
        return x + 1
    return x - 1

print(fn(torch.ones(3, 3)))
```

```python
# 新代码
@torch.compile
def fn(x):
    return torch.cond(
        x.sum() > 0,
        lambda x: x + 1,
        lambda x: x - 1,
        (x,),
    )

print(fn(torch.ones(3, 3)))
```

- 如果存在`.item()`调用，请尝试设置`torch._dynamo.config.capture_scalar_outputs = True`或`TORCHDYNAMO_CAPTURE_SCALAR_OUTPUTS=1`。
- 将函数中存在问题的部分封装为自定义算子。

## 打印和日志记录

打印、日志记录或发出警告都会导致图断裂。可以尝试使用`torch._dynamo.config.reorderable_logging_functions`解决此问题。该配置会重新安排日志记录函数，使其在被追踪函数的末尾调用，从而避免图断裂。不过，例如发生mutation时，记录的内容可能有所不同。

注意：`reorderable_logging_functions`存在限制，这些函数必须返回`None`，且参数只能是Tensor、常量或格式字符串。

如果无需运行打印或日志记录函数，请考虑使用`torch.compiler.is_compiling()`或`torch._dynamo.config.ignore_logging_functions`完全跳过该函数。有关详细信息，请参考[此页面](fullgraph_true.md#策略3不编译该代码)。
