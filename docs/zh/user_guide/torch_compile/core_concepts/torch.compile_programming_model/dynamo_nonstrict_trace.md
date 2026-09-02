# 使用`torch._dynamo.nonstrict_trace`

**概要：**

- 使用`nonstrict_trace`可以在`torch.compile`编译区域内以非严格追踪方式追踪函数。如果Dynamo因函数内部的某些内容发生图断裂，而您确定该函数可以进行非严格追踪，可以采用此方法。

请考虑以下场景：

```python
def get_magic_num():
    # 此处显式调用graph break，用来模拟任意类型的Dynamo图断裂，
    # 例如函数以C实现，或使用了Dynamo尚不支持的Python语言特性。
    torch._dynamo.graph_break()
    return torch.tensor([42])
@torch.compile(fullgraph=True)
def func(x):
    n = get_magic_num()
    return x + n
try:
    func(torch.rand(10))
except Exception as e:
    print(e)
```

运行上述代码时，Dynamo会报错，因为用户指定了`fullgraph=True`，但Dynamo遇到了图断裂。

在这种情况下，如果仍希望保留`fullgraph=True`，通常有以下几种选择：

1. 图断裂源于Dynamo尚不支持的语言特性。此时可以重写代码，或在GitHub上提交问题。
2. 图断裂源于对C语言实现函数的调用。此时可以尝试使用自定义算子，也可以提供polyfill（Python参考实现），使Dynamo能够追踪其内部。
3. 最坏的情况是编译器内部错误。此时很可能需要在GitHub上提交问题。

除上述选项外，如果引发图断裂的函数调用满足以下要求，PyTorch还提供了`torch._dynamo.nonstrict_trace`：

- 满足[常规非严格追踪](non_strict_tracing_model.md)的要求。
- 输入和输出必须包含基本类型（例如`int`、`float`、`list`、`dict`、`torch.Tensor`），或已注册到`torch.utils._pytree`的用户自定义类型。
- 函数必须定义在`torch.compile`编译区域之外。
- 函数读取的任何非输入值都会被视为常量（例如全局Tensor），并且不会为其设置guard。

追踪对`torch._dynamo.nonstrict_trace`装饰函数的调用时，`torch.compile`会切换为[非严格追踪](non_strict_tracing_model.md)，最终FX图会包含该函数内部发生的所有相关Tensor操作。

对于上述示例，可以使用`torch._dynamo.nonstrict_trace`消除图断裂：

```python
@torch._dynamo.nonstrict_trace
def get_magic_num():
    # 此处显式调用graph break，用来模拟任意类型的Dynamo图断裂，
    # 例如函数以C实现，或使用了Dynamo尚不支持的Python语言特性。
    torch._dynamo.graph_break()
    return torch.tensor([42])
@torch.compile(fullgraph=True)
def func(x):
    n = get_magic_num()
    return x + n
print(func(torch.rand(10)))
# 没有图断裂，也没有错误。
```

也可以直接在`torch.compile`编译区域内使用：

```python
def get_magic_num():
    # 此处显式调用graph break，用来模拟任意类型的Dynamo图断裂，
    # 例如函数以C实现，或使用了Dynamo尚不支持的Python语言特性。
    torch._dynamo.graph_break()
    return torch.tensor([42])
@torch.compile(fullgraph=True)
def func(x):
    n = torch._dynamo.nonstrict_trace(get_magic_num)()
    return x + n
print(func(torch.rand(10)))
# 没有图断裂，也没有错误。
```
