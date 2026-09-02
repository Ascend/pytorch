# 嵌套图断裂

概要：

- 嵌套函数中的图断裂可能导致难以理解的编译器行为，本文将对此进行说明。
- 一个嵌套图断裂会导致`O(N)`次重复图断裂行为。

回顾一下，将`torch.compile`应用于函数时，也会追踪所有嵌套函数调用。**嵌套图断裂**是指嵌套函数调用中发生的任何图断裂。

```python
def inner(x):
    ...
    torch._dynamo.graph_break()  # 嵌套图断裂
    ...

@torch.compile
def outer(x):
    ...
    y = inner(x)
    ...
```

嵌套图断裂周围的恢复语义可能令人困惑，下面对其行为进行说明。

回顾一下，在`fullgraph=False`模式下，[图断裂的处理方式](dynamo_core_concepts.md#图断裂)是：编译目前已确定的FX图，以常规Python方式运行不受支持的代码，然后在该代码之后使用新的FX图恢复追踪。恢复函数实际上是一项相当复杂的技术，因此仅支持在顶层函数中恢复追踪。

在此限制下，可以按如下方式在嵌套图断裂后恢复追踪。

首先考虑以下示例。`torch.compile`从`f`开始追踪，直到遇到`inner1`中的图断裂。

```python
def inner1(x):
    x = x + 1
    torch._dynamo.graph_break()  # 因图断裂停止追踪
    return x + 2

def inner2(x):
    x = x + 4
    x = inner1(x)
    x = x + 8

@torch.compile
def f(x):
    # 从此处开始追踪
    x = x + 16
    x = inner2(x)
    x = x + 32

f(torch.randn(3))
```

由于只能从顶层函数恢复，因此会在`f`中调用`inner2`的位置发生图断裂。

```python
# torch.compile(f)(x)的语义大致如下：
def compiled_f_semantics(x):
    y = x + 16
    z = inner2(y)
    return torch.compile(resume_f_semantics)(z)

def resume_f_semantics(x):
    return x + 32

compiled_f_semantics(torch.randn(3))
```

随后，`inner2`会自动作为顶层函数编译。系统再次持续追踪，直到遇到`inner1`中的图断裂。

```python
def inner1(x):
    x = x + 1
    torch._dynamo.graph_break()  # 因图断裂停止追踪
    return x + 2

# 自动应用此torch.compile
@torch.compile
def inner2(x):
    # 从此处开始追踪
    x = x + 4
    x = inner1(x)
    x = x + 8

def compiled_f_semantics(x):
    y = x + 16
    z = inner2(y)
    return torch.compile(resume_f_semantics)(z)

def resume_f_semantics(x):
    return x + 32

compiled_f_semantics(torch.randn(3))
```

然后，在`inner2`中调用`inner1`的位置发生图断裂。

```python
def compiled_inner2_semantics(x):
    y = x + 4
    z = inner1(y)
    return torch.compile(resume_inner2_semantics)(z)

def resume_inner2_semantics(x):
    return x + 8
```

随后，`inner1`会自动作为顶层函数编译。图断裂来自`inner1`，因此按常规方式处理。

```python
# 自动应用此torch.compile
@torch.compile
def inner1(x):
    # 从此处开始追踪
    x = x + 1
    torch._dynamo.graph_break()  # 因图断裂停止追踪
    return x + 2

def compiled_f_semantics(x):
    y = x + 16
    z = compiled_inner2_semantics(y)
    return torch.compile(resume_f_semantics)(z)

def resume_f_semantics(x):
    return x + 32

def compiled_inner2_semantics(x):
    y = x + 4
    z = inner1(y)
    return torch.compile(resume_inner2_semantics)(z)

def resume_inner2_semantics(x):
    return x + 8

compiled_f_semantics(torch.randn(3))
```

`inner1`按常规方式处理：

```python
def compiled_inner1_semantics(x):
    y = x + 1
    torch._dynamo.graph_break()
    return torch.compile(resume_inner1_semantics)(y)

def resume_inner1_semantics(x):
    return x + 2
```

因此，初始代码在语义上等价于：

```python
def compiled_f_semantics(x):
    y = x + 16
    z = compiled_inner2_semantics(y)
    return torch.compile(resume_f_semantics)(z)

def resume_f_semantics(x):
    return x + 32

def compiled_inner2_semantics(x):
    y = x + 4
    z = compiled_inner1_semantics(y)
    return torch.compile(resume_inner2_semantics)(z)

def resume_inner2_semantics(x):
    return x + 8

def compiled_inner1_semantics(x):
    y = x + 1
    torch._dynamo.graph_break()
    return torch.compile(resume_inner1_semantics)(y)

def resume_inner1_semantics(x):
    return x + 2

compiled_f_semantics(torch.randn(3))
```

特别需要注意的是，系统追踪了3个顶层函数，并且对同一个图断裂追踪了3次。**这解释了使用`torch.compile`时为何可能遇到重复的图断裂。**

综上，嵌套图断裂的处理方式如下：

- 从顶层函数开始一直追踪到嵌套图断裂。
- 在顶层函数调用第二层函数的位置发生图断裂。
- 编译目前追踪到的PyTorch操作并运行编译后的图。
- 调用第二层函数，该函数会自动作为顶层函数编译。
- 在第二层函数调用之后恢复追踪。

请注意，处理此图断裂的运行时间为`O(NK)`，其中`N`是嵌套深度，`K`是从顶层函数到图断裂的指令数。最终会追踪`O(N²)`个frame，并对同一个图断裂追踪`O(N)`次。
