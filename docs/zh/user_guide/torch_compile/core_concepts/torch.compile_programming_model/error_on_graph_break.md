# 切换`error_on_graph_break`

**概要：**

- 当`fullgraph=False`时，可以使用`torch._dynamo.error_on_graph_break()`更灵活地处理图断裂。

目前已经介绍了`torch.compile`处理图断裂的两种方式：

1. `fullgraph=True`在遇到第一个图断裂时报错，并保证只从代码中追踪出一个图。
2. `fullgraph=False`即使遇到图断裂也会继续追踪。

如果希望大部分代码不允许图断裂，但少数存在问题的函数很难消除图断裂，而且可以接受这些图断裂，可以使用`torch._dynamo.error_on_graph_break()`。

`torch.compile`具有`error_on_graph_break`设置（初始值为`False`）。当其为`False`时，如果代码中发生图断裂或编译器错误，`torch.compile`会尝试在图断裂/错误之后继续编译。如果设为`True`，`torch.compile`会中止编译，并将错误传播到用户代码。

`error_on_graph_break=True`与`fullgraph=True`的一个重要区别是，前者**不保证只捕获一个图**。使用`torch._dynamo.error_on_graph_break()`上下文管理器/装饰器，可以**在编译期间任意切换**`error_on_graph_break`。相比之下，`fullgraph`一旦设为`True`，就无法再改回`False`。此外，`error_on_graph_break`的优先级低于`fullgraph`，只有在`fullgraph=False`时才会生效。

## `error_on_graph_break(False)`示例

```python
@torch._dynamo.error_on_graph_break(False)
def code_with_a_difficult_graph_break(x):
    x = x + 1
    torch._dynamo.graph_break()
    return x + 2

def inner(x):
    return code_with_a_difficult_graph_break(x)

# 注意：fullgraph=False
@torch._dynamo.error_on_graph_break(True)
@torch.compile
def fn(x):
    return inner(x)

# 不报错，但存在图断裂
fn(torch.randn(3))
```

如果希望尽量减少图断裂（即遵循`fullgraph=True`编程模型），但部分代码包含不影响性能且难以绕过的图断裂，在`error_on_graph_break(True)`区域内使用`error_on_graph_break(False)`会很有帮助。

`error_on_graph_break()`也可以用作上下文管理器：

```python
# 注意：fullgraph=False
@torch._dynamo.error_on_graph_break(True)
@torch.compile
def fn(x):
    x = x + 1
    with torch._dynamo.error_on_graph_break(False):
        torch._dynamo.graph_break()  # 不报错
    return x + 2

# 不报错，但存在图断裂
fn(torch.randn(3))
```

对于无法编辑源代码的代码（例如框架代码），可以使用monkey patch切换`error_on_graph_break`：

```python
class ThirdPartyModule(torch.nn.Module):
    def forward(self, x):
        x = x + 1
        torch._dynamo.graph_break()
        return x + 2

tp_mod = ThirdPartyModule()
tp_mod.forward = torch._dynamo.error_on_graph_break(False)(tp_mod.forward)

@torch._dynamo.error_on_graph_break(True)
@torch.compile
def fn(x):
    return tp_mod.forward(x)

# 不报错，但存在图断裂
fn(torch.randn(3))
```

## `error_on_graph_break(True)`示例

```python
@torch._dynamo.error_on_graph_break(True)
def inner2(x):
    x = x + 1
    torch._dynamo.graph_break()  # 报错
    return x + 2

def inner(x):
    return inner2(x)

# fullgraph=False，error_on_graph_break=False
@torch.compile
def fn(x):
    x = x + 4
    torch._dynamo.graph_break()  # 不报错
    return inner(x)

try:
    fn(torch.randn(3))
except Exception as e:
    print(e)
```

如果希望灵活使用`torch.compile`（即遵循`fullgraph=False`编程模型），但部分代码对性能非常关键，需要确保其中不包含图断裂，在`error_on_graph_break(False)`区域内使用`error_on_graph_break(True)`会很有帮助。

## `error_on_graph_break`的嵌套行为

`torch._dynamo.error_on_graph_break()`也会影响嵌套调用的`error_on_graph_break`设置：

```python
def inner(x):
    x = x + 1
    torch._dynamo.graph_break()
    return x + 2

def inner2(x):
    with torch._dynamo.error_on_graph_break(False):
        return inner(x)

@torch._dynamo.error_on_graph_break(True)
@torch.compile
def fn(x):
    return inner2(x)

# 不报错
fn(torch.randn(3))
```

可以在另一个`torch._dynamo.error_on_graph_break()`区域内使用`torch._dynamo.error_on_graph_break()`：

```python
def inner(x):
    x = x + 1
    with torch._dynamo.error_on_graph_break(False):
        torch._dynamo.graph_break()
    return x + 2

def inner2(x):
    with torch._dynamo.error_on_graph_break(True):
        return inner(x)

@torch.compile
def fn(x):
    return inner2(x)

# 不报错
fn(torch.randn(3))
```

## 与`fullgraph`的交互

`fullgraph=True`的优先级高于`error_on_graph_break`：

```python
@torch._dynamo.error_on_graph_break(False)
def inner(x):
    x = x + 1
    torch._dynamo.graph_break()
    return x + 2

@torch.compile(fullgraph=True)
def fn(x):
    return inner(x)

try:
    fn(torch.randn(3))
except Exception as e:
    print(e)
```

`fullgraph=True`无法切换回`fullgraph=False`：

```python
@torch.compile(fullgraph=False)
def inner(x):
    x = x + 1
    torch._dynamo.graph_break()
    return x + 2

@torch.compile(fullgraph=True)
def fn(x):
    return inner(x)

try:
    fn(torch.randn(3))
except Exception as e:
    print(e)
```

```python
@torch.compile(fullgraph=True)
def inner(x):
    x = x + 1
    torch._dynamo.graph_break()
    return x + 2

@torch.compile(fullgraph=False)
def fn(x):
    return inner(x)

try:
    fn(torch.randn(3))
except Exception as e:
    print(e)
```

## `fullgraph=True/False`与`error_on_graph_break`汇总

下表汇总了`fullgraph=True/False`与`error_on_graph_break`的区别：

| | `error_on_graph_break=True` | `error_on_graph_break=False`（默认值） |
| --- | --- | --- |
| `fullgraph=True` | 图断裂会导致错误。只报告第一个图断裂。**保证只有一个图。**<br><br>`fullgraph`无法切换为`False`。`error_on_graph_break`不起作用。<br><br>用户代码必须与`torch.compile`完全兼容。保证不会因图断裂造成性能损失（因为不存在图断裂）。<br><br>适用于对图断裂敏感的代码：框架/库代码，或要求获得最高性能的场景。可防止下游用户代码意外允许图断裂。 | 与`fullgraph=True`且`error_on_graph_break=True`相同，因为在`fullgraph=True`时`error_on_graph_break`不起作用。 |
| `fullgraph=False`（默认值） | 图断裂会导致错误。只报告第一个图断裂。**不保证只有一个图。**<br><br>`error_on_graph_break`可以切换为`False`。<br><br>用户代码必须与`torch.compile`完全兼容。保证不会因图断裂造成性能损失（因为不存在图断裂）。<br><br>适用于对图断裂敏感的用户代码。可以将`error_on_graph_break`切换为`False`，以处理包含难以绕过的图断裂的代码部分。 | 遇到图断裂后继续编译，并报告所有图断裂。<br><br>`error_on_graph_break`可以切换为`True`。<br><br>无需对用户代码进行大量修改即可工作。图断裂可能对性能产生负面影响。<br><br>适用于开箱即用的场景、“非特殊”代码，或不要求最大限度提升性能的场景。 |
