# 跳过的函数

**概要：**

- 有时，`torch.compile`会完全放弃编译某个函数，改为以eager模式运行，这可能导致失去优化机会。
- 可以通过一些方法绕过被跳过的函数，在存在问题的代码周围重新启用追踪。

使用`fullgraph=False`时，`torch.compile`有时无法在遇到图断裂或其他编译器错误后恢复追踪。在许多此类情况下，`torch.compile`会完全跳过该函数的编译，并以eager模式运行。

请注意，跳过只应用于当前函数，**不会**应用于任何嵌套函数调用。`torch.compile`仍会尝试编译嵌套调用。

<!-- TODO：修复跳过函数的日志记录。 -->

```python
def inner1(x):
    return x + 1
def inner2(x):
    return x + 2
@torch.compile
def fn(x):
    x = inner1(x)
    torch._dynamo.skip_frame()
    x = inner2(x)
fn(torch.randn(3))
```

在上述示例中，`torch.compile`会追踪`fn`（包括`inner1`），直到遇到`skip_frame`。随后会跳过`fn`并以eager模式运行；调用`inner1`和`inner2`时会编译它们。

跳过函数可能导致失去优化机会，因此请务必检查希望编译的代码是否被跳过；如果被跳过，应设法绕过该行为。

## 循环中的图断裂

如果循环中发生图断裂，`torch.compile`无法恢复追踪：

```python
@torch.compile
def fn(x):
    for i in range(5):
        x = x + 1
        if i == 3:
            torch._dynamo.graph_break()
    return x
fn(torch.randn(3))
```

在此示例中，可以展开循环以避免跳过：

```python
@torch.compile
def fn(x):
    def inner(i):
        nonlocal x
        x = x + 1
        if i == 3:
            torch._dynamo.graph_break()
    inner(0)
    inner(1)
    inner(2)
    inner(3)
    inner(4)
    return x
fn(torch.randn(3))
```

通常，解决导致跳过的图断裂也会解决跳过问题。

## 上下文管理器中的图断裂

另一种常见的无法恢复的图断裂是大多数上下文管理器中的图断裂：

```python
class CustomCtxManager:
    def __enter__(self):
        pass
    def __exit__(self, exc_type, exc_value, traceback):
        pass
@torch.compile
def fn(x):
    with CustomCtxManager():
        x = x + 1
        torch._dynamo.graph_break()
        return x + 1
fn(torch.randn(3))
```

可以将图断裂移到上下文管理器之外，从而避免跳过：

```python
@torch.compile
def fn(x):
    with CustomCtxManager():
        x = x + 1
    torch._dynamo.graph_break()
    with CustomCtxManager():
        return x + 1
fn(torch.randn(3))
```

对于某些上下文管理器，Dynamo可以在图断裂后恢复追踪。其中一些可以在`torch/_dynamo/variables/torch.py`的`supported_ctx_manager_classes`中找到。通常，`torch/_dynamo/variables/ctx_manager.py`中由`ContextWrappingVariable`子类表示的任何上下文管理器都支持在图断裂后恢复。例如：

```python
import contextlib
@torch.compile
def fn(x):
    with contextlib.nullcontext():
        with torch.no_grad():
            x = x + 1
            torch._dynamo.graph_break()
            return x + 1
fn(torch.randn(3))
```

## try块中的图断裂

try块中的图断裂无法恢复：

```python
@torch.compile
def fn(x):
    try:
        x = x + 1
        torch._dynamo.graph_break()
        return x + 1
    except Exception as e:
        pass
fn(torch.randn(3))
```

可以将图断裂移到try块之外，从而避免跳过：

```python
@torch.compile
def fn(x):
    try:
        x = x + 1
    except Exception as e:
        pass
    torch._dynamo.graph_break()
    try:
        return x + 1
    except Exception as e:
        pass
fn(torch.randn(3))
```

## 达到重编译限制

请参考[更改缓存大小限制](recompilation.md#更改缓存大小限制)。

## 编译器错误

某些编译器错误会导致函数被跳过，其他编译器错误则会直接导致严重错误，而不是跳过函数。

## 处理被跳过的函数

通常，可以通过修复导致函数被跳过的底层图断裂或错误来解决被跳过函数的问题。

如果导致函数被跳过的图断裂/错误难以修复，请考虑将其隔离到单独的函数中，以尽量减少被跳过的内容。

```python
def inner1(x):
    return x + 1
def inner2(x):
    return x + 2
@torch.compile
def fn(x):
    x = inner1(x)
    def problematic_code():
        torch._dynamo.skip_frame()
    problematic_code()
    x = inner2(x)
fn(torch.randn(3))
```
