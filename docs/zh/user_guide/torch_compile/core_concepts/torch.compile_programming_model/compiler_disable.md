# 禁用编译和抑制错误

对于某些模型架构，模型中的部分内容尤其难以编译，可能存在大量图断裂或发生崩溃。您可能希望显式禁用模型中这些存在问题的部分，从而将`torch.compile`应用于能够正常工作的部分。可以使用`@torch.compiler.disable`装饰器实现这一点。当`torch.compile`尝试调用被禁用的函数时，它会中断计算图并跳过对该函数的追踪，在调用结束后恢复追踪。默认情况下，从被禁用函数发起的所有递归调用也会被禁用。使用`recursive=False`选项可以允许编译递归调用。

```python
def inner1(x):
    torch._dynamo.graph_break()  # 不追踪
    return x + 1  # 不追踪

@torch.compiler.disable
def outer1(x):
    x = x + 2  # 不追踪
    torch._dynamo.graph_break()  # 不追踪
    return inner1(x)

@torch.compile
def f(x):
    x = outer1(x)
    return x + 4  # 追踪

print(f(torch.ones(3)))
```

```python
def inner2(x):
    torch._dynamo.graph_break()  # 追踪
    return x + 1  # 追踪

@torch.compiler.disable(recursive=False)
def outer2(x):
    x = x + 2  # 不追踪
    torch._dynamo.graph_break()  # 不追踪
    return inner2(x)

@torch.compile
def g(x):
    x = outer2(x)
    return x + 4  # 追踪

print(g(torch.ones(3)))
```

例如，可以使用`torch.compiler.disable`对推荐模型中的稀疏架构禁用`torch.compile`，因为稀疏架构难以编译。预处理函数和日志记录函数通常也会导致大量图断裂，无法从编译中获益。

如果遇到编译器崩溃但仍希望继续运行，可以设置`torch._dynamo.config.suppress_errors = True`。编译器崩溃时，系统会跳过对该函数的追踪，稍后再重试。**这不是最佳实践**，更好的做法是最终根据需要手动添加`disable`注解。
