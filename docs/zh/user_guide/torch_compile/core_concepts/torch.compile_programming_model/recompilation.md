# 处理重编译

重编译是保证`torch.compile`可靠性所必需的，但可能显著增加编译时间。因此，在保持可靠性的同时尽量减少重编译，对于缩短编译时间至关重要。

可以使用tlparse或`TORCH_LOGS=recompiles`查看重编译及其原因。

## 启用动态形状

以下示例会因形状不匹配而重新编译：

```python
@torch.compile
def fn(x):
    return x + 1
fn(torch.ones(3))
fn(torch.ones(4))
```

请确保`torch.compile`的dynamic选项未设为`False`。默认选项`dynamic=None`只会在首次编译后尝试动态形状。可以设置`dynamic=True`，从一开始就尽可能按动态方式编译：

```python
@torch.compile(dynamic=True)
def gn(x):
    return x + 1
gn(torch.ones(3))
gn(torch.ones(4))
```

有关动态形状的更多信息，包括处理动态形状导致的错误/重编译，请参考[动态形状手册](https://docs.google.com/document/d/1GgvOe7C8_NVOMLOCwDaYV1mXXyHMXY7ExoewHqooxrs/edit?tab=t.0#heading=h.fh8zzonyw8ng)。

## 使用Tensor包装常量

默认情况下，`int`/`float`变量会被视为常量，并对其精确值设置guard。在以下示例中，每次函数调用都会发生重编译。

```python
@torch.compile
def fn(x, c):
    return x + c
for i in range(5):
    fn(torch.ones(i), 0.5 + i)
```

特别是对于学习率调度器，使用常量进行初始化可能导致重编译：

```python
mod = torch.nn.Linear(3, 3)
opt = torch.optim.Adam(mod.parameters(), lr=0.01)
sched = torch.optim.lr_scheduler.ExponentialLR(opt, 0.9)
@torch.compile
def gn(inp):
    opt.zero_grad(True)
    out = mod(inp).sum()
    out.backward()
    opt.step()
    sched.step()
for i in range(5):
    gn(torch.ones(3, 3))
```

在这两个示例中，都可以使用Tensor包装`float`变量以防止重编译。

```python
# 第一个示例
for i in range(5):
    fn(torch.ones(i), torch.tensor(0.5 + i))
# 第二个示例
opt = torch.optim.Adam(mod.parameters(), lr=torch.tensor(0.01))
sched = torch.optim.lr_scheduler.ExponentialLR(opt, torch.tensor(0.9))
for i in range(5):
    gn(torch.ones(3, 3))
```

## 更改缓存大小限制

函数的重编译次数存在上限，由`torch._dynamo.config.cache_size_limit`和`torch._dynamo.config.accumulated_cache_size_limit`决定（这两个值的确切区别详见[`torch/_dynamo/cache_size.py`](https://github.com/pytorch/pytorch/blob/4ce6e6ec8890a3f6ee604c9efb3ff153825ce575/torch/_dynamo/cache_size.py#L14)）。如果达到Dynamo缓存限制，之后所有编译尝试**都会跳过该函数（以eager模式运行）**。如果guard通过，Dynamo仍会尝试在后续函数调用中使用之前编译的字节码。请注意，达到重编译限制时，**所有嵌套函数调用都会被跳过**（Dynamo会尝试使用嵌套函数之前编译的字节码）。Dynamo还会发出警告，其中包含受影响的函数和达到的限制。在以下示例中，每次函数调用都会尝试重编译。达到缓存大小限制（默认为8）后，系统会停止尝试重编译。（为演示每次都强制重编译，此处设置了`dynamic=False`。）

```python
@torch.compile(dynamic=False)
def fn(x):
    return x + 1
for i in range(1, 10):
    # 由于dynamic=False，每次都会重编译
    fn(torch.ones(i))
```

如果已知重编译次数存在合理的常数上限，可以提高缓存大小限制。如果重编译成本超过编译带来的收益，则可以考虑降低缓存大小限制。

```python
torch._dynamo.config.cache_size_limit = 16
@torch.compile(dynamic=False)
def gn(x):
    return x + 1
for i in range(1, 10):
    gn(torch.ones(i))
```

## 通过图断裂降低重编译成本

如果大型计算图反复重编译并导致编译时间过长，可以有意引入图断裂，以性能损失为代价降低重编译成本。

```python
def very_large_function(x):
    return x + 1

@torch.compile(dynamic=False)
def fn(x, c):
    y = very_large_function(x)  # 每次都重新编译
    return y + c

for i in range(1, 5):
    fn(torch.ones(3), i)

@torch.compile(dynamic=False)
def gn(x, c):
    y = very_large_function(x)  # 只编译一次
    torch._dynamo.graph_break()
    return y + c  # 每次都重新编译

for i in range(1, 5):
    gn(torch.ones(3), i)
```
