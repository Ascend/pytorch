# `torch.compile`的应用位置

建议将`torch.compile`应用于不会引发过多问题的最高层函数。通常是：

- 包含优化器但不包含循环的`train`或`eval`步骤；
- 顶层`nn.Module`；
- 或某些子`nn.Module`。

`torch.compile`尤其不擅长处理DDP或FSDP等分布式包装模块，因此请考虑将`torch.compile`应用于传给包装器的内部模块。

```python
# 推理
model = ...
model.compile()

for _ in range(N_ITERS):
    inp = ...
    out = model(inp)
```

```python
# 训练
model = ...
opt = torch.optim.Adam(model.parameters())

@torch.compile
def train(mod, data):
    opt.zero_grad(True)
    pred = mod(data[0])
    loss = torch.nn.CrossEntropyLoss()(pred, data[1])
    loss.backward()
    opt.step()

for _ in range(N_ITERS):
    inp = ...
    train(model, inp)
```

```python
# DistributedDataParallel
model = ...
model.compile()
model_ddp = DistributedDataParallel(model, ...)

for _ in range(N_ITERS):
    inp = ...
    out = model_ddp(inp)
```

<!-- TODO：添加特定模型领域的示例，以及compile(model)与model.compile()的对比 -->

## `compile(model)`与`model.compile()`

由于`torch.compile`与`nn.Module`实例交互时存在一些细微差别，如果希望将`nn.Module`实例作为顶层函数编译，建议使用其实例的`.compile()`方法。嵌套模块调用会被正确追踪，此时无需调用`.compile()`。

```python
# 请勿这样做
model = MyModel()
model = torch.compile(model)
model(inp)

# 请这样做
model = MyModel()
model.compile()
model(inp)

# 这种方式也可以
@torch.compile
def fn(model, inp):
    return model(inp)
model = MyModel()
fn(model, inp)
```
