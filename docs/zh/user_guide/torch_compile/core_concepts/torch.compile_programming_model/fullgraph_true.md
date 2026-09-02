# 使用`fullgraph=True`识别并消除图断裂

使用`torch.compile(fullgraph=False)`（默认设置）是开始使用`torch.compile`的好方法：借助图断裂能力，它可以直接支持所有Python程序，并在常见场景中提供良好性能。

但是，如果希望从模型中获得更高性能，就应明确考虑需要编译哪些代码区域：

- 建议使用`torch.compile(fullgraph=True)`查找并消除代码中的图断裂。
- 如果您是库开发者（或正在测试代码能否与`torch.compile`配合使用），建议使用`torch.compile(fullgraph=True)`进行测试。

与`fullgraph=False`相比，`torch.compile(fullgraph=True)`提供更强的保证：系统始终会捕获一个FX图进行编译；如果因图断裂而无法捕获，则会报错。**也就是说，您必须解决遇到的每一个图断裂。**

解决图断裂有多种策略。

## 策略1：重写不受支持的代码，改用Dynamo支持的特性

许多图断裂错误消息会给出如何重写代码以避免图断裂的建议。如果图断裂仍难以解决，请采用下一种策略，或向[PyTorch GitHub仓库](https://github.com/pytorch/pytorch/issues)提交问题。

有关更多图断裂示例及其解决方法，请参考[常见图断裂](common_graph_breaks.md)。

示例：Dynamo不支持对作为待编译函数输入的`list_iterator`对象调用`next`。

```python
@torch.compile(fullgraph=True)
def f(xs):
    a = next(xs)
    b = next(xs)
    return a + b

xs = [torch.tensor(1.), torch.tensor(2.)]
try:
    out = f(iter(xs))
except Exception as e:
    print(e)
```

请改写已编译函数，使其接受列表作为输入。

```python
@torch.compile(fullgraph=True)
def f_rewritten(xs):
    it = iter(xs)
    a = next(it)
    b = next(it)
    return a + b

f_rewritten(xs)
```

## 策略2：纯函数始终可以通过逃生机制进行编译

**概要**：Python函数的范围极其广泛，Dynamo不可能在不发生图断裂的情况下追踪每个Python函数。对于Dynamo无法无图断裂追踪的“纯”Python函数，系统提供了以下逃生机制，以尝试追踪这些函数：

1. 对纯Triton kernel使用`custom_op`或`triton_op`。
2. 对仅使用PyTorch Tensor操作的纯函数使用`nonstrict_trace`。
3. 对所有其他纯函数使用`custom_op`。

“纯函数”具有以下属性：

- 确定性。给定相同输入，纯函数始终返回相同输出。
- 无外部副作用。纯函数不存在修改外部状态或执行I/O等外部可见的副作用。允许仅存在于函数内部的副作用（例如修改中间Tensor）。一个值得注意的例外是，通常允许对函数输入Tensor执行会产生修改的`torch.*`操作。
- 显式输入/输出。所有输入数据都必须通过函数参数传入，所有输出都必须由函数返回。

有关示例，请参考[纯函数](non_strict_tracing_model.md#纯函数)。

理论上，Dynamo可以处理各种非纯函数，但可能尚未覆盖某些特定Python语言特性。不过，纯函数始终可以通过逃生机制进行编译。

如果发生图断裂，可以将其周围的代码重构为纯函数，并使用绕过Dynamo追踪的逃生机制：

1. 如果希望函数中的Tensor操作出现在Dynamo输出图中（从而可以进行优化），请使用`torch._dynamo.nonstrict_trace`。`nonstrict_trace`会指示Dynamo使用**非严格追踪**。
2. 如果希望函数对于`torch.compile`（包括前端Dynamo和后端）保持不透明，请使用自定义算子。

请注意，并没有任何机制阻止将这些逃生机制应用于非纯函数，但**我们不提供任何可靠性保证**。

示例：如果Dynamo不支持某个可以进行非严格追踪的Python特性或API（例如该函数使用PyTorch操作），请[使用`torch._dynamo.nonstrict_trace`捕获该函数](dynamo_nonstrict_trace.md)。

```python
# Dynamo不支持此函数（因为调用了graph_break()）。
def g(x):
    y = x.sin()
    torch._dynamo.graph_break()
    z = y.sin()
    return z

@torch.compile(fullgraph=True)
def f(x):
    w = x.sin()
    return g(w)

x = torch.randn(3)
try:
    f(x)  # 图断裂：调用了torch._dynamo.graph_break()
except Exception as e:
    print(e)

@torch.compile(fullgraph=True)
def f_rewritten(x):
    w = x.sin()
    return torch._dynamo.nonstrict_trace(g)(w)
f_rewritten(x)  # 正常工作
```

示例：使用[自定义算子](custom_ops.md)创建对于`torch.compile`不透明的函数。

```python
from torch.utils.cpp_extension import load_inline

# 平方操作的C++ 源代码
cpp_source = """
torch::Tensor square_cpu(torch::Tensor input) {
    // 检查输入是否为CPU Tensor
    TORCH_CHECK(input.device().is_cpu(), "Input must be a CPU tensor");

    // 创建形状和dtype与输入相同的输出Tensor
    torch::Tensor output = torch::empty_like(input);

    // 获取数据指针
    float* input_data = input.data_ptr<float>();
    float* output_data = output.data_ptr<float>();

    // 获取元素总数
    int64_t numel = input.numel();

    // 循环计算每个元素的平方
    for (int64_t i = 0; i < numel; i++) {
        output_data[i] = input_data[i] * input_data[i];
    }

    return output;
}
"""

# 内联加载扩展
square_module = load_inline(
    name="square_cpu_kernel",
    cpp_sources=cpp_source,
    functions=["square_cpu"],
    verbose=True
)

def square(x):
    return square_module.square_cpu(x)

@torch.compile(fullgraph=True)
def f(x):
    return square(x)

try:
    f(torch.randn(3, 3))  # 图断裂
except Exception as e:
    print(e)
```

```python
# 使用torch.library.custom_op定义新的自定义算子。
# 自定义算子对于torch.compile是不透明的，
# 即torch.compile不会查看其内部。

@torch.library.custom_op("mylib::square", mutates_args=())
def square(x: torch.Tensor) -> torch.Tensor:
    return square_module.square_cpu(x)

# 使用register_fake为算子添加``FakeTensor`` kernel
@square.register_fake
def _(x):
    return x.new_empty(x.size())

print(f(torch.randn(3, 3)))  # 无图断裂
```

有关将`triton_op`用于自定义Triton kernel的更多信息，请参考[用户定义的Triton kernel教程](https://docs.pytorch.org/tutorials/recipes/torch_compile_user_defined_triton_kernel_tutorial.html)。

## 策略3：不编译该代码

并非所有代码都适合编译。`torch.compile`是用于Tensor计算的编译器，无法优化磁盘I/O等操作。请尝试重构代码，使编译区域内不调用不受支持的代码。

```python
@torch.compile(fullgraph=True)
def f(x):
    y = x ** 2  / 2
    torch.save(y, "foo.pt")
    z = y ** 3 / 6
    return z

x = torch.randn(3)
try:
    f(x)  # 图断裂：不支持torch.save
except Exception as e:
    print(e)
```

```python
def f_rewritten(x):
    y = g(x)
    torch.save(y, "foo.pt")
    z = h(y)
    return z

@torch.compile(fullgraph=True)
def g(x):
    y = x ** 2  / 2
    return y

@torch.compile(fullgraph=True)
def h(y):
    z = y ** 3 / 6
    return z

f_rewritten(x)
```

如果存在不需要在编译模式下运行的问题函数，请考虑使用`torch.compiler.is_compiling()`跳过该函数。

```python
@torch.compile(fullgraph=True)
def f(x):
    y = x ** 2  / 2
    if not torch.compiler.is_compiling():
        torch.save(y, "foo.pt")
    z = y ** 3 / 6
    return z

x = torch.randn(3)
f(x)  # 不调用torch.save
```

如果某个函数会在许多位置调用，并且可以接受`torch.compile`无条件跳过它，可以将其添加到`torch._dynamo.config.ignore_logging_functions`。

```python
def bad_fn(y):
    torch.save(y, "foo.pt")

torch._dynamo.config.ignore_logging_functions.add(bad_fn)

@torch.compile(fullgraph=True)
def f(x):
    y = x ** 2  / 2
    bad_fn()
    z = y ** 3 / 6
    return z

x = torch.randn(3)
f(x)  # 不调用torch.save
```

可以添加到`ignore_logging_functions`的函数类型存在一些限制。具体而言：

- 函数可以接受任意参数，但**必须**返回`None`。
- 函数应为模块级函数、`logging.Logger.<method>`（对所有`logging.Logger`实例忽略该方法），或`logger_obj.<method>`（仅对特定`logger_obj`实例忽略该方法）。

由于实现细节，其他函数可能会被忽略，也可能不会。如果希望忽略的函数未被`ignore_logging_functions`忽略，请提交问题。
