# 非严格追踪编程模型

**概要：**

- **非严格追踪**是一种比Dynamo限制更少的Python代码追踪方式，但可能产生不易察觉的错误结果。
- 非严格追踪会运行Python函数，并利用Python和PyTorch的算子重载能力，将执行期间发生的Tensor操作记录到trace中。
- 如果函数满足一些约束，尤其是函数为**纯函数**且不直接操作`Tensor.data_ptr()`，则该函数**可以进行非严格追踪**。
- 非严格追踪可能对某些变量进行**特化**并将其视为**常量**，把变量值固化到trace中。

`torch.compile`内部组件（`make_fx`、AOTDispatcher）使用**非严格追踪**。[`torch._dynamo.nonstrict_trace`](dynamo_nonstrict_trace.md)也可以在`torch.compile`编译的代码中标记需要使用非严格追踪的代码区域。非严格追踪会运行Python函数，并利用Python和PyTorch的算子重载能力，将执行期间发生的Tensor操作记录到trace中。

**`make_fx`**是非严格追踪的主要入口。对于以下函数，使用给定输入执行时只会进入上方分支，因此捕获的图只包含该分支。

```python
from torch.fx.experimental.proxy_tensor import make_fx
def f(x):
    if x.shape[0] > 2:
        return x ** 2 / 6
    else:
        return x * 3
x = torch.randn(3)
gm = make_fx(f, tracing_mode="fake")(x)
gm.print_readable()
```

非严格追踪与Dynamo（严格）追踪的区别在于，**非严格追踪并不安全**：对于给定函数，它捕获的Tensor操作图可能与原函数具有不同语义。对于Python函数，Dynamo追踪会捕获Tensor操作图和残余字节码，二者结合后与Python函数具有相同语义。

## 纯函数

非严格追踪仅对**纯函数**可靠，因此只能对纯函数进行非严格追踪。

纯函数具有以下属性：

- **确定性。**给定相同输入，纯函数始终返回相同输出。
- **无副作用。**纯函数不存在修改外部状态或执行I/O操作等副作用。
- **显式输入/输出。**所有输入数据都必须通过函数参数传入，所有输出都必须由函数返回。

以下是一些非纯函数示例，其捕获图的行为与原函数不同。

### 示例1：无显式输入（例如访问全局Tensor）

```python
var = torch.tensor(1)
def function_with_global_access(y):
    return y + var
x = torch.tensor([0, 1, 2])
# 为了演示，需要设置 _allow_non_fake_inputs=True才能捕获全局变量。
gm = make_fx(
    function_with_global_access, tracing_mode="fake", _allow_non_fake_inputs=True
)(x)
# 非严格追踪捕获全局变量的值 (1.)
print("1. call function", function_with_global_access(x))
print("1. call graph", gm(x))
# 但是，更改全局变量后，捕获的图与原函数产生不同结果
var = torch.tensor(2)
print("2. call function", function_with_global_access(x))
print("2. call graph", gm(x))
# 要捕获可接受不同`var` Tensor的图，必须将其设为显式输入：
def function_fixed(y, var):
    return y + var
var = torch.tensor(3)
gm = make_fx(function_fixed, tracing_mode="fake")(x, var)
print("3. call function", function_fixed(x, var))
print("3. call graph", gm(x, var))
var = torch.tensor(4)
print("4. call function", function_fixed(x, var))
print("4. call graph", gm(x, var))
```

有关原因，请参考[特化和常量](#特化和常量)。

### 示例2：副作用（打印）

```python
def function_with_side_effect(y):
    print(y)
x = torch.tensor([0, 1, 2])
_ = function_with_side_effect(x)
```

在Python中运行`f`会产生打印Tensor的副作用。

```python
gm = make_fx(function_with_side_effect, tracing_mode="fake")(x)
```

进行非严格追踪时，打印会在图捕获期间发生。

```python
_ = gm(x)
```

图中不会保存对`print`语句的调用，因此执行图时不会打印任何内容。

### 示例3：副作用（修改输入列表）

```python
lst = []
def function_with_input_list_mutation(lst):
    val = lst.pop()
    return val
x = torch.tensor([0, 1, 2])
y = torch.tensor([0, 1, 2])
# 每次执行函数时，列表长度都会缩短
lst = [x, y]
function_with_input_list_mutation(lst)
print("len(lst) after one call", len(lst))
function_with_input_list_mutation(lst)
print("len(lst) after two calls", len(lst))
# 使用非严格追踪时，列表长度会在图捕获期间缩短，
# 但调用图时不会缩短。
lst = [x, y]
gm = make_fx(function_with_input_list_mutation, tracing_mode="fake")(lst)
print("len(lst) after graph capture", len(lst))
gm(lst)
print("len(lst) after one call to graph", len(lst))
gm(lst)
print("len(lst) after two calls to graph", len(lst))
```

### 不直接操作data_ptr

直接操作`Tensor.data_ptr`无法进行非严格追踪。直观原因是PyTorch无法判断您以*何种方式*操作了`data_ptr`。

```python
import ctypes
# 创建只包含一个元素的Tensor
tensor = torch.tensor([42], dtype=torch.int32)  # 为简单起见使用int32
def function_with_data_ptr(tensor):
    # 获取数据指针
    ptr = tensor.data_ptr()
    # 将指针转换为ctypes指针
    ctypes_ptr = ctypes.cast(ptr, ctypes.POINTER(ctypes.c_int32))
    # 增加指针所指向的值
    ctypes_ptr.contents.value += 1
    return tensor
try:
    make_fx(function_with_data_ptr, tracing_mode="fake")(tensor)
except Exception as e:
    print(e)
```

## 特化和常量

非严格追踪捕获的图可能针对某些值进行了特化，也就是说，捕获的图只对这些值有效。我们称该图将这些值视为**常量**。

非严格追踪期间，所有非Tensor变量都被视为常量：

```python
def f(x, y):
    return x + y
x = torch.tensor([0, 1, 2])
y = 3.14
gm = make_fx(f, tracing_mode="fake")(x, y)
gm.print_readable()
```

3.14是图中的常量。

非严格追踪也会针对输入Tensor的属性进行特化。

```python
def f(x):
    if x.shape[0] > 2:
        return x ** 2 / 6
    else:
        return x * 3
x = torch.randn(3)
gm = make_fx(f, tracing_mode="fake")(x)
gm.print_readable()
```

此外，它还会针对未直接传入函数的所有变量进行特化：

```python
var = torch.tensor(1)
def f(x):
    return x + y
x = torch.randn(3)
gm = make_fx(f, tracing_mode="fake")(x)
gm.print_readable()
```
