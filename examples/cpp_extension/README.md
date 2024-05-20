# 构建自定义算子wheel包工程

## 简介

本样例介绍自定义API样例打包工程，包括如何编写自定义API，绑定正反向自动求导，编译生成wheel包。

[样例工程代码](https://gitee.com/ascend/samples/tree/master/operator/AddCustomSample/FrameworkLaunch/CppExtensions/setup)

[使用说明](https://gitee.com/ascend/samples/tree/master/operator/AddCustomSample/FrameworkLaunch#%E4%BD%BF%E7%94%A8%E7%BC%96%E8%AF%91wheel%E5%8C%85%E7%9A%84%E6%96%B9%E5%BC%8F%E8%B0%83%E7%94%A8)

> 注：使用该样例工程之前需要先编译并部署底层的自定义算子包 [编译部署自定义算子包](https://gitee.com/ascend/samples/tree/master/operator/AddCustomSample/FrameworkLaunch#%E7%BC%96%E8%AF%91%E7%AE%97%E5%AD%90%E5%B7%A5%E7%A8%8B%E9%83%A8%E7%BD%B2%E7%AE%97%E5%AD%90%E5%8C%85)

## 代码介绍

### 目录结构

```
setup
├── csrc # c++代码
│   ├── extension_add1.cpp # 自定义算子适配层
│   ├── extension_add.cpp # 自定义算子适配层
│   ├── function.h # 自定义算子函数声明
│   ├── pytorch_npu_helper.hpp # 算子下发框架
│   └── register.cpp # 自定义算子注册绑定等
├── custom_ops # python打包目录
│   ├── add_custom.py
│   └── __init__.py
├── graph # 图模式相关代码
│   ├── CMakeLists.txt
│   ├── codegen.cpp
│   ├── custom_reg_op.h
│   └── operator_reg.h
├── setup.py # 编译打包文件
└── test # 测试用例
    ├── test_add_custom_graph.py # 图模式测试用例
    └── test_add_custom.py

```

### 注册自定义算子

#### 注册自定义算子schema

首先通过TORCH_LIBRARY宏注册一个名为`myops`的命名空间，注意命名空间名字必须是唯一的。在`myops`命名空间里注册两个自定义schema，分别为正向和反向。如果需要注册多个schema，只需在同一个命名空间里继续添加即可。

```c++
// register.cpp
#include <torch/library.h>

TORCH_LIBRARY(myops, m) {
    m.def("my_op(Tensor self, Tensor other) -> Tensor");
    m.def("my_op_backward(Tensor self) -> (Tensor, Tensor)");
}
```

#### 适配层kernel编写

以add算子为例，编写自定义add的kernel，通过aclnn方式调用底层npu算子实现。

```c++
// extension_add.cpp
// 为NPU设备注册前向实现
at::Tensor my_op_impl_npu(const at::Tensor& self, const at::Tensor& other) {
    // 创建输出内存
    at::Tensor result = at::Tensor(self);

    // 调用底层aclnn接口计算
    EXEC_NPU_CMD(aclnnAddCustom, self, other, result);
    return result;
}

// 为NPU设备注册反向实现
std::tuple<at::Tensor, at::Tensor> my_op_backward_impl_npu(const at::Tensor& self) {
    // 创建输出内存
    at::Tensor result = at::Tensor(self);

    return {result, result};
}
```

#### 为NPU设备绑定前向和反向的对应实现

给新注册的schema添加了PrivateUse1设备的前反向对应实现，npu在Pytorch2.1及以上版本用的dispatch key是`PrivateUse1`，2.1以下是`XLA`。

```c++
// extension_add.cpp
TORCH_LIBRARY_IMPL(myops, PrivateUse1, m) {
    m.impl("my_op", &my_op_impl_npu);
    m.impl("my_op_backward", &my_op_backward_impl_npu);
}
```

#### 添加前反向绑定

调用my_op的时候通过注册的schema和对应的disptach key来找到对应的实现。

```c++
// 寻找注册在my_op上的不同设备的实现
at::Tensor my_op_impl(const at::Tensor& self, const at::Tensor& other) {
    static auto op = torch::Dispatcher::singleton()
        .findSchemaOrThrow("myops::my_op", "")
        .typed<decltype(my_op_impl)>();
    return op.call(self, other);
}
// 寻找注册my_op_backward上的不同设备的实现
std::tuple<at::Tensor, at::Tensor> my_op_backward_impl(const at::Tensor& self) {
    static auto op = torch::Dispatcher::singleton()
        .findSchemaOrThrow("myops::my_op_backward", "")
        .typed<decltype(my_op_backward_impl)>();
    return op.call(self);
}
```



Pytorch提供了`torch::autograd::Function`方法实现前反向绑定，在里面定义`forward`和`backward`函数，通过`apply`方法调用，同时把自动求导的方法实现注册到`AutogradPrivateUse1`，实现自动求导，如果pytorch版本是2.1以下，需要将`AutogradPrivateUse1`换成`AutogradXLA`。

```c++
class MyAddFunction : public torch::autograd::Function<MyAddFunction> {
    public:
        static at::Tensor forward(AutogradContext *ctx, at::Tensor self, at::Tensor other) {
            at::AutoDispatchBelowADInplaceOrView guard;
            return my_op_impl(self, other);
        }

        static tensor_list backward(AutogradContext *ctx, tensor_list grad_outputs) {
            auto grad_output = grad_outputs[0];
            auto result = my_op_backward_impl(grad_output);
            return {std::get<0>(result), std::get<1>(result)};
        }
};

at::Tensor my_op_impl_autograd(const at::Tensor& self, const at::Tensor& other) {
    return MyAddFunction::apply(self, other);
}

TORCH_LIBRARY_IMPL(myops, PrivateUse1, m) {
    m.impl("my_op", &my_op_impl_npu);
    m.impl("my_op_backward", &my_op_backward_impl_npu);
}

// 给自定义op绑定NPU的自动求导实现
// 如果是pytorch 2.1以下的版本，AutogradPrivateUse1需要改成AutogradXLA
TORCH_LIBRARY_IMPL(myops, AutogradPrivateUse1, m) {
    m.impl("my_op", &my_op_impl_autograd);
}
```

#### 通过pybind绑定C++和Python接口

通过pybind11提供的接口将C++侧接口和Python侧接口绑定，这样在python可以用`add_custom`调用。

```c++
// register.cpp
#include <torch/extension.h>
#include "function.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("add_custom", &my_op_impl_autograd, "x + y");
}
```

`add_custom`：python侧调用的接口名，可以自定义名称。

`my_op_impl_autograd`：c++侧的函数名，需要是上面已经定义的函数。

`"x + y"`：接口描述，无实际用途，可选参数。

### 添加python侧调用接口

c++侧代码最终会编译生成so（具体编译步骤在下面章节[编译打包工程](#编译打包工程)会介绍），这里假设生成的so名称为`custom_ops_lib`，需要在python提供对外接口以供调用。在custom_ops目录下的add_custom.py里面新增python侧接口，并在__init__.py里面import该接口。

```python
# add_custom.py
import custom_ops_lib

def add_custom(self, other):
    return custom_ops_lib.add_custom(self, other)

# __init__.py
import custom_ops_lib
from .add_custom import add_custom
```

### 图模式适配（可选）

此步骤可以让用户注册的自定义算子增加入图能力，实现图模式的功能，如不需要入图则可以跳过。

> 注：入图前需要完成前面步骤的自定义算子注册

#### 添加meta设备的实现

meta tensor用于图模式中的infershape，是一种没有具体值的tensor，只有形状，数据类型，内存结构等信息。注册方法与npu设备一样，需要实现前反向，并且注册到`Meta`这个dispatch key上。

```c++
// extension_add.cpp

// 为Meta设备注册前向实现
at::Tensor my_op_impl_meta(const at::Tensor& self, const at::Tensor& other) {
    return empty_like(self);
}

// 为Meta设备注册反向实现
std::tuple<at::Tensor, at::Tensor> my_op_backward_impl_meta(const at::Tensor& self) {
    auto result = empty_like(self);
    return std::make_tuple(result, result);
}

TORCH_LIBRARY_IMPL(myops, Meta, m) {
    m.impl("my_op", &my_op_impl_meta);
    m.impl("my_op_backward", &my_op_backward_impl_meta);
}
```

#### 编译生成GE构图API

通过自动生成脚本将算子原型REG_OP转换为GE构图API，并通过torchair仓提供的接口完成converter注册，从而使能自定义算子的入图能力。

> 注：此步骤为NPU设备上入图特有的操作。

将自定义算子的算子原型REG_OP添加到graph目录下的custom_reg_op.h文件中。算子原型来源于算子工程编译结果。

```c++
// custom_reg_op.h
#include "operator_reg.h"

namespace ge {
REG_OP(AddCustom)
    .INPUT(x, ge::TensorType::ALL())
    .INPUT(y, ge::TensorType::ALL())
    .OUTPUT(z, ge::TensorType::ALL())
    .OP_END_FACTORY_REG(AddCustom);
}

```

执行编译命令，会在graph目录下生成文件auto_generated_ge_raw_custom_ops.py，将其拷贝到自己的工程或者拷贝源码至自己的调用文件里，保证能够调用到即可。

```bash
mkdir build
cd build
cmake ..
make generate_ge_raw_custom_ops
```

通过torchair提供的装饰器@register_fx_node_ge_converter完成自定义aten ir的converter注册。

```python
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union, Callable
import torchair
from torchair.ge_concrete_graph.utils import dtype_promote
from torchair.ge_concrete_graph.fx2ge_converter import register_fx_node_ge_converter
from torchair.ge_concrete_graph.ge_graph import Tensor, TensorSpec
from auto_generated_ge_raw_custom_ops import AddCustom

@register_fx_node_ge_converter(torch.ops.myops.my_op.default)
def conveter_custom_op(
        input1: Tensor,
        input2: Tensor,
        out: Tensor = None,
        meta_outputs: Any = None):
    input1, input2 = dtype_promote(input1, input2, target_dtype=meta_outputs.dtype)
    return AddCustom(input1, input2)
```

至此完成自定义算子入图适配工作，用户可以运行参考用例中的示例验证。

### 编译打包工程

完成自定义算子注册以后，需要通过setuptools工具来同时编译c++代码，并将生成的so和custom_ops目录下的python代码一起生成wheel包。setuptools工具通过执行setup.py来编译打包，里面会指定编译的一些参数。

```python
# setup.py
import torch
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension
import torch_npu
from torch_npu.utils.cpp_extension import NpuExtension

USE_NINJA = os.getenv('USE_NINJA') == '1'
exts = []
ext1 = NpuExtension(
    name="custom_ops_lib",
    # 如果还有其他cpp文件参与编译，需要在这里添加
    sources=["./csrc/extension_add.cpp", "./csrc/extension_add1.cpp", "./csrc/register.cpp"],
    extra_compile_args = [
        '-I' + os.path.join(PYTORCH_NPU_INSTALL_PATH, "include/third_party/acl/inc"),
    ],
)
exts.append(ext1)

setup(
    name="custom_ops",
    version='1.0',
    keywords='custom_ops',
    ext_modules=exts,
    packages=find_packages(),
    cmdclass={"build_ext": BuildExtension.with_options(use_ninja=USE_NINJA)},
)
```

setup里面指定了wheel包的名称、版本号、需要编译的扩展、打包的文件、编译命令等，同时ext_modules里面指定了c++编译源文件、头文件路径、编译选项等参数。通过执行`python3 setup.py build bdist_wheel`命令会编译打包，在dist目录下会生成wheel包，通过pip install的方式可以安装该wheel包，具体使用方法参考测试用例。

