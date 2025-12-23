# 适配开发

基于C++ extensions方式，通过torch\_npu来调用单算子API的适配开发过程。

## 前提条件

完成CANN软件的安装具体请参见《[CANN 软件安装指南](https://www.hiascend.com/document/detail/zh/canncommercial/83RC1/softwareinst/instg/instg_0000.html?Mode=PmIns&InstallType=local&OS=openEuler&Software=cannToolKit)》（商用版）或《[CANN 软件安装指南](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/83RC1/softwareinst/instg/instg_0000.html?Mode=PmIns&InstallType=local&OS=openEuler&Software=cannToolKit)》（社区版），完成PyTorch框架的安装具体请参见《[Ascend Extension for PyTorch 软件安装指南](https://www.hiascend.com/document/detail/zh/Pytorch/720/configandinstg/instg/insg_0001.html)》。

## 适配文件结构

```
├── build_and_run.sh                // 自定义算子wheel包编译安装并执行用例的脚本
├── csrc                            // 算子适配层c++代码目录
│   ├── add_custom.cpp              // 自定义算子正反向适配代码以及绑定
│   ├── function.h                  // 正反向接口头文件
│   ├── pytorch_npu_helper.hpp      // 自定义算子调用和下发框架，用户无需关注
│   └── registration.cpp            // 自定义算子Aten IR注册文件
├── custom_ops                      // 自定义算子包python侧代码
│   ├── add_custom.py               // 提供自定义算子python调用接口
│   └── __init__.py                 // python初始化文件
├── setup.py                        // wheel包编译文件
└── test                            // 测试用例目录
    ├── test_add_custom_graph.py    // 执行torch.compile模式下用例脚本（算子入图）
    └── test_add_custom.py          // 执行eager模式下算子用例脚本
    
```

## 操作步骤

1.  在算子适配层c++代码目录（csrc）中完成C++侧算子代码适配。<a id="li92751732161014"></a>
    1.  在registration.cpp中，注册自定义算子schema。

        PyTorch提供TORCH\_LIBRARY宏来定义新的命名空间，并在该命名空间里注册schema。注意命名空间的名字必须是唯一的。具体示例如下：

        ```cpp
        // registration.cpp
        #include <torch/library.h>
        TORCH_LIBRARY(myops, m) {
            m.def("add_custom(Tensor self, Tensor other) -> Tensor");       // 注册正向接口
            m.def("add_custom_backward(Tensor self) -> (Tensor, Tensor)");  // 注册反向接口
        }
        ```

    2.  在add\_custom.cpp完成自定义算子前向和反向的适配以及绑定，主要用于创建输出内存，调用底层算子进行计算，获取返回结果并输出，为[1.a](#li92751732161014)中注册的接口提供了具体实现。
        1.  完成NPU设备的前向和反向代码实现以及绑定，具体示例如下：

            ```cpp
            // add_custom.cpp
            #include <torch/library.h>
            #include <torch/csrc/autograd/custom_function.h>
            #include "pytorch_npu_helper.hpp"
            using torch::autograd::Function;
            using torch::autograd::AutogradContext;
            using variable_list = std::vector<at::Tensor>;
            // 前向实现
            at::Tensor add_custom_impl_npu(const at::Tensor& self, const at::Tensor& other) {
                // 创建输出内存
                at::Tensor result = at::empty_like(self);
                // 调用aclnn接口计算
                EXEC_NPU_CMD(aclnnAddCustom, self, other, result);
                return result;
            }
            // 反向实现
            std::tuple<at::Tensor, at::Tensor> add_custom_backward_impl_npu(const at::Tensor& grad) {
                at::Tensor result = grad; // 创建输出内存
                return {result, result};
            }
            // 通过继承torch::autograd::Function类实现前向和反向绑定
            class AddCustomFunction : public torch::autograd::Function<AddCustomFunction> {
                public:
                    static at::Tensor forward(AutogradContext *ctx, at::Tensor self, at::Tensor other) {
                        at::AutoDispatchBelowADInplaceOrView guard;
                        static auto op = torch::Dispatcher::singleton()
                                        .findSchemaOrThrow("myops::add_custom", "")
                                        .typed<decltype(add_custom_impl_npu)>();
                        auto result = op.call(self, other);
                        return result;
                    }
                    static variable_list backward(AutogradContext *ctx, variable_list grad_outputs) {
                        auto grad_output = grad_outputs[0];
                        static auto op = torch::Dispatcher::singleton()
                                      .findSchemaOrThrow("myops::add_custom_backward", "")
                                      .typed<decltype(add_custom_backward_impl_npu)>();
                        auto result = op.call(grad_output);
                        return {std::get<0>(result), std::get<1>(result)};
                    }
            };
            // 使用的时候调用apply()方法
            at::Tensor add_custom_autograd(const at::Tensor& self, const at::Tensor& other) {
                return AddCustomFunction::apply(self, other);
            }
            // 为NPU设备注册前向和反向实现
            // NPU设备在pytorch 2.1及以上版本使用的设备名称是PrivateUse1，在2.1以下版本用的是XLA，如果是2.1以下版本PrivateUse1需要改成XLA
            TORCH_LIBRARY_IMPL(myops, PrivateUse1, m) {
                m.impl("add_custom", &add_custom_impl_npu);
                m.impl("add_custom_backward", &add_custom_backward_impl_npu);
            }
            // 给op绑定NPU的自动求导实现
            // 如果是pytorch 2.1以下的版本，AutogradPrivateUse1需要改成AutogradXLA
            TORCH_LIBRARY_IMPL(myops, AutogradPrivateUse1, m) {
                m.impl("add_custom", &add_custom_autograd);
            }
            ```

        2.  仅注册的自定义算子需要入图使用时，为Meta设备注册并完成前向和反向实现，具体示例如下：<a id="li118541236103118"></a>

            ```cpp
            // add_custom.cpp
            #include <torch/library.h>
            // 为Meta设备注册前向实现
            at::Tensor add_custom_impl_meta(const at::Tensor& self, const at::Tensor& other) {
                return at::empty_like(self);
            }
            // 为Meta设备注册反向实现
            std::tuple<at::Tensor, at::Tensor> add_custom_backward_impl_meta(const at::Tensor& self) {
                auto result = at::empty_like(self);
                return std::make_tuple(result, result);
            }
            // 为Meta设备注册前向和反向实现
            TORCH_LIBRARY_IMPL(myops, Meta, m) {
                m.impl("add_custom", &add_custom_impl_meta);
                m.impl("add_custom_backward", &add_custom_backward_impl_meta);
            }
            ```

    3.  <a id="li19878192404815"></a>
        完成前向和反向适配以及绑定以后，在function.h正反向接口头文件中声明c++自动求导接口（例如：add\_custom\_autograd），并在registration.cpp自定义算子Aten IR注册文件中将C++自动求导的接口（例如：add\_custom\_autograd）通过pybind绑定一个python接口（例如：add\_custom），以便在python代码里调用。  

        function.h正反向接口头文件：

        ```cpp
        // function.h
        #ifndef FUNCTION_H_
        #define FUNCTION_H_
        #include <ATen/ATen.h>
        at::Tensor add_custom_autograd(const at::Tensor& self, const at::Tensor& other);
        #endif //  FUNCTION_H_
        ```

        registration.cpp自定义算子Aten IR注册文件：

        ```cpp
        // registration.cpp
        #include <torch/extension.h>
        #include "function.h"
        
        // 通过pybind将c++接口和python接口绑定
        PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
            m.def("add_custom", &add_custom_autograd, "x + y"); // 其中add_custom为python侧调用的名称
        }
        ```

2.  python侧自定义算子代码适配。
    1.  在add\_custom.py文件中，新增python调用接口，此接口用户可自定义（例如：npu\_add\_custom）。

        > [!NOTE]   
        > custom\_ops\_lib.**add\_custom**中**add\_custom**需和[1.c](#li19878192404815)中python接口（例如：add\_custom）保持一致。

        ```Python
        # add_custom.py
        import torch
        import custom_ops_lib
        def npu_add_custom(self, other):
            return custom_ops_lib.add_custom(self, other)
        ```

    2.  在\_\_init\_\_.py文件中，导入add\_custom.py中新增的python调用接口。

        ```Python
        # __init__.py
         
        from .add_custom import *
        ```

3.  通过执行setup.py脚本，会将c++文件编译成so，并将其和python文件一起打包制作成wheel包。生成的wheel包在dist目录下。

    ```
    python3 setup.py build bdist_wheel
    ```

4.  自定义算子入图。

    自定义算子入图为NPU上图模式特有，入图依赖[1.b.ii](#li118541236103118)中meta注册以及自定义算子开发产生的REG\_OP算子原型。

    若AddCustom的REG\_OP原型为示例如下：

    ```
    REG_OP(AddCustom)
      .INPUT(x, ge::TensorType::ALL())
      .INPUT(y, ge::TensorType::ALL())
      .OUTPUT(z, ge::TensorType::ALL())
      .OP_END_FACTORY_REG(AddCustom);
    ```

    则注册自定义算子converter示例如下：

    ```Python
    # test_add_custom_graph.py
    from torchair import register_fx_node_ge_converter
    from torchair.ge import Tensor
    # 注意命名空间和schema名称需要与前面注册保持一致
    @register_fx_node_ge_converter(torch.ops.myops.add_custom.default)
    def convert_npu_add_custom(x: Tensor, y: Tensor, z: Tensor = None, meta_outputs: Any = None):
        return torchair.ge.custom_op(
            "AddCustom",     # 和REG_OP中算子原型名称保持一致，例如AddCustom
            inputs={
                "x": x,      # 和REG_OP中INPUT保持一致
                "y": y,      # 和REG_OP中INPUT保持一致
            },
            outputs=['z']    # 和REG_OP中OUTPUT保持一致
    )
    ```

