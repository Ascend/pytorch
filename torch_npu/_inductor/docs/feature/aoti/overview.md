# AOTI 特性介绍

## 特性简介

AOTInductor 是 TorchInductor 的专用版本，旨在处理导出的 PyTorch 模型，对其进行优化，并生成动态链接库及其他相关产物。这些编译产物广泛应用于服务端推理部署场景，支持非Python环境下的推理执行。

主要功能包括：

- 基于AOTInductor实现模型的编译、打包
- 实现C++与Python场景下模型的加载和运行
- 继承社区AOTI基本架构，实现基于NPU依赖（Runtime、Triton-Ascend等）的图编译打包、导入导出、动态形状、图下沉叠加能力
- 支持各类NPU基础、自动融合、模板类和用户手写算子
- 支持AOTI产物运行时自定义解压路径
- 支持C++运行环境的动态形状分档（padding、split）

## 如何使用

### 基本使用方法

#### 编译模型为 AOTI 格式并且打包

要使用AOTInductor编译模型，首先需要用```torch.export.export()```将模型捕获为计算图；然后通过```torch._inductor.aoti_compile_and_package```执行编译，并将编译之后的产物打包。

```python
import os
import torch
import torch_npu
import torch_npu._inductor

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(10, 16)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(16, 1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x

with torch.no_grad():
    device = "npu"
    model = Model().to(device=device)
    example_inputs=(torch.randn(8, 10, device=device),)
    batch_dim = torch.export.Dim("batch", min=1, max=1024)
    # [Optional] Specify the first dimension of the input x as dynamic.
    exported = torch.export.export(model, example_inputs, dynamic_shapes={"x": {0: batch_dim}})
    # [Note] In this example we directly feed the exported module to aoti_compile_and_package.
    # Depending on your use case, e.g. if your training platform and inference platform
    # are different, you may choose to save the exported model using torch.export.save and
    # then load it back using torch.export.load on your inference platform to run AOT compilation.
    output_path = torch._inductor.aoti_compile_and_package(
        exported,
        # [Optional] Specify the generated shared library path. If not specified,
        # the generated artifact is stored in your system temp directory.
        package_path=os.path.join(os.getcwd(), "model.pt2"),
        # [Optional] Specify Inductor configs
        # This specific max_autotune option will turn on more extensive kernel autotuning for
        # better performance.
        inductor_configs={"max_autotune": True,},
    )

```

#### 加载并运行 AOTI 模型（Python 接口）

可以使用Python接口```torch._inductor.aoti_load_package```加载并运行AOTI模型。

```python
import os
import torch
import torch_npu
import torch_npu._inductor

device = "npu"
model = torch._inductor.aoti_load_package(os.path.join(os.getcwd(), "model.pt2"))
print(model(torch.randn(8, 10, device=device)))
```

#### 加载并运行 AOTI 模型（C++ 接口）

可以使用C++接口加载并运行AOTI模型。Inductor NPU对比GPU需要额外依赖```libtorch_npu.so```，该so由PTA的编译脚本```build_libtorch_npu.py```生成。

```C++
#include <iostream>
#include <vector>
#include <exception>
#include <chrono>

#include <torch/torch.h>
#include <torch/script.h>
#include <torch/csrc/inductor/aoti_package/model_package_loader.h>
#include <torch_npu/csrc/libs/torch_npu.h>


int main() {
    try {
        c10::InferenceMode mode;

        torch::inductor::AOTIModelPackageLoader loader("model.pt2");
        torch::inductor::AOTIModelContainerRunner* runner = loader.get_runner();

        torch::Device npu_device(torch::DeviceType::PrivateUse1);
        torch::Tensor input = torch::randn({8, 10}, torch::dtype(torch::kFloat32)).to(npu_device);
        std::vector<at::Tensor> inputs = {input};

        std::vector<torch::Tensor> outputs = runner->run(inputs);

        auto start = std::chrono::system_clock::now();
        outputs = runner->run(inputs);
        auto end = std::chrono::system_clock::now();

        std::chrono::duration<double> elapsed_seconds = end - start;
        std::cout << "Inference time: " << elapsed_seconds.count() << "s" << std::endl;

        std::cout << "Output shape: [" << outputs[0].sizes() << "]" << std::endl;
        std::cout << "Output (first 10 elements): " 
                  << at::narrow(at::flatten(outputs[0]), 0, 0, 8) << std::endl;

        torch::save(outputs, "./cpp_outputs.pt");

        torch_npu::finalize_npu();
        return 0;
    } catch (const std::exception &e) {
        std::cout << "Error: " << e.what() << std::endl;
        return -1;
    }
}
```

对于C++文件的构建，可以使用提供的```CMakeLists.txt```模板,它会将```inference.cpp```文件编译成名为```aoti_example```的可执行二进制文件。

```cmake
cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(aoti_example)

if(NOT DEFINED TORCH_PATH)
    execute_process(
        COMMAND python -c "import torch; print(torch.__path__[0])"
        OUTPUT_VARIABLE TORCH_PATH
    )
    string(STRIP ${TORCH_PATH} TORCH_PATH)
    set(TORCH_PATH "${TORCH_PATH}" CACHE STRING "path of pytorch")
endif()
message("torch install location: " ${TORCH_PATH})

list(APPEND CMAKE_PREFIX_PATH ${TORCH_PATH}/share/cmake)

find_package(Torch REQUIRED)

if(NOT DEFINED TORCH_NPU_PATH)
    execute_process(
        COMMAND python -c "import torch; import torch_npu; print(torch_npu.__path__[0])"
        OUTPUT_VARIABLE TORCH_NPU_PATH
    )
    string(STRIP ${TORCH_NPU_PATH} TORCH_NPU_PATH)
    set(TORCH_NPU_PATH "${TORCH_NPU_PATH}" CACHE STRING "path of pytorch-npu")
endif()
message("torch_npu install location: " ${TORCH_NPU_PATH})

# Should replace here by real env.
# ======================================================================================
include_directories(${TORCH_NPU_PATH}/include)
link_directories(${TORCH_PATH}/libs)
link_directories("/path/to/your/libtorch_npu/lib")
# ======================================================================================

add_executable(aoti_example inference.cpp)

target_link_libraries(aoti_example torch_npu)

message("-----${TORCH_LIBRARIES}")
target_link_libraries(aoti_example "${TORCH_LIBRARIES}")
set_property(TARGET aoti_example PROPERTY CXX_STANDARD 17)
```

其中```/path/to/your/libtorch_npu/lib```需要替换为编译的```libtorch_npu.so```路径。

## 使用约束

暂不支持叠加Catlass，仅做功能兼容支持。

## 设备支持说明

- Atlas A5 系列产品
