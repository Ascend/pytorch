# 编译libtorch\_npu

libtorch\_npu是torch\_npu插件的C++版本，包含运行torch\_npu插件所需的头文件、库文件以及CMake配置文件。用户可以通过libtorch\_npu使用torch\_npu插件中开放的C++接口。

## 编译操作

1.  参见[安装前准备](preparing_installation.md)、[安装PyTorch](installing_PyTorch.md)完成依赖与PyTorch的安装。
2.  获取libtorch\_npu源码。

    ```
    git clone -b v2.7.1-7.3.0 https://gitcode.com/Ascend/pytorch.git
    cd pytorch
    git submodule update --init --recursive
    ```

    以v2.7.1-7.3.0为例，拉取对应Ascend Extension for PyTorch分支代码。请参见《版本说明》中的“[相关产品版本配套说明](../release_notes/related_product_version_compatibility_notes.md)”章节下载Ascend Extension for PyTorch其他版本的分支代码。

3.  执行编译生成libtorch\_npu安装包。

    ```
    python3 build_libtorch_npu.py
    ```

    编译时依赖的cmake需为3.18.0版本及以上，可参见[安装3.18.0版本cmake](installing_cmake_3-18-0.md)。

    默认编译release版本，如需debug版本，添加DEBUG=1环境变量。编译完成后，当前目录下生成libtorch\_npu目录，包含以下文件。

    -   include：生成的C++头文件。
    -   lib：生成的C++库文件。
    -   share：包含Torch\_npuConfig.cmake，用于用户编译构建时获取必要的头文件，库文件等配置文件。

## libtorch推理测试

以Ascend Extension for PyTorch源码仓v2.7.1-7.3.0分支下“pytorch/examples/libtorch\_resnet”模型为例，介绍libtorch推理的快速使用。

1.  需提前安装torch、torch\_npu、torchvision、hypothesis、expecttest以及packaging。
    -   torch与torch\_npu、torchvision安装请参见[安装PyTorch](installing_PyTorch.md)以及[安装torchvision](installing_torchvision.md)。
    -   hypothesis、expecttest、packaging安装可执行如下命令。如果使用非root用户进行安装，需要在命令后加`--user`，例如：**pip3 install expecttest --user**。

        ```
        pip3 install expecttest
        pip3 install packaging
        pip3 install hypothesis
        ```

2.  编译文件中添加NPU编译配置。

    已完成NPU适配的编译文件请参见“pytorch/examples/libtorch\_resnet/**CMakeLists.txt**”，可以直接用于编译工作。

    用户如果使用自定义CMakeLists.txt编译文件，需添加以下内容用于引用libtorch\_npu插件，以便进行后续基于NPU的编译工作。

    ```
    set(torch_npu_path path_to_libtorch_npu)         # 设置libtorch_npu的路径
    include_directories(${torch_npu_path}/include)   # 设置引用libtorch_npu的头文件路径
    link_directories(${torch_npu_path}/lib)          # 设置引用libtorch_npu的库文件路径
    
    target_link_libraries(libtorch_resnet torch_npu) # 链接torch_npu库
    ```

3.  为了使模型在NPU设备上初始化和运行，用户需在C++代码中将GPU接口修改为适配NPU的接口。当前脚本中已完成对应修改，用户可参见以下内容对实际开发的脚本进行修改。已完成NPU适配的模型代码文件请参见“pytorch/examples/libtorch\_resnet/**libtorch\_resnet.cpp**”。

    代码示例如下，引入torch\_npu头文件并设置初始化的Device，在NPU使用结束时需要调用torch\_npu::finalize\_npu\(\)释放资源，否则可能会有报错提示。

    ```Cpp
    // 使用libtorch_npu相关接口，需引用libtorch_npu的头文件
    #include<torch_npu/torch_npu.h>
    
    // 使用NPU设备前需进行初始化
    torch_npu::init_npu("npu:0");
    
    // 通过传NPU字符串构造NPU设备
    auto device = at::Device("npu:0");
    
    // 使用NPU设备结束需进行反初始化
    torch_npu::finalize_npu();
    ```

    **表 1**  C++接口说明

    |接口|说明|
    |--|--|
    |torch_npu::init_npu()|使用NPU设备前需进行初始化，输入值格式为npu:*id*，其中id为NPU卡号。|
    |at::Device()|通过传NPU字符串构造NPU设备，输入值格式为npu:*id*，其中id为NPU卡号。|
    |torch_npu::finalize_npu()|使用NPU设备结束需进行反初始化，输入值格式为npu:*id*，其中id为NPU卡号。|


4.  执行编译并推理。

    “pytorch/examples/libtorch\_resnet/**resnet\_trace.py**”脚本用于导出torchscript文件，可用于libtorch推理。

    编译与推理脚本可参见“pytorch/ci/**libtorch\_resnet.sh**”，当前提供的脚本已集成了导出torchscript文件、编译与推理部分，执行如下命令编译并推理：

    ```
    bash libtorch_resnet.sh
    ```

    打印以下内容，表示编译成功。

    **图 1**  命令回显  
    ![](figures/command_output.png "命令回显")

    > [!NOTE]  
    > aarch64环境下报torch.libs/\*.so库不存在，请参见[torch.libs/libopenblasp-r0-56e95da7.3.24.so不存在](torch-libs-libopenblasp-r0-56e95da7-3-24-so_not_exist.md)。

