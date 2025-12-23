# 适配开发

支持Ascend C实现自定义算子Kernel，并集成在PyTorch框架，通过PyTorch的API实现算子调用。

## 前提条件

完成CANN软件的安装具体请参见《[CANN 软件安装指南](https://www.hiascend.com/document/detail/zh/canncommercial/83RC1/softwareinst/instg/instg_0000.html?Mode=PmIns&InstallType=local&OS=openEuler&Software=cannToolKit)》（商用版）或《[CANN 软件安装指南](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/83RC1/softwareinst/instg/instg_0000.html?Mode=PmIns&InstallType=local&OS=openEuler&Software=cannToolKit)》（社区版），完成PyTorch框架的安装具体请参见《[Ascend Extension for PyTorch 软件安装指南](https://www.hiascend.com/document/detail/zh/Pytorch/720/configandinstg/instg/insg_0001.html)》。

## 适配文件结构

```
├ examples
│  ├ cpp_extension
│  │   ├ op_extension          // Python脚本, 初始化模块
│  │   ├ csrc                  // C++目录
│  │   │   ├ kernel            // 算子kernel实现    
│  │   │   ├ host              // 算子注册torch
│  │   │   │   ├ tiling        // 算子tiling实现
│  │   ├ test                  // 测试用例目录
│  │   ├ CMakeLists.txt        // Cmake文件
│  │   ├ setup.py              // setup文件
│  │   ├ README.md             // 模块使用说明
```

## 操作步骤

>  [!NOTE]  
> 以下流程以add\_custom kernel算子适配为例进行说明。

1.  kernel算子实现。
    1.  在./cpp\_extension/csrc/kernel目录下完成kernel算子实现，具体可参考《[CANN Ascend C算子开发指南](https://www.hiascend.com/document/detail/zh/canncommercial/83RC1/opdevg/Ascendcopdevg/atlas_ascendc_10_0001.html)》。
    2.  在CMakeLists.txt中配置对应kernel的编译选项。

        > [!NOTE]  
        >Ascend C编写的算子，是否需要workspace具有不同编译选项。以下示例中提供了两种算子的实现：
        >-   不需要workspace：add\_custom算子。
        >-   需要workspace：matmul\_leakyrelu\_custom算子。

        -   add\_custom算子示例：

            ```cpp
            ascendc_library(no_workspace_kernel STATIC
              csrc/kernel/add_custom.cpp
            )
            ```

        -   matmul\_leakyrelu\_custom算子示例：

            ```cpp
            ascendc_library(workspace_kernel STATIC
              csrc/kernel/matmul_leakyrelu_custom.cpp
            )
            ascendc_compile_definitions(workspace_kernel PRIVATE
            -DHAVE_WORKSPACE
            -DHAVE_TILING
            )
            ```

2.  在./cpp\_extension/csrc/host目录下，封装实现的kernel算子，注册为PyTorch的API，子目录tiling存放算子的tiling函数。以下示例以自定义torch.ops.npu.my\_add API（封装add\_custom kernel算子）为例：
    1.  Aten IR定义。

        PyTorch通过TORCH\_LIBRARY宏将C++算子实现绑定到Python。Python侧可以通过torch.ops.namespace.op\_name方式进行调用。如果在相同namespace下的不同API，定义在不同文件，需要遵循PyTorch官方规则使用TORCH\_LIBRARY\_FRAGMENT。自定义my\_add API注册在npu命名空间如下：

        ```
        TORCH_LIBRARY_FRAGMENT(npu, m)
        {
          m.def("my_add(Tensor x, Tensor y) -> Tensor");
        }
        ```

        通过此注册，Python侧可通过torch.ops.npu.my\_add调用自定义的API。

    2.  Aten IR实现。
        1.  按需包含头文件。注意需要包含对应的核函数调用接口声明所在的头文件，alcrtlaunch\_\{kernel\_name\}.h（该头文件为Ascend C工程框架自动生成），kernel\_name为算子核函数的名称。
        2.  算子适配，根据Aten IR定义适配算子，包括按需实现输出Tensor申请，workspace申请，调用kernel算子等。torch\_npu的算子下发和执行是异步的，通过TASKQUEUE实现。以下示例中通过EXEC\_KERNEL\_CMD宏封装了算子的ACLRT\_LAUNCH\_KERNEL方法，将算子执行入队到torch\_npu的TASKQUEUE。

            ```cpp
            #include "utils.h"
            #include "aclrtlaunch_add_custom.h"
            at::Tensor run_add_custom(const at::Tensor &x, const at::Tensor &y)
            {
              at::Tensor z = at::empty_like(x);
              uint32_t blockDim = 8;
              uint32_t totalLength = 1;
              for (uint32_t size : x.sizes()) {
                  totalLength *= size;
              }
              EXEC_KERNEL_CMD(add_custom, blockDim, x, y, z, totalLength);
              return z;
            }
            ```

    3.  Aten IR注册。

        通过PyTorch提供的TORCH\_LIBRARY\_IMPL注册算子实现，运行在NPU设备需要注册在PrivateUse1上。示例如下：

        ```cpp
        TORCH_LIBRARY_IMPL(npu, PrivateUse1, m)
        {
          m.impl("my_add", TORCH_FN(run_add_custom));
        }
        ```

3.  运行自定义算子。
    1.  设置编译的AI处理器型号，将CMakeLists.txt内的SOC\_VERSION修改为所需产品型号。对应代码位置如下：

        ```
        set(SOC_VERSION "Ascendxxxyy" CACHE STRING "system on chip type")
        ```

        需将Ascendxxxyy修改为对应产品型号。

        > [!NOTE]   
        >SOC\_VERSION获取方法如下：
        >-   非<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：在安装昇腾AI处理器的服务器执行**npu-smi info**命令进行查询，获取**Name**信息。实际配置值为AscendName，例如**Name**取值为xxxyy，实际配置值为Ascendxxxyy。
        >-   <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：在安装昇腾AI处理器的服务器执行**npu-smi info -t board -i** _id_ **-c** _chip\_id_命令进行查询，获取**Chip Name**和**NPU Name**信息，实际配置值为Chip Name\_NPU Name。例如**Chip Name**取值为Ascend_xxx_，NPU Name取值为1234，实际配置值为Ascend_xxx__\__1234。
        >    其中：
        >    -   id：设备id，通过**npu-smi info -l**命令查出的NPU ID即为设备id。
        >    -   chip\_id：芯片id，通过**npu-smi info -m**命令查出的Chip ID即为芯片id。

    2.  运行setup脚本，编译生成whl包。

        > [!NOTE]  
        > 编译工程通过setuptools为用户封装好编译算子kernel和集成到PyTorch，如果需要更多的编译配置，可按需更改CmakeLists.txt文件。

        ```Python
        python setup.py bdist_wheel
        ```

    3.  安装whl包。

        ```Python
        cd dist
        pip install *.whl
        ```

    4.  运行样例。

        ```Python
        cd test
        python test.py
        ```

