# 概述

OpPlugin是Ascend Extension for PyTorch的算子插件，为使用PyTorch框架的开发者提供便捷的NPU算子库调用能力。本手册的主要目标是指导具有一定PyTorch基础的用户完成算子适配工作。本文档提供单算子适配概述、总体思路和开发指导等内容。


## 什么是算子适配

算子适配是指针对特定硬件平台(此处为华为Ascend昇腾芯片及配套运行环境)，对AI框架(如PyTorch)中的算子进行兼容性改造与性能优化的技术过程。算子作为深度学习任务的最小计算单元(如卷积、矩阵乘法、激活函数等)，其原生实现往往面向通用硬件(CPU/GPU/NPU)，算子适配通过接口标准化、计算逻辑重构、底层硬件能力调用等手段，使算子能够适配目标硬件的架构特性，同时确保计算语义一致性与功能完整性。

从技术本质来看，算子适配是连接上层AI框架算子与底层硬件计算资源的桥梁，核心是解决“语义兼容”与“能力映射”两大问题——既保证算子在目标平台的计算结果与原生平台一致，又实现底层硬件计算单元(如Ascend AI Core)的高效调用。


## 为什么要做算子适配

算子适配并非单纯的技术改造，而是打通 PyTorch 生态与 Ascend 平台协同壁垒、释放硬件潜能并满足多样化业务需求的关键举措，具体核心价值体现在以下几方面：

1、兼容性保障：实现算子在 Ascend 平台的功能可执行性，确保算子的输入输出格式、数据类型、计算语义与 PyTorch 原生算子对齐，消除跨平台接口差异、数据格式不兼容等问题，避免运行时语法错误或计算结果偏差。

2、生态适配完整性：支撑 PyTorch 生态与 Ascend 平台的深度融合，确保基于 PyTorch 开发的深度学习模型(训练 / 推理场景)能够无缝迁移至 Ascend 平台，无需修改模型上层代码即可高效运行，完善昇腾硬件的 AI 生态支持能力。

3、自定义能力扩展：支持新增面向 Ascend NPU 的自定义算子开发，针对特定业务场景(如专属算法、行业定制化计算逻辑)提供算子级功能扩展，弥补原生框架算子或现有适配算子的能力缺口，满足差异化、个性化的计算需求，进一步拓展昇腾 NPU 的应用边界。

4、性能最大化：充分发挥 Ascend 硬件的架构优势(如 AI Core 的并行计算能力、异构存储层次、专用加速指令集)，通过计算优化、数据布局调整、内存访问优化等手段，降低算子的计算时延、内存占用与功耗，实现算子在目标平台的性能最优。


## 如何进行算子适配

算子适配需遵循标准化流程，确保算子在Ascend平台的功能正确性与性能最优，主要步骤如下：
1、环境准备：安装配套软件、PyTorch 框架，拉取 torch_npu 源码并进入 OpPlugin 目录。
2、算子分析：参考 PyTorch 原生 Aten IR 定义，明确算子名称、入参 / 返回值、语义等信息。
3、选择适配方式：优先 aclnn 算子（存于 op_plugin/ops/opapi），兼容需求可选 aclop 算子（存于 op_plugin/ops/aclops）。
4、yaml 配置：在 op_plugin_functions.yaml 中声明算子版本、schema、适配方式；需前反向绑定或支持 symint 的算子，分别在 derivatives.yaml、symint 字段补充配置。
5、代码实现：按对应适配方式创建 cpp 文件，实现算子接口及变体，不同版本用编译宏区分。
6、辅助适配：补充接口文档、对外配置、meta 注册，编写单元测试。
7、编译验证：编译安装 torch_npu 包，测试算子功能与性能。

> [!NOTE] 
> 图模式算子开发请参考《PyTorch 图模式使用指南\(TorchAir\)》中的“[自定义算子插件化入图](https://www.hiascend.com/document/detail/zh/Pytorch/720/modthirdparty/torchairuseguide/torchair_00047.html)”章节。


# 算子适配总体思路

## 前提条件

1、配套软件安装：请参见《[Ascend Extension for PyTorch 安装前准备](https://www.hiascend.com/document/detail/zh/Pytorch/720/configandinstg/instg/insg_0002.html)》完成配套软件安装和环境变量配置

2、PyTorch安装：请参见《[Ascend Extension for PyTorch 软件安装指南](https://www.hiascend.com/document/detail/zh/Pytorch/720/configandinstg/instg/insg_0001.html)》完成PyTorch框架的安装

3、torch_npu源码下载：如果用户使用快速安装方式安装torch\_npu插件，适配前需执行如下命令拉取torch\_npu仓对应分支的代码并进入OpPlugin目录。

```
git clone https://gitcode.com/ascend/pytorch.git -b v2.7.1-7.3.0 --recursive
cd pytorch/third_party/op-plugin
```

-   *2.7.1*为PyTorch版本，用户需根据实际情况指定PyTorch版本。
-   *7.3.0*为Ascend Extension for PyTorch软件版本。


## 适配原则

-   OpPlugin对外接口与PyTorch原生Aten IR保持一致。原生Aten IR(Aten Intermediate Representation)是PyTorch 深度学习框架底层的核心中间表示形式，是连接PyTorch上层用户接口与底层硬件执行逻辑的关键数据结构与计算描述载体。OpPlugin通过“接口兼容、语义等价、数据格式一致”的适配层设计，确保上层PyTorch模型代码无需任何修改，即可无缝调用适配后的Ascend平台算子。
    Aten IR接口说明，请参考[pytorch/aten/src/ATen/native](https://github.com/pytorch/pytorch/tree/main/aten/src/ATen/native#readme)。
-   相同算子不同版本使用op\_plugin\_functions.yaml统一维护对外接口，不同版本的适配代码都在一个文件中，通过编译宏VERSION\_BETWEEN来区分不同版本。
-   相同算子不同适配方式放置于不同的文件夹中，使用不同的命名空间。当前仅支持opapi适配(aclnn等)和aclop适配(通过GE注册的算子)方式。
-   接口适配规则，非必要不使用NPUNativeFunction::命名空间中的接口。原生算子使用at::xx调用，自定义算子使用custom\_ops::xx调用，调用其他适配接口使用OpPlugin内部的接口，比如aclnn使用op\_api::xx，aclop算子使用acl\_op::xx。


## 适配文件结构

```
├── op_plugin
│   ├── config                        # 算子适配配置文件目录
│   │   ├── derivatives.yaml          # 算子前反向绑定配置文件
│   │   └── op_plugin_functions.yaml  # 算子对外接口配置文件
│   ├── ops                           # 算子适配代码实现文件目录
│   │   ├── aclops                    # aclop算子
│   │   │   ├── AbsKernelNpu.cpp
│   │   │   └── ...
│   │   └── opapi                     # aclnn算子
│   │       ├── AbsKernelNpuOpApi.cpp
│   │       └── ...
│   ├── OpInterface.h         	      # 编译自动生成op_plugin对外接口的头文件，用于框架侧调用算子
│   ├── OpInterface.cpp               # 编译自动生成op_plugin对外接口路由实现，内部实现不同类型算子分支选择代码
│   ├── AclOpsInterface.h             # 编译自动生成aclop算子插件适配所对应头文件 
│   ├── OpApiInterface.h              # 编译自动生成aclnn算子插件适配所对应头文件
│   ├── ...    
```


## 适配开发流程
PyTorch官方提供的native\_functions.yaml文件定义了PyTorch Native Functions的具体算子定义和分发细节，定义则通过.cpp文件实现。OpPlugin仓库与原生类似，使用yaml文件定义了NPU适配的算子，算子具体适配则存放在.cpp文件中。因此适配算子主要分为两步：

1.  算子yaml配置：自动生成算子声明和注册代码，在yaml文件中完成算子配置的实现。
2.  算子代码适配：算子功能开发的基本流程，在cpp文件中完成算子功能适配的实现。
3.  算子辅助适配：算子接口相关的其他内容，包括算子接口文档、对外接口适配、meta注册、开发者测试等。
4.  算子编译执行：算子开发完成，基于搭建环境进行算子自验证和联调，确保功能实现符合预期。


# 算子适配开发

## 算子yaml配置

> [!NOTE]  
> 以下abs的yaml配置和适配文件为已有配置和文件，此处仅为示例，用户需根据实际场景更改。

OpPlugin采用和原生PyTorch类似的逻辑在yaml中声明算子的各类信息，通过在yaml中配置算子，自动生成算子声明和注册代码。算子的Aten IR定义位于op\_plugin/config/op\_plugin\_functions.yaml文件中，所有版本的定义都在这个文件里面，通过配置不同版本来区分。
以torch API abs/abs\_out为例，包含基于aclnn算子和aclop算子，适配包括两部分，一是算子接口yaml配置，二是算子kernel的适配代码。


### yaml算子配置规则

    ```yaml
    # op_plugin_functions.yaml
    all_version: [v1.11, v2.0, v2.1, v2.2, v2.3, v2.4, v2.5, v2.6, v2.7, v2.8, v2.9, v2.10]

    # 原生算子
    official:
      - func: abs(Tensor self) -> Tensor
        acl_op: all_version
        op_api: all_version
        gen_opapi:
          structured_inherit: abs.out

    # 自定义算子
    custom:
      - func: my_abs(Tensor self) -> Tensor
        acl_op: all_version
        op_api: all_version
        exposed: all_version

    # 入参带有symint的算子
    symint:
      - func: zeros(SymInt[] size, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
        acl_op: [v2.1, newest]
    ```

    参数说明：

    -   all\_version表示当前PyTorch支持的所有版本，可通过[]设置算子支持的版本范围，例如[v2.1, newest]代表该算子支持从v2.1到最新版本。
    -   official和custom分别表示该字段下的算子为PyTorch原生和自定义算子；symint字段表明该算子支持symint类型的入参，该种算子请参考[symint算子适配](#symint算子适配)。
    -   func字段的核心作用是定义算子的 schema(算子描述规范)，其内容完全遵循PyTorch原生Aten IR算子schema的定义规则，通过“算子名称+入参列表+返回参数”的结构化形式，完整描述算子的调用接口与语义约束。具体规则可参考原生定义([LINK](https://github.com/pytorch/pytorch/tree/main/aten/src/ATen/native#readme))。
    -   acl\_op字段后面填版本名称，表示在该版本支持acl\_op调用，如果支持的版本与all\_version表示的版本一致，则可以用"all\_version"表示，可选字段。
    -   op\_api字段后面填版本名称，表示在该版本支持op\_api调用，如果支持的版本与all\_version表示的版本一致，则可以用"all\_version"表示，可选字段。
    -   gen\_opapi对于支持op\_api调用的算子，如果适配代码简单，可以直接调用底层算子，不需要额外的适配，则可以考虑用结构化适配的方式自动生成适配代码，详见章节[结构化适配介绍(可选)](#结构化适配介绍可选)。
    -   exposed字段后面填商用算子版本，一般只用设置正向算子接口 。


    > [!NOTE]  
    >如果存在某个算子适配有两个版本不一致，则需要两个都加上，如std.correction在PyTorch1.11.0版本和PyTorch2.1.0及以上版本的入参名称不同，则需要分开写成两个，通过version区分。
    >```yaml
    >  - func: std.correction(Tensor self, int[1]? dim, *, int? correction, bool keepdim=False) -> Tensor
    >    acl_op: v1.11
    >    op_api: v1.11
    >  - func: std.correction(Tensor self, int[1]? dim=None, *, Scalar? correction=None, bool keepdim=False) -> Tensor
    >    acl_op: [v2.1, newest]
    >    op_api: [v2.1, newest]
    >```


### 自动前反向绑定算子配置

> [!NOTE]  
> 仅适用于需要进行前反向绑定的算子。

PyTorch的算子自动反向微分依赖于算子的前反向绑定，即前向函数和反向函数的绑定。对于原生的算子，官方已有前反向绑定逻辑，插件侧有对应前向算子和反向算子配置即可。对于自定义算子，则需要在插件侧配置前反向自动绑定。具体操作包括：
1. 适配前向和反向算子：与[算子适配开发](#算子适配开发)中一致，分别适配前向算子和反向算子，并在op\_plugin\_functions.yaml中配置前向和反向算子。
2. 配置前反向绑定，将前向和反向算子进行绑定：OpPlugin与原生PyTorch一致，通过op\_plugin/config/derivatives.yaml配置算子的前反向绑定关系，同时相比原生新增了version字段用于表示支持的版本，如下所示：

    ```yaml
    # derivatives.yaml
    - name: l1_loss(Tensor self, Tensor target, int reduction=Mean) -> Tensor
      self: l1_loss_backward(grad, self, target, reduction)
      target: l1_loss_backward(grad, self, target, reduction) * -1
      version: [v2.1, newest]
    ```

> [!NOTE]
> 所有版本的算子前反向绑定都在同一个derivatives.yaml里面，通过version字段来区分版本。


### symint算子配置

> [!NOTE]  
> symint类型算子需参考此部分进行适配。
> 以下yaml配置和适配文件为已有配置和文件，此处仅为示例，用户需根据实际场景更改。

symint为PyTorch在v2.0及以上版本新增的数据类型，op\_plugin/config/op\_plugin\_functions.yaml配置中对应添加了symint类型。配置在symint字段下的函数表示底层函数实现支持了symint类型入参。对于底层不支持symint的函数，则无需在symint字段配置。当需要在symint字段配置时，用户进行如下操作进行算子适配:

1. 算子在配置yaml中除了在official或custom下声明函数外，还需要同时在symint下配置该算子。
2. 算子名称在原有名称上添加\_symint后缀，如配置支持入参为symint类型的zeros算子，yaml配置如下所示：

    ```yaml
    # 官方算子
    official:
     - func: zeros(SymInt[] size, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
       acl_op: v2.1, v2.2, v2.3, v2.4, v2.5, v2.6, v2.7, v2.8, v2.9, v2.10
    
    symint:
     - func: zeros(SymInt[] size, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
       acl_op: v2.1, v2.2, v2.3, v2.4, v2.5, v2.6, v2.7, v2.8, v2.9, v2.10
    ```

3. 算子实现在同算子文件下，新增算子名称为zeros\_symint，且入参中第一个参数的类型为symint相关的类型c10::SymIntArrayRef。由于symint特性只在PyTorch2.0以上支持，symint相关适配代码需要根据实际版本支持情况添加版本编译宏VERSION\_BETWEEN来控制编译。

    ```cpp
    #include "op_plugin/AclOpsInterface.h"
    #include "op_plugin/utils/custom_functions/aclops/inner_compute.h"
    
    namespace acl_op {
    #if VERSION_BETWEEN(V2R1, VERSION_NEWEST)
    at::Tensor zeros_symint(
        c10::SymIntArrayRef size,
        c10::optional<at::ScalarType> dtype_opt,
        c10::optional<at::Layout> layout_opt,
        c10::optional<at::Device> device_opt,
        c10::optional<bool> pin_memory_opt)
    {
        return zeros_common_nocheck(c10::asIntArrayRefUnchecked(size), dtype_opt, layout_opt, device_opt, pin_memory_opt);
    }
    #endif
    
    } // namespace acl_op
    ```


## 算子代码适配

当前支持适配基于aclnn算子和aclop算子两类算子，aclnn算子是较新算子实现方式，推荐使用该方式，其适配文件位于op\_plugin/ops/opapi目录；aclop算子是早期的算子实现方式，不推荐使用，其适配文件位于op\_plugin/ops/aclops目录。
一个算子所有版本的适配代码都在一个文件中，通过编译宏VERSION\_BETWEEN来区分不同版本。
新增自定义算子需要同步新增算子适配文件，并参考如下示例进行相关算子实现的开发。

### aclnn算子适配(推荐)
#### 通用适配

aclnn算子适配文件路径为：op\_plugin/ops/opapi/AbsKernelNpuOpApi.cpp，文件命名规范为算子名称+KernelNpuOpApi，算子名称首字母大写。具体步骤如下：
1. 创建算子适配文件，引入依赖头文件，包括aclnn对外接口、算子声明、基础函数等。
2. 算子接口适配实现，aclnn算子需定义在op_api命名空间中，实现函数出入参与torch api保持一致。

    ```cpp
    //算子适配实现路径op_plugin/ops/opapi/AbsKernelNpuOpApi.cpp 
    // 1. 引入依赖头文件
    // 对外接口头文件，包含op_plugin所有ACLNN算子对外的函数原型
    #include "op_plugin/OpApiInterface.h" 
    // 引用ACLOP算子声明头文件
    #include "op_plugin/AclOpsInterface.h" 
    // torch调用ACLNN算子时，所依赖的基础函数对应的头文件
    #include "op_plugin/utils/op_api_common.h" 

    // 2. 算子接口适配实现
    // ACLNN算子定义在op_api命名空间中
    namespace op_api { 
    using npu_preparation = at_npu::native::OpPreparation; 

    // abs api实现函数，名称唯一，参数与torch api一致。 
    at::Tensor abs(const at::Tensor& self)
    { 
        DO_COMPATIBILITY(aclnnAbs, acl_op::abs(self)); 

        // 构造NPU输出tensor
        at::Tensor result = npu_preparation::apply_tensor_without_format(self); 

        // 计算NPU输出结果
        EXEC_NPU_CMD(aclnnAbs, self, result); 
        return result; 
    } 
    //abs变体代码添加位置
    }  // namespace op_api
    ```

3. 若接口包含多种变体，比如入参带out、原地操作(inplace)变体，需增加相应的适配代码，适配参考如下：

    - 入参带out变体：

    ```cpp
    // abs_out api实现函数，名称唯一，参数与torch api一致
    at::Tensor& abs_out(const at::Tensor& self, at::Tensor& result)
    { 
        // 查找ACLNN算子实现，查找失败则使用ACLOP算子实现
        DO_COMPATIBILITY(aclnnAbs, acl_op::abs_out(self, result)); 
        npu_preparation::check_tensor({self}, result, self); 
        // 异步调用npu执行
        EXEC_NPU_CMD(aclnnAbs, self, result); 
        return result; 
    }
    ```

    - 原地操作(inplace)变体：

    ```cpp
    // abs_ api实现函数，名称唯一，参数与torch api一致。该接口为inplace操作，即输出结果存放在输入tensor中
    at::Tensor& abs_(at::Tensor& self)
    {
        DO_COMPATIBILITY(aclnnAbs, acl_op::abs_(self));
        op_api::abs_out(self, self);
        return self;
    }
    ```

4. 不同版本间适配代码有差异的，所有代码均放在同一个文件中，用编译宏来区分。

    ```cpp
    #include "op_plugin/AclOpsInterface.h"
    #include "op_plugin/OpApiInterface.h"
    #include "op_plugin/utils/op_api_common.h"
    namespace op_api {
    using npu_preparation = at_npu::native::OpPreparation;
    // 1.11的函数入参和2.0及以上版本有区别，需要单独实现，因此用宏来控制
    #if VERSION_BETWEEN(V1R11, V1R11)
    at::Tensor embedding(const at::Tensor& weight, const at::Tensor& indices, int64_t padding_idx, bool scale_grad_by_freq, bool sparse)
    {
        DO_COMPATIBILITY(aclnnEmbedding, acl_op::embedding(weight, indices, padding_idx, scale_grad_by_freq, sparse));
        // 计算输出tensor的size
        auto output_size = op_infer::array_to_small_vector(indices.sizes());
        output_size.emplace_back(weight.size(weight.dim() - 1));
        // 构造NPU输出tensor
        at::Tensor result = npu_preparation::apply_tensor_without_format(output_size, weight.options());
        // 计算NPU输出结果
        EXEC_NPU_CMD(aclnnEmbedding, weight, indices, result);
        return result;
    }
    #endif

    #if VERSION_BETWEEN(V2R0, VERSION_NEWEST)
    at::Tensor embedding_symint(const at::Tensor& weight, const at::Tensor& indices, c10::SymInt padding_idx, bool scale_grad_by_freq, bool sparse)
    {
        DO_COMPATIBILITY(aclnnEmbedding, acl_op::embedding_symint(weight, indices, padding_idx, scale_grad_by_freq, sparse));
        // 计算输出tensor的size
        auto output_size = op_infer::array_to_small_vector(indices.sizes());
        output_size.emplace_back(weight.size(weight.dim() - 1));
        // 构造NPU输出tensor
        at::Tensor result = npu_preparation::apply_tensor_without_format(output_size, weight.options());
        // 计算NPU输出结果
        EXEC_NPU_CMD(aclnnEmbedding, weight, indices, result);
        return result;
    }
    #endif
    } // namespace op_api
    ```


#### 结构化适配(可选)

> [!NOTE]  
>仅aclnn算子可使用此方法进行适配。

结构化适配指通过在op\_plugin\_functions.yaml中进行配置，自动完成算子适配实现，自动生成的适配文件位于op\_plugin/ops/opapi/StructKernelNpuOpApi.cpp。
判断是否可结构化依据：opapi对应的aclnn算子与Aten IR的语义对齐，适配层除申请output tensor，无其他适配逻辑。

YAML配置有以下两种方式，可根据实际情况进行选择。每个结构化适配的函数必须在op\_plugin\_functions.yaml中配置，具体实现如下：

1. 常规场景

  ```yaml
  - func: func_name(ArgType arg0[=default], ArgType arg1[=default], ...) -> Return
    op_api: v2.7
    gen_opapi:
      out:
        size: arg0
        dtype: arg1.scalar_type()
        name: arg0
      exec: aclnnFuncName
  ```

  字段含义如下：

  -   gen\_opapi：表示对应API可结构化，其他字段需要配置在此字段下。
  -   out：表示函数的输出，此字段下面包含size和dtype字段，如果包含多个输出，可配置成out0、out1等。对于out类接口，此字段不可自定义，需要与Aten IR定义的输出参数名相同。对于inplace类接口，不需要配置此字段。
  -   size：配置输出tensor的shape大小，如果大小和schema中的某个参数相同，可以配置成输入参数的名字。也可配置成自定义infershape函数，infershape函数需在KernelNpuOutputSize.h中实现。对于out类接口，如果输出shape不变，可省略此字段。配置方式主要包含以下几种：

      ```yaml
      - func: func_name(ArgType arg0[=default], ArgType arg1[=default], ...) -> Return
      Aten IR定义：
      - func: func_name(ArgType arg0, ArgType arg1, ...) -> Return
      方式一：和输入参数相同
        size: arg0

      方式二：枚举每个维度的值
        size: '{4, arg0.size(0), arg0.size(1), arg1.size(0)}'

      方式三：条件表达式
        size: 'arg1 == 1? arg0.sizes(): at::ArrayRef<int64_t>()'

      方式四：在KernelNpuOutputSize.h中自定义infershape函数, 例如broadcast_ops_npu_output_size
        size: broadcast_ops_npu_output_size(arg0, arg1)
      ```

  -   dtype：配置输出tensor的dtype大小，如果大小和schema中的某个参数相同，可以配置成输入参数的名字。也可配置成自定义inferdtype函数，inferdtype函数需在KernelNpuOutputDtype.h中实现。对于out类接口，如果输出dtype不需要check，可省略此字段。配置方式主要包含以下几种：

      ```yaml
      Aten IR定义：
      - func: func_name(ArgType arg0, ArgType arg1, ...) -> Return
      方式一：和输入参数相同
        dtype: arg0

      方式二：配置成已知的dtype类型
        dtype: at::kFloat

      方式三：条件表达式
        dtype: 'isIntegralType(arg0.scalar_type(), true) ? at::kFloat : arg0.scalar_type()'

      方式四：在KernelNpuOutputDtype.h中自定义inferdtype函数。
        dtype: inferdtype(arg0, arg1)
      ```

  -   name：输出结果涉及named tensor逻辑，可配置此字段，当前仅支持name和输入参数相同的配置，不涉及可忽略。
  -   exec：配置EXEC\_NPU\_CMD对应的参数，如果除aclnnname，其它参数顺序和Aten IR的顺序相同，可只配置aclnnname，如_aclnnAbs_。以abs为例，exec字段可以配置成下面两种方式。

      ```yaml
      - func: abs.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
        方式一：
        exec: aclnnAbs, self, out

        方式二：
        exec: aclnnAbs
      ```

2. 继承场景

    ```yaml
    - func: func_name(ArgType arg0[=default], ArgType arg1[=default], ...) -> Return
      op_api: v2.7
      gen_opapi:
        structured_inherit: func_name.out
    ```

    structured\_inherit：如果原函数或inplace类接口的字段配置与out类接口的字段配置相同，可通过此字段继承对应的out类接口。

    以abs为例，原函数和out类函数的out属性和exec相同，可通过structured\_inherit字段继承。

    ```yaml
      - func: abs(Tensor self) -> Tensor
        op_api: [v1.11, newest]
        gen_opapi:
          structured_inherit: abs.out
    ```


### aclop算子适配(不推荐)

aclop算子是早期的算子实现方式，不推荐使用。适配文件路径为：op\_plugin/ops/aclops/AbsKernelNpu.cpp，文件命名规范为算子名称+KernelNpu，算子名称首字母大写。具体步骤如下：
1. 创建算子适配文件，引入依赖头文件，包括aclop对外接口、算子声明、基础函数等。
2. 算子接口适配实现，aclop算子需定义在acl_op命名空间中，实现函数出入参与torch api保持一致。

    ```cpp
    // 算子适配实现文件路径op_plugin/ops/aclops/AbsKernelNpu.cpp 
    // 1. 引入依赖头文件
    // 对外接口头文件，包含op_plugin所有aclop算子对外的函数原型
    #include "op_plugin/AclOpsInterface.h" 
    // torch调用ACLOP算子时，所依赖的基础函数对应的头文件
    #include "op_plugin/utils/OpAdapter.h" 
    
    // 2. 算子接口适配实现
    // CANN算子定义在acl_op命名空间中
    namespace acl_op { 
    using npu_preparation = at_npu::native::OpPreparation; 
    using npu_utils = at_npu::native::NpuUtils; 
    // 不对外暴露的接口，都定义在匿名空间中。常见为xx_nocheck等，直调ACLOP算子，不做内存、shape校验的函数。 
    namespace{ 
    at::Tensor& abs_out_nocheck(at::Tensor& result, const at::Tensor& self)
    { 
        at_npu::native::OpCommand cmd; 
        cmd.Name("Abs") 
            .Input(self) 
            .Output(result) 
            .Run(); 
        return result; 
    } 
    } // namespace acl_op
    
    // abs api实现函数，名称唯一，参数与torch api一致。 
    at::Tensor abs(const at::Tensor& self)
    { 
        // 构造输出tensor，调用ACLOP算子。 
        auto output_size = op_infer::infershape_for_elewise(self); 
        at::Tensor result = npu_preparation::apply_tensor(self, output_size); 
        abs_out_nocheck(result, self); 
        return result; 
    }
    //abs变体代码添加位置
    } // namespace acl_op
    ```

3. 若接口包含多种变体，比如入参带out、原地操作(inplace)变体，需增加相应的适配代码，适配参考如下：

    - 入参带out变体：

    ```cpp
    // abs_out api实现函数，名称唯一，参数与torch api
    at::Tensor& abs_out(const at::Tensor& self, at::Tensor& result)
    { 
        // CheckOut作用：校验result的size、dtype等是否符合预期。若dtype不符合预期，则抛错。若size不符合则进行resize操作
        npu_preparation::CheckOut({self}, result, self); 
        // check_match作用：校验result是否为连续。因ACLOP算子无法支持非连续输出，result非连续时，需要单独处理。
        if (!npu_utils::check_match(&result)) { 
            // 若result非连续，创建连续tensor(contig_tensor)，接收ACLOP算子(abs)的输出。再将contig_tensor拷贝到原始输出result。
            at::Tensor contiguous_result = npu_utils::format_contiguous(result); 
            abs_out_nocheck(contigTensor, self); 
            npu_utils::format_fresh_view(result, contiguous_result); 
        } else { 
            // 若result连续，直接调用ACLOP算子。
            abs_out_nocheck(result, self); 
        } 
        return result; 
    }
    ```

    - 原地操作(inplace)变体：

    ```cpp
    // abs_ api实现函数，名称唯一，参数与torch api一致。该接口为inplace操作，即输出结果存放在输入tensor中。
    at::Tensor& abs_(at::Tensor& self)
    {
        // 调用out接口，避免因self作为输出时，非连续场景下，直调ACLOP算子结果出错。
        return acl_op::abs_out(self, self);
    }
    ```

4. 不同版本间适配代码有差异的，所有代码均放在同一个文件中，用编译宏来区分。

    ```cpp
    #include "op_plugin/AclOpsInterface.h"
    #include "op_plugin/utils/custom_functions/aclops/inner_compute.h"
    namespace acl_op {
    // 1.11的函数入参和2.0及以上版本有区别，因此用宏来控制
    #if VERSION_BETWEEN(V1R11, V1R11)
    at::Tensor embedding(
        const at::Tensor& weight,
        const at::Tensor& indices,
        int64_t padding_idx,
        bool scale_grad_by_freq,
        bool sparse)
    {
        return embedding_common_nocheck(weight, indices);
    }
    #endif

    // 2.0及以上版本的代码都一致
    #if VERSION_BETWEEN(V2R0, VERSION_NEWEST)
    at::Tensor embedding_symint(
        const at::Tensor& weight,
        const at::Tensor& indices,
        c10::SymInt padding_idx,
        bool scale_grad_by_freq,
        bool sparse)
    {
        return embedding_common_nocheck(weight, indices);
    }
    #endif
    } // namespace acl_op
    ```

## 算子辅助适配

1. 算子接口说明文档：在op-plugin仓op\-plugin/codegen/templates/\_op\_plugin\_docs.py文件补充新增接口的说明文档，一般只用设置正向，具体示例如下：
    ```python
    _add_torch_npu_docstr(
        "npu_transpose",
        """
    torch_npu.npu_transpose(self, perm, require_contiguous=True) -> Tensor
    功能描述
    返回原始张量视图，其维度已permute，结果连续。支持FakeTensor模式。

    参数说明
    self (Tensor) - 输入张量。
    perm (ListInt) - 对应维度排列。
    require_contiguous(Bool，默认值为True) - 用户是否需要对输入Tensor做转连续。设置为False时，表示不对输入Tensor做转连续。用户明确输入Tensor为连续Tensor或转置Tensor时，才能设置为True。
    示例
    >>> x = torch.randn(2, 3, 5).npu()
    >>> x.shape
    torch.Size([2, 3, 5])
    >>> x1 = torch_npu.npu_transpose(x, (2, 0, 1))
    >>> x1.shape
    torch.Size([5, 2, 3])
    """
    )
    ```

2. 算子接口对外公开配置

对外公开接口需在以下文件中新增接口配置：
  - op-plugin/test/allowlist\_for_publicAPI.json
  - op-plugin/test/core_tests/torch\_npu_OpApi\_schema\_all.json
    以"op-api:"开头的表示Python接口，以"func:"开头的接口表示C++接口

3. 算子接口meta实现

在fx、compile等功能使用时，需注册算子接口的meta实现，使得走faketensor时可以正常执行。目前算子的meta实现，统一注册在文件op_plugin/python/meta/\_meta\_registrations.py。
  ```python
  @impl(m, "npu_transpose")
  def npu_transpose_meta(self, perm, require_contiguous=True):
      output = self.permute(perm)
      return torch.empty_like(output, dtype=self.dtype)
  ```

4. 算子接口开发者测试

开发者测试（UT）通过功能正确性验证、边界条件覆盖等，确保算子实现预期，降低联调成本，同时作为长期维护的质量基线，保障算子适配全生命周期的稳定性，自定义算子适配test目录为test/test\_custom\_ops。
以npu_transpose为例，需要实现以下用例：

    ```python
    import torch
    import numpy as np
    import torch_npu

    from torch_npu.testing.testcase import TestCase, run_tests
    from torch_npu.testing.common_utils import create_common_tensor


    class TestTransepose(TestCase):
        def test_transepose(self):
            def cpu_op_exec(input1, perm):
                output = input1.permute(perm)
                output = output.numpy()
                return output

            def npu_op_exec(input1, perm):
                output = torch_npu.npu_transpose(input1, perm)
                output = output.to("cpu")
                output = output.numpy()
                return output

            shape_format = [
                [[np.float32, 0, (5, 3, 6, 4)], [1, 0, 2, 3]],
                [[np.float16, 0, (5, 3, 6, 4)], [0, 3, 2, 1]],
            ]

            for item in shape_format:
                cpu_input1, npu_input1 = create_common_tensor(item[0], 0, 100)
                cpu_output = cpu_op_exec(cpu_input1, item[1])
                npu_output = npu_op_exec(npu_input1, item[1])

                self.assertRtolEqual(cpu_output, npu_output)


    if __name__ == "__main__":
        run_tests()
    ```


## 算子编译执行

算子适配完成后，需编译torch\_npu包，推荐使用容器场景进行编译安装，具体操作可参考《AscendExtension for PyTorch 软件安装指南》中的“[方式二：源码编译安装](https://www.hiascend.com/document/detail/zh/Pytorch/720/configandinstg/instg/insg_0005.html)”章节的“方式一(推荐)：容器场景”。torch\_npu安装完成后即可进行新增算子接口的测试验证。


# 附录

## PyTorch scheme规则
官方schema指导：https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/README.md

由于PTA 2.1版本使用官方的torchgen进行代码生成，因此要遵循官方的一些生成规范，未满足schema规范的算子会在编译时报错。当前有涉及到的有：

- 函数名已"new_"开头、"_like"结尾或入参中带有tensor_options但又没tensor入参的，需要有CompositeExplicitAutograd的dispatch。

  ```yaml
  - func: empty_with_format(int[] size, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None, int acl_format=2) -> Tensor
    dispatch:
      CompositeExplicitAutograd: empty_with_format
  ```
- 函数名中带有"rand"、带有"dropout"、或者入参带有generator的，需要有nondeterministic_seeded的tag。

  ```yaml
  - func: dropout_with_byte_mask(Tensor self, float p, bool train) -> Tensor
    tags: nondeterministic_seeded
  ```


## 算子适配常见API接口

torch_npu 算子操作基础接口：https://gitcode.com/Ascend/pytorch/blob/master/torch_npu/csrc/framework/utils/OpPreparation.h。常见接口如下：

1. **`at_npu::native::OpPreparation::apply_tensor`**  
   - 功能：创建与输入张量属性（设备、数据类型、格式）一致的输出张量，适用于大多数算子的输出初始化。  
   - 示例：  
     ```cpp
     at::Tensor result = at_npu::native::OpPreparation::apply_tensor(rois, output_size);
     ```

2. **`at_npu::native::OpPreparation::apply_tensor_without_format`**  
   - 功能：创建与输入张量尺寸和数据类型匹配的输出张量，但不指定格式（如通道顺序），适用于对格式无强制要求的场景。  
   - 示例：  
     ```cpp
     at::Tensor result = at_npu::native::OpPreparation::apply_tensor_without_format(output_size, input.options());
     ```

3. **`at_npu::native::OpPreparation::check_tensor`**  
   - 功能：校验输出张量与输入张量的属性一致性（尺寸、数据类型等），若不匹配则调整，保障算子稳健性。  
   - 示例：  
     ```cpp
     at_npu::native::OpPreparation::check_tensor({self}, out, out, output_size);
     ```

4. **`copy_scalar_to_device`**  
   - 功能：将主机（CPU）上的标量值复制到设备（NPU），确保标量数据能在NPU计算中正确使用，解决跨设备数据访问问题。  
   - 示例（参考逻辑）：  
     ```cpp
     at::Scalar scalar = 2.0f;
     at::Tensor device_scalar = copy_scalar_to_device(scalar, input.device());
     ```

5. **`binary_op_check`**  
   - 功能：校验二元算子的两个输入张量是否满足运算条件（如设备一致性、数据类型兼容性等），提前发现不合法输入并抛出异常。  
   - 示例（参考逻辑）：  
     ```cpp
     binary_op_check(input1, input2, "add");
     // 确保input1和input2在设备、数据类型等方面符合add算子的要求
     ```


## 算子适配常见宏定义

算子适配宏定义接口：https://gitcode.com/Ascend/op-plugin/blob/7.2.0/op_plugin/utils/op_api_common.h。常见宏定义如下：

1. **`DO_COMPATIBILITY`**  
   - 功能：用于算子兼容性处理，当NPU原生算子不可用时自动降级为备选实现（如CPU版本），确保不同环境下的功能兼容性。  
   - 示例：  
     ```cpp
     DO_COMPATIBILITY(aclnnForeachTan, at::native::foreach_tensor_tan_slow_(self));
     ```

2. **`EXEC_NPU_CMD`**  
   - 功能：封装NPU底层算子调用逻辑，自动处理输入输出张量传递，简化与NPU硬件接口的交互，支持批量拆分执行。  
   - 示例：  
     ```cpp
     EXEC_NPU_CMD(aclnnForeachAddScalarV2, temp_tensors1, scalar_, temp_result);
     ```

3. **`OPS_ERROR`**  
   - 功能：生成带错误码的异常信息，用于参数校验或运行时错误提示，提升错误定位精度。  
   - 示例：  
     ```cpp
     TORCH_CHECK(src.is_sparse(), "add(sparse, dense) is not supported. Use add(dense, sparse) instead.", OPS_ERROR(ErrCode::VALUE));
     ```

4. **`VERSION_BETWEEN`**  
   - 功能：根据NPU版本范围进行条件编译，仅在指定版本区间内生效，适配不同硬件版本特性差异（结合CMake版本判断逻辑）。  
   - 示例（逻辑参考）：  
     ```cpp
     #if VERSION_BETWEEN(V2R1, VERSION_NEWEST)
     // 特定版本生效的代码
     #endif
     ```

5. **`FLOP_COUNT`**  
   - 功能：统计算子的浮点运算次数（FLOPs），用于性能分析和优化。  
   - 示例（逻辑参考）：  
     ```cpp
     FLOP_COUNT(FlopCounter::mm_flop, input, weight_t);
     ```
