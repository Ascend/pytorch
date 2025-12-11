# 适配开发

## 简介

OpPlugin是Ascend Extension for PyTorch的算子插件，为使用PyTorch框架的开发者提供便捷的NPU算子库调用能力。OpPlugin算子插件的编译与使用均依赖昇腾Ascend Extension for PyTorch。在编译OpPlugin之前，请参见《[CANN 软件安装指南](hiascend.com/document/detail/zh/canncommercial/83RC1/softwareinst/instg/instg_0000.html)》（商用版）或《[CANN 软件安装指南](hiascend.com/document/detail/zh/CANNCommunityEdition/83RC1/softwareinst/instg/instg_0000.html)》（社区版）完成CANN软件的安装，参见《[Ascend Extension for PyTorch 软件安装指南](https://www.hiascend.com/document/detail/zh/Pytorch/720/configandinstg/instg/insg_0001.html)》完成PyTorch框架的安装。本文档提供单算子适配开发指导，主要包括适配原则、适配文件结构和算子适配开发三部分内容。图模式算子开发请参考《PyTorch 图模式使用指南\(TorchAir\)》中的“[自定义算子插件化入图](https://www.hiascend.com/document/detail/zh/Pytorch/720/modthirdparty/torchairuseguide/torchair_00047.html)”章节。

## 前提条件

已安装gcc和cmake，具体可参见《Ascend Extension for PyTorch 软件安装指南》中的“[（可选）安装扩展模块](https://www.hiascend.com/document/detail/zh/Pytorch/720/configandinstg/instg/insg_0008.html)”章节。

如果用户使用快速安装方式安装torch\_npu插件，适配前需执行如下命令拉取torch\_npu仓对应分支的代码并进入OpPlugin目录。

```
git clone https://gitcode.com/ascend/pytorch.git -b v2.7.1-7.3.0 --recursive
cd pytorch/third_party/op-plugin
```

-   *2.7.1*为PyTorch版本，用户需根据实际情况指定PyTorch版本。
-   *7.3.0*为Ascend Extension for PyTorch软件版本。

## 适配原则

-   OpPlugin对外接口与torch原生Aten IR保持一致。Aten IR接口说明，请参考[pytorch/aten/src/ATen/native](https://github.com/pytorch/pytorch/tree/main/aten/src/ATen/native#readme)。
-   不同的torch版本使用op\_plugin\_functions.yaml维护本身的对外接口，配置规则可参见[算子适配开发](#算子适配开发)。
-   不同适配方式的算子放置于不同的文件夹中，使用不同的命名空间，当前仅有opapi适配算子（aclnn等）和aclop适配算子（通过GE注册的算子）。
-   非必要不使用NPUNativeFunction::命名空间中的接口。自定义算子使用custom\_ops::xx调用，原生算子使用at::xx调用，调用其他适配接口使用OpPlugin内部的接口，比如aclnn使用op\_api::xx，aclop算子使用acl\_op::xx。
-   当前只支持aclnn算子和aclop算子。

## 适配文件结构

```
.
├── op_plugin
│   ├── config                         # 算子配置文件目录
│   │   ├── derivatives.yaml          # 算子前反向绑定配置文件
│   │   └── op_plugin_functions.yaml  # 算子对外接口配置文件
│   ├── ops                            # 算子适配文件目录
│   │   ├── aclops                    # aclop算子适配目录
│   │   │   ├── AbsKernelNpu.cpp
│   │   │   └── ...
│   │   └── opapi                     # aclnn算子适配目录
│   │       ├── sparse                # sparse相关算子适配目录
│   │       │   └── SparseTensorUtils.h
│   │       ├── AbsKernelNpuOpApi.cpp
│   │       └── ...
│   ├── OpInterface.h         	  # 编译PyTorch框架后自动生成op_plugin对外接口的头文件，用于框架侧调用算子
│   ├── OpInterface.cpp               # 编译PyTorch框架后自动生成op_plugin对外接口路由实现，内部实现不同类型算子分支选择代码
│   ├── AclOpsInterface.h             # 编译PyTorch框架后自动生成aclop算子插件适配所对应头文件 
│   ├── OpApiInterface.h              # 编译PyTorch框架后自动生成aclnn算子插件适配所对应头文件
│   ├── ...    
```

## 算子适配开发

PyTorch官方提供的native\_functions.yaml文件定义了PyTorch Native Functions的具体算子定义和分发细节，定义则通过.cpp文件实现。OpPlugin仓库与原生类似，使用yaml文件定义了NPU适配的算子，算子具体适配则存放在.cpp文件中。

> [!NOTE]  
> 以下abs的yaml配置和适配文件为已有配置和文件，此处仅为示例，用户需根据实际场景更改。

因此适配算子主要分为两步：

1.  在yaml文件中配置算子。
2.  完成算子适配的实现。

以torch API abs/abs\_out为例，包含基于aclnn算子和aclop算子，适配包括两部分，一是算子接口yaml配置，二是算子kernel的适配代码。

1.  算子yaml配置。

    OpPlugin采用和原生PyTorch类似的逻辑在yaml中声明算子的各类信息，通过在yaml中配置算子，自动生成算子声明和注册代码。算子的Aten IR定义位于op\_plugin/config/op\_plugin\_functions.yaml文件中，所有版本的定义都在这个文件里面，通过配置不同版本来区分。

    yaml中算子配置规则如下面所示：

    ```yaml
    # op_plugin_functions.yaml
    all_version: [v1.11, v2.0, v2.1, v2.2, v2.3, v2.4, v2.5, v2.6, v2.7, v2.8, v2.9, v2.10]
    # 官方算子
    official:
      - func: abs(Tensor self) -> Tensor
        acl_op: all_version
        op_api: v1.11, v2.1, v2.2, v2.3, v2.4, v2.5, v2.6, v2.7, v2.8, v2.9, v2.10
        gen_opapi:
          structured_inherit: abs.out
    # 自定义算子
    custom:
      - func: my_abs(Tensor self) -> Tensor
        acl_op: v1.11, v2.1, v2.2, v2.3, v2.4, v2.5, v2.6, v2.7, v2.8, v2.9, v2.10
        op_api: all_version
    #入参带有symint的算子
    symint:
      - func: zeros(SymInt[] size, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
        acl_op: [v2.1, newest]
    ```

    参数说明：

    -   all\_version表示当前PyTorch支持的所有版本。
    -   official和custom分别表示该字段下的算子为PyTorch原生和自定义算子；symint字段表明该算子支持symint类型的入参，该种算子请参考[symint算子适配](#symint算子适配)。
    -   func定义了算子的schema，主要有名称、入参和返回参数，具体规则可参考原生定义（[LINK](https://github.com/pytorch/pytorch/tree/main/aten/src/ATen/native#readme)）。
    -   acl\_op字段后面填版本名称，表示在该版本支持acl\_op调用，如果支持的版本与all\_version表示的版本一致，则可以用"all\_version"表示，可选字段。
    -   op\_api字段后面填版本名称，表示在该版本支持op\_api调用，如果支持的版本与all\_version表示的版本一致，则可以用"all\_version"表示，可选字段。
    -   gen\_opapi对于支持op\_api调用的算子，如果适配代码简单，可以直接调用底层算子，不需要额外的适配，则可以考虑用结构化适配的方式自动生成适配代码，详见章节[结构化适配介绍（可选）](#结构化适配介绍可选)。

        > [!NOTE]  
        >如果存在某个Aten IR有两个版本不一致，则需要两个都加上，如std.correction在PyTorch1.11.0版本和PyTorch2.1.0及以上版本的入参名称不同，则需要分开写成两个，通过version区分。
        >```yaml
        >  - func: std.correction(Tensor self, int[1]? dim, *, int? correction, bool keepdim=False) -> Tensor
        >    acl_op: v1.11
        >    op_api: v1.11
        >  - func: std.correction(Tensor self, int[1]? dim=None, *, Scalar? correction=None, bool keepdim=False) -> Tensor
        >    acl_op: [v2.1, newest]
        >    op_api: [v2.1, newest]
        >```

2.  算子适配实现。

    当前支持适配基于aclnn算子和aclop算子两类算子，aclnn算子是较新算子实现方式，推荐使用该方式，其适配文件位于op\_plugin/ops/opapi目录；aclop算子是早期的算子实现方式，不推荐使用，其适配文件位于op\_plugin/ops/aclops目录。

    一个算子所有版本的适配代码都在一个文件中，通过编译宏VERSION\_BETWEEN来区分不同版本。

    新增自定义算子需要同步新增算子适配文件，并参考如下示例进行相关算子实现的开发。

    -   aclnn算子适配。

        如果所有版本的适配代码一致，则不需要额外添加编译宏，适配文件路径为：op\_plugin/ops/opapi/AbsKernelNpuOpApi.cpp，文件命名规范为算子名称+KernelNpuOpApi，算子名称首字母大写。

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

        **abs**接口有很多变体，比如入参带out的变体，原地操作（inplace）变体，如有需要可增加相应的适配代码。

        -   入参带out变体：

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

        -   原地操作（inplace）变体：

            ```cpp
            // abs_ api实现函数，名称唯一，参数与torch api一致。该接口为inplace操作，即输出结果存放在输入tensor中
            at::Tensor& abs_(at::Tensor& self)
            {
                DO_COMPATIBILITY(aclnnAbs, acl_op::abs_(self));
                op_api::abs_out(self, self);
                return self;
            }
            ```

        > [!NOTE]  
        >不同版本间适配代码有差异的，所有代码均放在同一个文件中，用编译宏来区分。
        >```cpp
        >#include "op_plugin/AclOpsInterface.h"
        >#include "op_plugin/OpApiInterface.h"
        >#include "op_plugin/utils/op_api_common.h"
        >namespace op_api {
        >using npu_preparation = at_npu::native::OpPreparation;
        >// 1.11的函数入参和2.0及以上版本有区别，需要单独实现，因此用宏来控制
        >#if VERSION_BETWEEN(V1R11, V1R11)
        >at::Tensor embedding(const at::Tensor& weight, const at::Tensor& indices, int64_t padding_idx, bool scale_grad_by_freq, bool sparse)
        >{
        >    DO_COMPATIBILITY(aclnnEmbedding, acl_op::embedding(weight, indices, padding_idx, scale_grad_by_freq, sparse));
        >    // 计算输出tensor的size
        >    auto output_size = op_infer::array_to_small_vector(indices.sizes());
        >    output_size.emplace_back(weight.size(weight.dim() - 1));
        >    // 构造NPU输出tensor
        >    at::Tensor result = npu_preparation::apply_tensor_without_format(output_size, weight.options());
        >    // 计算NPU输出结果
        >    EXEC_NPU_CMD(aclnnEmbedding, weight, indices, result);
        >    return result;
        >}
        >#endif
        >#if VERSION_BETWEEN(V2R0, VERSION_NEWEST)
        >at::Tensor embedding_symint(
        >    const at::Tensor& weight,
        >    const at::Tensor& indices,
        >    c10::SymInt padding_idx,
        >    bool scale_grad_by_freq,
        >    bool sparse)
        >{
        >    DO_COMPATIBILITY(aclnnEmbedding, acl_op::embedding_symint(weight, indices, padding_idx, scale_grad_by_freq, sparse));
        >    // 计算输出tensor的size
        >    auto output_size = op_infer::array_to_small_vector(indices.sizes());
        >    output_size.emplace_back(weight.size(weight.dim() - 1));
        >    // 构造NPU输出tensor
        >    at::Tensor result = npu_preparation::apply_tensor_without_format(output_size, weight.options());
        >    // 计算NPU输出结果
        >    EXEC_NPU_CMD(aclnnEmbedding, weight, indices, result);
        >    return result;
        >}
        >#endif
        >} // namespace op_api
        >```

    -   aclop算子适配。

        如果所有版本的适配代码一致，则不需要额外添加编译宏，适配文件路径为：op\_plugin/ops/aclops/AbsKernelNpu.cpp，文件命名规范为算子名称+KernelNpu，算子名称首字母大写。

        ```cpp
        // 算子适配实现文件路径op_plugin/ops/aclops/AbsKernelNpu.cpp 
        // 1. 引入依赖头文件
        // 对外接口头文件，包含op_plugin所有aclop算子对外的函数原型
        #include "op_plugin/AclOpsInterface.h" 
        // torch调用ACLOP算子时，所依赖的基础函数对应的头文件
        #include "op_plugin/utils/OpAdapter.h" 
        
        // 2. 算子接口适配实现
        // opplugin内适配的算子对外接口都定义在op_plugin命名空间中，外部调用方式为op_plugin::abs、op_plugin::abs_out；内部不同类型的算子适配采用不同的命名空间
        // CANN算子定义在acl_op命名空间中， 
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
        } // namespace 
        
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

        abs接口有很多变体，比如入参带out的变体，原地操作（inplace）变体，如有需要可增加相应的适配代码。

        -   入参带out变体：

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

        -   原地操作（inplace）变体：

            ```cpp
            // abs_ api实现函数，名称唯一，参数与torch api一致。该接口为inplace操作，即输出结果存放在输入tensor中。
            at::Tensor& abs_(at::Tensor& self)
            {
                // 调用out接口，避免因self作为输出时，非连续场景下，直调ACLOP算子结果出错。
                return acl_op::abs_out(self, self);
            }
            ```

        > [!NOTE]  
        > 不同版本间适配代码有差异的，所有代码均放在同一个文件中，用编译宏来区分。
        >```cpp
        >#include "op_plugin/AclOpsInterface.h"
        >#include "op_plugin/utils/custom_functions/aclops/inner_compute.h"
        >namespace acl_op {
        >// 1.11的函数入参和2.0及以上版本有区别，因此用宏来控制
        >#if VERSION_BETWEEN(V1R11, V1R11)
        >at::Tensor embedding(
        >    const at::Tensor& weight,
        >    const at::Tensor& indices,
        >    int64_t padding_idx,
        >    bool scale_grad_by_freq,
        >    bool sparse)
        >{
        >    return embedding_common_nocheck(weight, indices);
        >}
        >#endif
        >// 2.0及以上版本的代码都一致
        >#if VERSION_BETWEEN(V2R0, VERSION_NEWEST)
        >at::Tensor embedding_symint(
        >    const at::Tensor& weight,
        >    const at::Tensor& indices,
        >    c10::SymInt padding_idx,
        >    bool scale_grad_by_freq,
        >    bool sparse)
        >{
        >    return embedding_common_nocheck(weight, indices);
        >}
        >#endif
        >} // namespace acl_op
        >```

## 自动前反向绑定算子配置

> [!NOTE]  
> 仅适用于需要进行前反向绑定的算子。

PyTorch的算子自动反向微分依赖于算子的前反向绑定，即前向函数和反向函数的绑定。对于原生的算子，官方已有前反向绑定逻辑，插件侧有对应前向算子和反向算子适配即可（只需要在op\_plugin\_functions.yaml里面配置）。对于自定义算子，则需要在插件侧配置前反向自动绑定。

针对需要绑定前反向的算子（包括自定义算子和前反向绑定逻辑与原生不一致的原生算子）提供自动绑定前向算子和反向算子的功能。

-   适配前向和反向算子：与[算子适配开发](#算子适配开发)中一致，分别适配前向算子和反向算子，并在op\_plugin\_functions.yaml中配置前向和反向算子。
-   配置前反向绑定，将前向和反向算子进行绑定：OpPlugin与原生PyTorch一致，通过derivatives.yaml配置算子的前反向绑定关系，同时相比原生新增了version字段用于表示支持的版本，如下所示：

    ```yaml
    # derivatives.yaml
    - name: l1_loss(Tensor self, Tensor target, int reduction=Mean) -> Tensor
      self: l1_loss_backward(grad, self, target, reduction)
      target: l1_loss_backward(grad, self, target, reduction) * -1
      version: [v2.0, newest]
    ```

    所有版本的算子前反向绑定都在同一个derivatives.yaml里面，通过version字段来区分版本。

## symint算子适配

> [!NOTE]  
>-  symint类型算子需参考此部分进行适配。
>-  以下yaml配置和适配文件为已有配置和文件，此处仅为示例，用户需根据实际场景更改。

symint为PyTorch在2.0及以上版本新增的数据类型，yaml配置中对应添加了symint字段。配置在symint字段下的函数表示底层函数实现支持了symint类型入参。对于底层不支持symint的函数，则无需在symint字段配置。部分情况需要在symint字段配置时，用户需进行如下操作进行算子适配:

-   算子在配置yaml中除了在official或custom下声明函数外，还需要同时在symint下配置该算子。
-   算子名称在原有名称上添加\_symint后缀，如配置支持入参为symint类型的zeros算子，其yaml配置为：

    ```yaml
    # 官方算子
    official:
     - func: zeros(SymInt[] size, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
       acl_op: v2.1, v2.2, v2.3, v2.4, v2.5, v2.6, v2.7, v2.8, v2.9, v2.10
    
    symint:
     - func: zeros(SymInt[] size, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
       acl_op: v2.1, v2.2, v2.3, v2.4, v2.5, v2.6, v2.7, v2.8, v2.9, v2.10
    ```

    其算子实现如下，其中算子名称为zeros\_symint，且入参中第一个参数的类型为symint相关的类型c10::SymIntArrayRef，同时由于symint特性只在PyTorch2.0以上支持，symint相关适配代码需要根据实际版本支持情况添加版本编译宏VERSION\_BETWEEN来控制编译：

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

## 结构化适配介绍（可选）

> [!NOTE]  
>仅aclnn算子可使用此方法进行适配。

结构化适配指通过在op\_plugin\_functions.yaml中进行配置，自动完成算子适配实现。判断是否可结构化依据：opapi对应的aclnn算子与Aten IR的语义对齐，适配层除申请output tensor，无其他适配逻辑。自动生成的适配文件位于op\_plugin/ops/opapi/StructKernelNpuOpApi.cpp。

YAML配置有以下两种方式，可根据实际情况进行选择。每个结构化适配的函数必须在op\_plugin\_functions.yaml中配置，具有如下格式：

-   方式一（常规场景）：

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

    各个字段的含义如下：

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

-   方式二（继承场景）：

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

## 后续处理

算子适配完成后，需编译torch\_npu包，推荐使用容器场景进行编译，具体操作可参考《AscendExtension for PyTorch 软件安装指南》中的“[方式二：源码编译安装](https://www.hiascend.com/document/detail/zh/Pytorch/720/configandinstg/instg/insg_0005.html)”章节的“方式一（推荐）：容器场景”。

