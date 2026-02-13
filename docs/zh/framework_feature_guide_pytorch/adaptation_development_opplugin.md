# 适配开发

## 适配文件结构

```
├── op_plugin
│   ├── config                                 # 算子适配配置文件目录
│   │   ├── derivatives.yaml                   # 算子前反向绑定配置文件
│   │   └── op_plugin_functions.yaml           # 算子对外接口配置文件
│   ├── ops                                    # 算子适配代码实现文件目录
│   │   ├── aclops                             # aclop算子适配目录
│   │   │   ├── AbsKernelNpu.cpp
│   │   │   └── ...
│   │   └── opapi                              # aclnn算子适配目录
│   │       ├── AbsKernelNpuOpApi.cpp
│   │       └── ...
|   ├── python
|   │   └── meta
|   │       └── _meta_registrations.py         # 算子meta实现注册文件
│   ├── OpInterface.h         	               # 编译自动生成op_plugin对外接口的头文件，用于框架侧调用算子
│   ├── OpInterface.cpp                        # 编译自动生成op_plugin对外接口路由实现，内部实现不同类型算子分支选择代码
│   ├── AclOpsInterface.h                      # 编译自动生成aclop算子插件适配所对应头文件 
│   ├── OpApiInterface.h                       # 编译自动生成aclnn算子插件适配所对应头文件
│   └── ...
├── codegen
│   └── templates
│       └── _op_plugin_docs.py                 # 算子对外接口文档
├── test
│   ├── allowlist_for_publicAPI.json           # 对外公开接口白名单
│   ├── core_tests
│   |   └── torch_npu_OpApi_schema_all.json    # 算子接口schema配置文件
|   └── test_custom_ops                        # 算子开发者测试目录
│       └── ...
```

## 算子yaml适配

> [!NOTE]  
> 以下abs的yaml配置和适配文件为已有配置和文件，此处仅为示例，用户需根据实际场景更改。

OpPlugin采用和原生PyTorch类似的逻辑在yaml中声明算子的各类信息，通过在yaml中配置算子，自动生成算子声明和注册代码。算子的Aten IR定义位于op\_plugin/config/op\_plugin\_functions.yaml文件中，所有版本的定义都在这个文件里面，通过配置不同版本来区分。


### yaml算子适配规则<a id="yaml算子适配规则"></a>

```yaml
# op_plugin_functions.yaml
all_version: [v2.1, v2.2, v2.3, v2.4, v2.5, v2.6, v2.7, v2.8, v2.9, v2.10]

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
    internal_format_opapi: all_version

# 入参带有symint的算子
symint:
  - func: zeros(SymInt[] size, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
    acl_op: [v2.1, newest]
```

参数说明：
-   all\_version：表示当前PyTorch支持的所有版本，版本列表会根据torch_npu演进调整，具体以代码为准。可通过[]设置算子支持的版本范围，例如[v2.1, newest]代表该算子支持从v2.1到最新版本。
-   official和custom：分别表示该字段下的算子为PyTorch原生和自定义算子；symint字段表明该算子支持symint类型的入参，该种算子请参考[symint算子适配](#symint算子适配)。
-   func：表示定义算子的schema(算子描述规范)，其内容完全遵循PyTorch原生Aten IR算子schema的定义规则，通过“算子名称+入参列表+返回参数”的结构化形式，完整描述算子的调用接口与语义约束。具体规则可参考[PyTorch scheme规则](reference.md#section001)。
-   acl\_op：表示在该版本支持acl\_op调用，如果支持的版本与all\_version表示的版本一致，则可以用"all\_version"表示，可选字段。
-   op\_api：表示在该版本支持op\_api调用，如果支持的版本与all\_version表示的版本一致，则可以用"all\_version"表示，可选字段。
-   gen\_opapi：对于支持op\_api调用的算子，如果适配代码简单，可以直接调用底层算子，不需要额外的适配，则可以考虑用结构化适配的方式自动生成适配代码，详见章节[结构化适配介绍(可选)](#结构化适配介绍可选)。
-   exposed：表示商用算子支持的版本，一般只用设置正向算子接口。
-   internal_format_opapi：表示支持昇腾亲和格式NZ数据分发到op_api算子调用的白名单机制。当前对于入参为昇腾亲和格式的数据默认被分发到acl_op调用；只有当算子显示添加internal_format_opapi字段并加入白名单后，才会将NZ格式数据分发到op_api调用路径。


> [!NOTE]  
> 如果存在某个算子适配有两个版本不一致，则需要两个都加上，如std.correction在PyTorch1.11.0版本和PyTorch2.1.0及以上版本的入参名称不同，则需要分开写成两个，通过version区分。<br>
>```yaml
>  - func: std.correction(Tensor self, int[1]? dim, *, int? correction, bool keepdim=False) -> Tensor
>    acl_op: v1.11
>    op_api: v1.11
>  - func: std.correction(Tensor self, int[1]? dim=None, *, Scalar? correction=None, bool keepdim=False) -> Tensor
>    acl_op: [v2.1, newest]
>    op_api: [v2.1, newest]
>```


### 自动前反向绑定算子适配

> [!NOTE]  
> 仅适用于需要进行前反向绑定的算子。

在神经网络中，前向函数用于计算输出和损失，反向函数用于计算梯度，这两个函数是互相关联的。Pytorch在执行算子操作时，不仅会执行前向计算，还会保存反向函数中的必要信息，因此需要执行算子的前反向绑定，即前向函数和反向函数的绑定。
对于原生的算子，官方已有前反向绑定逻辑，插件侧有对应前向算子和反向算子配置即可。对于自定义算子，则需要在插件侧配置前反向自动绑定。具体操作包括：
1. 实现前向和反向算子yaml适配：与[yaml算子适配规则](#yaml算子适配规则)中一致，分别适配前向算子和反向算子，并在op\_plugin\_functions.yaml中配置前向和反向算子。
2. 配置前反向绑定，将前向和反向算子进行绑定：OpPlugin与原生PyTorch一致，通过op\_plugin/config/derivatives.yaml配置算子的前反向绑定关系，同时相比原生新增了version字段用于表示支持的版本。

```yaml
# derivatives.yaml
all_version: [v2.1, v2.2, v2.3, v2.4, v2.5, v2.6, v2.7, v2.8, v2.9, v2.10]
backward:
- name: l1_loss(Tensor self, Tensor target, int reduction=Mean) -> Tensor
  self: l1_loss_backward(grad, self, target, reduction)
  target: l1_loss_backward(grad, self, target, reduction) * -1
  version: [v2.1, newest]

- name: npu_add_layer_norm(Tensor x1, Tensor x2, Tensor gamma, Tensor beta, float epsilon=1e-05, bool additional_output=False) -> (Tensor, Tensor, Tensor, Tensor)
  output_differentiability: [true, false, false, true]
  x1, x2, gamma, beta: npu_add_layer_norm_backward(grads[0], x1, x2, result2, result1, gamma, grads[1])
  version: [v2.1, newest]

- name: gather(Tensor self, int dim, Tensor index, *, bool sparse_grad=False) -> Tensor
  self: npu_gather_backward(grad, self.sym_sizes(), dim, index, sparse_grad)
  index: non_differentiable
  result: auto_linear
  version: all_version
```

参数说明：
-   name：需要前反向绑定的算子接口，同op\_plugin\_functions.yaml中函数声明。
-   self和算子接口输入参数：定义入参的梯度计算方法，对于简单的可以直接用数据公式说明，对于复杂的通过底层实现的反向函数声明。
-   output_differentiability：定义输出的可微行，通过列表的方式定义多个输出中哪些是可微的。
-   result：算子接口的返回结果。

> [!NOTE]  
> 所有版本的算子前反向绑定都在同一个derivatives.yaml里面，通过version字段来区分版本。


### symint算子适配

> [!NOTE]  
> symint类型算子需参考此部分进行适配。<br>
> 以下yaml配置和适配文件为已有配置和文件，此处仅为示例，用户需根据实际场景更改。

symint为PyTorch在v2.0及以上版本新增的数据类型，op\_plugin/config/op\_plugin\_functions.yaml配置中对应添加了symint类型。配置在symint字段下的函数表示底层函数实现支持了symint类型入参。对于底层不支持symint的函数，则无需在symint字段配置。当需要在symint字段配置时，用户进行如下操作进行算子适配:

1. 算子在配置yaml中除了在official或custom下声明函数外，还需要同时在symint下配置该算子。
2. 算子名称在原有名称上添加\_symint后缀，如配置支持入参为symint类型的zeros算子，yaml配置如下所示：

    ```yaml
    # 官方算子
    official:
     - func: zeros(SymInt[] size, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
       acl_op: all_version
    
    symint:
     - func: zeros(SymInt[] size, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
       acl_op: all_version
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

### （推荐）aclnn算子适配

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
    > [!NOTE]  
    > 更多算子适配常见API接口可参见[LINK](reference.md#section002)，更多常见宏定义可参见[LINK](reference.md#section003)。

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


#### （可选）结构化适配

> [!NOTE]  
>仅aclnn算子可使用此方法进行适配。

结构化适配指通过在op\_plugin\_functions.yaml中进行配置，自动完成算子适配实现，自动生成的适配文件位于op\_plugin/ops/opapi/StructKernelNpuOpApi.cpp。
判断是否可结构化依据：opapi对应的aclnn算子与Aten IR的语义对齐，适配层除申请output tensor，无其他适配逻辑。

YAML配置有以下两种方式，可根据实际情况进行选择。每个结构化适配的函数必须在op\_plugin\_functions.yaml中配置，具体实现如下：

- 常规场景

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
        # Aten IR定义：
        - func: func_name(ArgType arg0, ArgType arg1, ...) -> Return
        # 方式一：和输入参数相同
          size: arg0

        # 方式二：枚举每个维度的值
          size: '{4, arg0.size(0), arg0.size(1), arg1.size(0)}'

        # 方式三：条件表达式
          size: 'arg1 == 1? arg0.sizes(): at::ArrayRef<int64_t>()'

        # 方式四：在KernelNpuOutputSize.h中自定义infershape函数, 例如broadcast_ops_npu_output_size
          size: broadcast_ops_npu_output_size(arg0, arg1)
        ```

    -   dtype：配置输出tensor的dtype大小，如果大小和schema中的某个参数相同，可以配置成输入参数的名字。也可配置成自定义inferdtype函数，inferdtype函数需在KernelNpuOutputDtype.h中实现。对于out类接口，如果输出dtype不需要check，可省略此字段。配置方式主要包含以下几种：

        ```yaml
        # Aten IR定义：
        - func: func_name(ArgType arg0, ArgType arg1, ...) -> Return
        # 方式一：和输入参数相同
          dtype: arg0

        # 方式二：配置成已知的dtype类型
          dtype: at::kFloat

        # 方式三：条件表达式
          dtype: 'isIntegralType(arg0.scalar_type(), true) ? at::kFloat : arg0.scalar_type()'

        # 方式四：在KernelNpuOutputDtype.h中自定义inferdtype函数。
          dtype: inferdtype(arg0, arg1)
        ```

    -   name：输出结果涉及named tensor逻辑，可配置此字段，当前仅支持name和输入参数相同的配置，不涉及可忽略。
    -   exec：配置EXEC\_NPU\_CMD对应的参数，如果除aclnnname，其它参数顺序和Aten IR的顺序相同，可只配置aclnnname，如_aclnnAbs_。以abs为例，exec字段可以配置成下面两种方式。

        ```yaml
        - func: abs.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
          # 方式一：
          exec: aclnnAbs, self, out

          # 方式二：
          exec: aclnnAbs
        ```

- 继承场景

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


### aclop算子适配

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

> [!NOTE]  
> 以下npu_transpose的辅助适配仅为示例，用户需根据实际场景更改。

### 接口说明文档适配

在codegen/templates/\_op\_plugin\_docs.py文件补充新增接口的说明文档，一般仅设置正向算子，具体示例如下：

  ```python
  _add_torch_npu_docstr(
      "npu_transpose",
      """
  torch_npu.npu_transpose(self, perm, require_contiguous=True) -> Tensor

  功能描述
  返回原始张量视图，其维度已permute，结果连续。支持FakeTensor模式。

  参数说明
  self (Tensor)：输入张量。
  perm (ListInt)：对应维度排列。
  require_contiguous(Bool)：用户是否需要对输入Tensor做转连续。设置为False时，表示不对输入Tensor做转连续。用户明确输入Tensor为连续Tensor或转置Tensor时，才能设置为True。默认值为True。

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


### 对外公开接口适配

对外公开接口需在以下文件中新增接口配置：
  - test/allowlist\_for\_publicAPI.json
    ```json
    {
    "torch_npu": 
      {
        "all_version": ["npu_transpose"]
      }
    }
    ```

  - test/core\_tests/torch\_npu\_OpApi\_schema\_all.json
    > [!NOTE]  
    > 以"op-api:"开头的表示Python接口，以"func:"开头的接口表示C++接口。
    
      ```json
      {
        "op_api: torch_npu.npu_transpose(*args, **kwargs)": {
          "version": ["all_version"]
        },
        "func: npu_transpose(Tensor self, int[] perm, bool require_contiguous=True) -> Tensor": {
          "version": ["all_version"]
        }
      }
      ```


### meta实现适配

在fx、compile等功能使用时，需注册算子接口的meta实现，使得走faketensor时可以正常执行。目前算子的meta实现，统一注册在文件op\_plugin/python/meta/\_meta\_registrations.py。
  ```python
  @impl(m, "npu_transpose")
  def npu_transpose_meta(self, perm, require_contiguous=True):
      output = self.permute(perm)
      return torch.empty_like(output, dtype=self.dtype)
  ```