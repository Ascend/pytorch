# Adaptation Development

## Adaptation File Structure

```text
├── op_plugin
│   ├── config                                 # Operator adapter configuration file directory
│   │   ├── derivatives.yaml                   # Operator forward/backward binding configuration file
│   │   └── op_plugin_functions.yaml           # Operator external interface configuration file
│   ├── ops                                    # Operator adapter implementation file directory
│   │   ├── aclops                             # aclop operator adapter directory
│   │   │   ├── AbsKernelNpu.cpp
│   │   │   └── ...
│   │   └── opapi                              # aclnn operator adapter directory
│   │       ├── AbsKernelNpuOpApi.cpp
│   │       └── ...
│   ├── python
│   │   └── meta
│   │       └── _meta_registrations.py         # Operator meta implementation registration file
│   ├── OpInterface.h                          # Auto-generated header file for op_plugin external interfaces (for framework-side operator invocation)
│   ├── OpInterface.cpp                        # Auto-generated routing implementation for op_plugin external interfaces (internal branch selection logic for different operator types)
│   ├── AclOpsInterface.h                      # Auto-generated header file for aclop operator plugin adapter 
│   ├── OpApiInterface.h                       # Auto-generated header file for aclnn operator plugin adapter
│   └── ...
├── codegen
│   └── templates
│       └── _op_plugin_docs.py                 # Operator external interface documentation
├── test
│   ├── allowlist_for_publicAPI.json           # Public interface allowlist
│   ├── core_tests
│   │   └── torch_npu_OpApi_schema_all.json    # Operator interface schema configuration file
│   └── test_custom_ops                        # Operator developer test directory
│       └── ...
```

## Operator YAML Adaptation

> [!NOTE]  
> The `abs` YAML configuration and adaptation files shown below are existing configurations and files, provided as examples only. You need to modify them based on your actual scenarios.

OpPlugin uses logic similar to native PyTorch to declare various types of operator information in YAML. By configuring operators in YAML, operator declaration and registration code is automatically generated. The Aten IR definitions of operators are located in the `op_plugin/config/op_plugin_functions.yaml` file. Definitions of all versions are in this file, distinguished by configuring different versions.

### YAML Operator Adaptation Rules<a id="yaml 算子适配规则"></a>

```yaml
# op_plugin_functions.yaml
all_version: [v2.1, v2.2, v2.3, v2.4, v2.5, v2.6, v2.7, v2.8, v2.9, v2.10]

# Native operators
official:
  - func: abs(Tensor self) -> Tensor
    acl_op: all_version
    op_api: all_version
    gen_opapi:
      structured_inherit: abs.out

# Custom operators
custom:
  - func: my_abs(Tensor self) -> Tensor
    acl_op: all_version
    op_api: all_version
    exposed: all_version
    internal_format_opapi: all_version

# Operators with symint input parameters
symint:
  - func: zeros(SymInt[] size, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
    acl_op: [v2.1, newest]
```

Parameters

- `all_version`: Indicates all versions supported by the current PyTorch. The version list will be adjusted as torch_npu evolves, and the actual code shall prevail. You can use `[]` to specify the version range supported by an operator. For example, `[v2.1, newest]` indicates that the operator supports versions from v2.1 to the latest.
- `official` and `custom`: Respectively indicate that operators under these fields are PyTorch native operators and custom operators. The `symint` field indicates that the operator supports symint-type input parameters. For such operators, refer to [Symint Operator Adaptation](#symint-operator-adaptation).
- `func`: Defines the operator schema (operator description specification). Its content fully follows the schema definition rules of PyTorch native Aten IR operators and completely describes the operator calling interface and semantic constraints through the structured form of "operator name + input parameter list + return parameter". For specific rules, refer to [PyTorch schema rules](reference.md#pytorch-schema-rules).
- `acl_op`: Indicates that acl_op calls are supported in this version. If the supported versions are the same as the versions indicated by `all_version`, you can use `all_version` instead. This is an optional field.
- `op_api`: Indicates that op_api calls are supported in this version. If the supported versions are the same as the versions indicated by `all_version`, you can use `all_version` instead. This is an optional field.
- `gen_opapi`: For operators that support op_api calls, if the adaptation code is simple and the underlying operator can be called directly without additional adaptation, you can consider using structured adaptation to automatically generate the adaptation code. For details, see [(Optional) Structured Adaptation](#optional-structured-adaptation).
- `exposed`: Indicates the versions supported by commercial operators. Generally, only the forward operator interface needs to be set.
- `internal_format_opapi`: Indicates the allowlist mechanism that supports dispatching data in the Ascend affinity format NZ to op_api operator calls. Currently, data whose input parameters are in the Ascend affinity format is dispatched to acl_op calls by default. Only when the operator explicitly adds the `internal_format_opapi` field and is added to the allowlist will NZ format data be dispatched to the op_api call path.

> [!NOTE]  
> If an operator adaptation has two versions with inconsistencies, you need to add both. For example, the parameter names of `std.correction` differ between PyTorch 1.11.0 and PyTorch 2.1.0 and later. In this case, you need to write them as two separate entries and distinguish them by `version`.<br>
>
>```yaml
>  - func: std.correction(Tensor self, int[1]? dim, *, int? correction, bool keepdim=False) -> Tensor
>    acl_op: v1.11
>    op_api: v1.11
>  - func: std.correction(Tensor self, int[1]? dim=None, *, Scalar? correction=None, bool keepdim=False) -> Tensor
>    acl_op: [v2.1, newest]
>    op_api: [v2.1, newest]
>```

### Automatic Forward-Backward Binding Operator Adaptation

> [!NOTE]  
> This applies only to operators that require forward-backward binding.

In neural networks, the forward function computes the output and loss, and the backward function computes the gradients. These two functions are interrelated. When PyTorch executes an operator, it not only performs the forward computation but also saves the necessary information in the backward function. Therefore, the forward-backward binding of the operator, that is, binding the forward function and the backward function, is required.
For native operators, the official forward-backward binding logic already exists, and the plugin side only needs the corresponding forward and backward operator configurations. For custom operators, you need to configure automatic forward-backward binding on the plugin side. The specific operations are as follows:

1. Implement YAML adaptation for the forward and backward operators: Consistent with [YAML Operator Adaptation Rules](#yaml-operator-adaptation-rules), adapt the forward and backward operators separately, and configure the forward and backward operators in `op_plugin_functions.yaml`.
2. Configure forward-backward binding to bind the forward and backward operators: Consistent with native PyTorch, OpPlugin configures the forward-backward binding relationships of operators through `op_plugin/config/derivatives.yaml`. Compared with native PyTorch, the `version` field is additionally added to indicate the supported version.

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

Parameters

- `name`: The operator interface that requires forward-backward binding, same as the function declaration in `op_plugin_functions.yaml`.
- `self` and operator interface input parameters: Define the gradient computation method for input parameters. Simple cases can be described directly using mathematical formulas, while complex cases are declared through the backward functions implemented at the lower layer.
- `output_differentiability`: Defines the differentiability of outputs, using a list to define which of the multiple outputs are differentiable.
- `result`: The return result of the operator interface.

> [!NOTE]  
> Forward-backward bindings of operators of all versions are defined in the same `derivatives.yaml`, with different versions distinguished by the `version` field.

### Symint Operator Adaptation

> [!NOTE]  
> Operators of the symint type must be adapted by referring to this section.<br>
> The following YAML configuration and adaptation files are existing configurations and files, provided as examples only. You need to modify them based on your actual scenarios.

symint is a data type newly added in PyTorch v2.0 and later. Correspondingly, the `symint` type is added to the configuration in `op_plugin/config/op_plugin_functions.yaml`. Functions configured under the `symint` field indicate that the underlying function implementation supports symint-type input parameters. Functions whose underlying implementation does not support symint do not need to be configured under the `symint` field. When configuration under the `symint` field is required, you need to perform the following operations to adapt the operator:

1. In the YAML configuration, in addition to declaring the function under `official` or `custom`, you also need to configure the operator under `symint`.
2. Add the `_symint` suffix to the original operator name. For example, to configure a `zeros` operator that supports symint-type input parameters, the YAML configuration is as follows:

    ```yaml
    # Official operator
    official:
     - func: zeros(SymInt[] size, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
       acl_op: all_version
    
    symint:
     - func: zeros(SymInt[] size, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
       acl_op: all_version
    ```

3. In the same file as the operator implementation, add an operator named `zeros_symint`, with the first input parameter being of the symint-related type `c10::SymIntArrayRef`. Because the symint feature is supported only in PyTorch 2.0 and later, the symint-related adaptation code needs to add the version compilation macro `VERSION_BETWEEN` to control compilation based on the actual version support.

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

## Operator Code Adaptation

Currently, two types of operators are supported for adaptation: aclnn operators and aclop operators. aclnn operators are a newer operator implementation method and are recommended. Their adaptation files are located in the `op_plugin/ops/opapi` directory. aclop operators are an earlier operator implementation method and are not recommended. Their adaptation files are located in the `op_plugin/ops/aclops` directory.
The adaptation code of all versions of an operator is placed in the same file, with different versions distinguished by the compilation macro `VERSION_BETWEEN`.
To add a custom operator, you need to create the corresponding operator adaptation file and develop the operator implementation by referring to the following examples.

### (Recommended) aclnn Operator Adaptation

#### General Adaptation

The adaptation file path for aclnn operators is `op_plugin/ops/opapi/AbsKernelNpuOpApi.cpp`. The file naming convention is Operator Name + `KernelNpuOpApi`, with the first letter of the operator name capitalized. The specific steps are as follows:

1. Create the operator adaptation file and include the dependency header files, including the aclnn external interface, operator declarations, base functions, and so on.
2. Implement the operator interface adaptation. aclnn operators must be defined in the `op_api` namespace, and the input and output parameters of the implementation function must be consistent with the torch API.

    ```cpp
    // Operator adaptation implementation file path: op_plugin/ops/opapi/AbsKernelNpuOpApi.cpp 
    // 1. Include dependency header files
    // Header file for external interfaces, containing the function prototypes of all ACLNN operators exposed by op_plugin
    #include "op_plugin/OpApiInterface.h" 
    // Header file for aclop operator declarations
    #include "op_plugin/AclOpsInterface.h" 
    // Header file of the base functions that torch depends on when calling ACLNN operators
    #include "op_plugin/utils/op_api_common.h" 

    // 2. Implement the operator interface adaptation
    // ACLNN operators are defined in the op_api namespace
    namespace op_api { 
    using npu_preparation = at_npu::native::OpPreparation; 

    // Implementation function of the abs API. The name is unique and the parameters are the same as those of the torch API 
    at::Tensor abs(const at::Tensor& self)
    { 
        DO_COMPATIBILITY(aclnnAbs, acl_op::abs(self)); 

        // Construct the NPU output tensor
        at::Tensor result = npu_preparation::apply_tensor_without_format(self); 

        // Compute the NPU output result
        EXEC_NPU_CMD(aclnnAbs, self, result); 
        return result; 
    } 
    // Location where abs variants are added
    }  // namespace op_api
    ```

    > [!NOTE]  
    > For more common API interfaces used in operator adaptation, see [LINK](reference.md#common-api-interfaces-for-operator-adaptation). For more common macro definitions, see [LINK](reference.md#common-macro-definitions-for-operator-adaptation).

3. If the interface contains multiple variants, such as a variant with an `out` parameter or an in-place operation variant, you need to add the corresponding adaptation code. Refer to the following for adaptation:

    - Variant with an `out` parameter:

      ```cpp
      // Implementation function of the abs_out API. The name is unique and the parameters are the same as those of the torch API
      at::Tensor& abs_out(const at::Tensor& self, at::Tensor& result)
      { 
          // Search for the ACLNN operator implementation. If not found, use the ACLOP operator implementation
          DO_COMPATIBILITY(aclnnAbs, acl_op::abs_out(self, result)); 
          npu_preparation::check_tensor({self}, result, self); 
          // Asynchronously call the NPU for execution
          EXEC_NPU_CMD(aclnnAbs, self, result); 
          return result; 
      }
      ```

    - In-place operation variant:

      ```cpp
      // Implementation function of the abs_ API. The name is unique and the parameters are the same as those of the torch API. This interface is an in-place operation, that is, the output result is stored in the input tensor
      at::Tensor& abs_(at::Tensor& self)
      {
          DO_COMPATIBILITY(aclnnAbs, acl_op::abs_(self));
          op_api::abs_out(self, self);
          return self;
      }
      ```

4. If the adaptation code differs between versions, all the code is placed in the same file and distinguished by compilation macros.

    ```cpp
    #include "op_plugin/AclOpsInterface.h"
    #include "op_plugin/OpApiInterface.h"
    #include "op_plugin/utils/op_api_common.h"
    namespace op_api {
    using npu_preparation = at_npu::native::OpPreparation;
    // The function input parameters of version 1.11 differ from those of version 2.0 and later, so a separate implementation is required. Therefore, the macro is used to control it
    #if VERSION_BETWEEN(V1R11, V1R11)
    at::Tensor embedding(const at::Tensor& weight, const at::Tensor& indices, int64_t padding_idx, bool scale_grad_by_freq, bool sparse)
    {
        DO_COMPATIBILITY(aclnnEmbedding, acl_op::embedding(weight, indices, padding_idx, scale_grad_by_freq, sparse));
        // Compute the size of the output tensor
        auto output_size = op_infer::array_to_small_vector(indices.sizes());
        output_size.emplace_back(weight.size(weight.dim() - 1));
        // Construct the NPU output tensor
        at::Tensor result = npu_preparation::apply_tensor_without_format(output_size, weight.options());
        // Compute the NPU output result
        EXEC_NPU_CMD(aclnnEmbedding, weight, indices, result);
        return result;
    }
    #endif

    #if VERSION_BETWEEN(V2R0, VERSION_NEWEST)
    at::Tensor embedding_symint(const at::Tensor& weight, const at::Tensor& indices, c10::SymInt padding_idx, bool scale_grad_by_freq, bool sparse)
    {
        DO_COMPATIBILITY(aclnnEmbedding, acl_op::embedding_symint(weight, indices, padding_idx, scale_grad_by_freq, sparse));
        // Compute the size of the output tensor
        auto output_size = op_infer::array_to_small_vector(indices.sizes());
        output_size.emplace_back(weight.size(weight.dim() - 1));
        // Construct the NPU output tensor
        at::Tensor result = npu_preparation::apply_tensor_without_format(output_size, weight.options());
        // Compute the NPU output result
        EXEC_NPU_CMD(aclnnEmbedding, weight, indices, result);
        return result;
    }
    #endif
    } // namespace op_api
    ```

#### (Optional) Structured Adaptation

> [!NOTE]  
> Only aclnn operators can use this method for adaptation.

Structured adaptation refers to automatically completing the operator adaptation implementation through configuration in `op_plugin_functions.yaml`. The automatically generated adaptation file is located at `op_plugin/ops/opapi/StructKernelNpuOpApi.cpp`.
The criterion for determining whether structured adaptation is applicable is that the aclnn operator corresponding to opapi is semantically aligned with the Aten IR, and the adaptation layer has no adaptation logic other than allocating the output tensor.

There are two YAML configuration methods, and you can choose either based on the actual situation. Each function that requires structured adaptation must be configured in `op_plugin_functions.yaml`. The specific implementation is as follows:

- Regular scenarios

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

    The fields are described as follows:

    - `gen_opapi`: Indicates that the corresponding API supports structured adaptation. Other fields must be configured under this field.
    - `out`: Represents the output of the function. This field contains the `size` and `dtype` fields. If there are multiple outputs, you can configure them as out0, out1, and so on. For out-type interfaces, this field cannot be customized and must be the same as the output parameter name defined in the Aten IR. For inplace-type interfaces, this field does not need to be configured.
    - `size`: Configures the shape of the output tensor. If the size is the same as a parameter in the schema, it can be configured as the name of that input parameter. You can also configure it as a custom infershape function, which must be implemented in `KernelNpuOutputSize.h`. For out-type interfaces, if the output shape remains unchanged, this field can be omitted. The configuration methods mainly include the following:

        ```yaml
        # Aten IR definition:
        - func: func_name(ArgType arg0, ArgType arg1, ...) -> Return
        # Method 1: Same as the input parameter
          size: arg0

        # Method 2: Enumerate the value of each dimension
          size: '{4, arg0.size(0), arg0.size(1), arg1.size(0)}'

        # Method 3: Conditional expression
          size: 'arg1 == 1? arg0.sizes(): at::ArrayRef<int64_t>()'

        # Method 4: Define a custom infershape function in KernelNpuOutputSize.h, for example, broadcast_ops_npu_output_size
          size: broadcast_ops_npu_output_size(arg0, arg1)
        ```

    - `dtype`: Configures the dtype of the output tensor. If the dtype is the same as a parameter in the schema, it can be configured as the name of that input parameter. You can also configure it as a custom inferdtype function, which must be implemented in `KernelNpuOutputDtype.h`. For out-type interfaces, if the output dtype does not need to be checked, this field can be omitted. The configuration methods mainly include the following:

        ```yaml
        # Aten IR definition:
        - func: func_name(ArgType arg0, ArgType arg1, ...) -> Return
        # Method 1: Same as the input parameter
          dtype: arg0

        # Method 2: Configure it as a known dtype type
          dtype: at::kFloat

        # Method 3: Conditional expression
          dtype: 'isIntegralType(arg0.scalar_type(), true) ? at::kFloat : arg0.scalar_type()'

        # Method 4: Define a custom inferdtype function in KernelNpuOutputDtype.h
          dtype: inferdtype(arg0, arg1)
        ```

    - `name`: When the output result involves named tensor logic, you can configure this field. Currently, only configurations where the name is the same as an input parameter are supported. You can ignore it if it is not involved.
    - `exec`: Configures the parameters corresponding to `EXEC_NPU_CMD`. If the order of the other parameters, except `aclnnname`, is the same as that in the Aten IR, only `aclnnname` needs to be configured, such as `_aclnnAbs_`. Taking abs as an example, the exec field can be configured in the following two ways.

        ```yaml
        - func: abs.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
          # Method 1:
          exec: aclnnAbs, self, out

          # Method 2:
          exec: aclnnAbs
        ```

- Inheritance scenario

    ```yaml
    - func: func_name(ArgType arg0[=default], ArgType arg1[=default], ...) -> Return
      op_api: v2.7
      gen_opapi:
        structured_inherit: func_name.out
    ```

    `structured_inherit`: If the field configuration of the original function or an inplace-type interface is the same as that of an out-type interface, you can use this field to inherit the corresponding out-type interface.

    Taking abs as an example, the out attributes and exec of the original function and the out-type function are the same, so they can be inherited through the `structured_inherit` field.

    ```yaml
      - func: abs(Tensor self) -> Tensor
        op_api: [v1.11, newest]
        gen_opapi:
          structured_inherit: abs.out
    ```

### aclop Operator Adaptation

> [!NOTE]
>
> <term>Ascend 950DT</term> does not currently support aclop operator adaptation.

aclop operators are an earlier operator implementation method and are not recommended. The adaptation file path is `op_plugin/ops/aclops/AbsKernelNpu.cpp`. The file naming convention is Operator Name + `KernelNpu`, with the first letter of the operator name capitalized. The specific steps are as follows:

1. Create the operator adaptation file and include the dependency header files, including the aclop external interface, operator declarations, base functions, and so on.
2. Implement the operator interface adaptation. aclop operators must be defined in the `acl_op` namespace, and the input and output parameters of the implementation function must be consistent with the torch API.

    ```cpp
    // Operator adaptation implementation file path: op_plugin/ops/aclops/AbsKernelNpu.cpp 
    // 1. Include dependency header files
    // Header file for external interfaces, containing the function prototypes of all aclop operators exposed by op_plugin
    #include "op_plugin/AclOpsInterface.h" 
    // Header file of the base functions that torch depends on when calling ACLOP operators
    #include "op_plugin/utils/OpAdapter.h" 
    
    // 2. Implement the operator interface adaptation
    // CANN operators are defined in the acl_op namespace
    namespace acl_op { 
    using npu_preparation = at_npu::native::OpPreparation; 
    using npu_utils = at_npu::native::NpuUtils; 
    // Interfaces that are not exposed to the outside are all defined in the anonymous namespace. Common examples are xx_nocheck and so on, which directly call ACLOP operators without memory or shape validation 
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
    
    // Implementation function of the abs API. The name is unique and the parameters are the same as those of the torch API 
    at::Tensor abs(const at::Tensor& self)
    { 
        // Construct the output tensor and call the ACLOP operator 
        auto output_size = op_infer::infershape_for_elewise(self); 
        at::Tensor result = npu_preparation::apply_tensor(self, output_size); 
        abs_out_nocheck(result, self); 
        return result; 
    }
    // Location where abs variants are added
    } // namespace acl_op
    ```

3. If the interface contains multiple variants, such as a variant with an `out` parameter or an in-place operation variant, you need to add the corresponding adaptation code. Refer to the following for adaptation:

    - Variant with an `out` parameter:

      ```cpp
      // Implementation function of the abs_out API. The name is unique and the parameters are the same as those of the torch API
      at::Tensor& abs_out(const at::Tensor& self, at::Tensor& result)
      { 
          // CheckOut is used to check whether the size and dtype of result meet the expectations. If the dtype does not meet the expectations, an error is raised. If the size does not meet the expectations, a resize operation is performed
          npu_preparation::CheckOut({self}, result, self); 
          // check_match is used to check whether result is contiguous. Because the ACLOP operator cannot support non-contiguous output, special handling is required when result is non-contiguous
          if (!npu_utils::check_match(&result)) { 
              // If result is non-contiguous, create a contiguous tensor (contig_tensor) to receive the output of the ACLOP operator (abs). Then copy contig_tensor to the original output result
              at::Tensor contiguous_result = npu_utils::format_contiguous(result); 
              abs_out_nocheck(ccontiguous_result, self); 
              npu_utils::format_fresh_view(result, contiguous_result); 
          } else { 
              // If result is contiguous, directly call the ACLOP operator
              abs_out_nocheck(result, self); 
          } 
          return result; 
      }
      ```

    - In-place operation variant:

      ```cpp
      // Implementation function of the abs_ API. The name is unique and the parameters are the same as those of the torch API. This interface is an in-place operation, that is, the output result is stored in the input tensor
      at::Tensor& abs_(at::Tensor& self)
      {
          // Call the out interface to prevent incorrect results when self is used as the output and the ACLOP operator is directly called in a non-contiguous scenario
          return acl_op::abs_out(self, self);
      }
      ```

4. If the adaptation code differs between versions, all the code is placed in the same file and distinguished by compilation macros.

    ```cpp
    #include "op_plugin/AclOpsInterface.h"
    #include "op_plugin/utils/custom_functions/aclops/inner_compute.h"
    namespace acl_op {
    // The function input parameters of version 1.11 differ from those of version 2.0 and later, so the macro is used to control them
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

    // The code for version 2.0 and later is consistent
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

## Auxiliary Operator Adaptation

> [!NOTE]  
> The following auxiliary adaptation for npu_transpose is provided as an example only. You need to modify it based on your actual scenarios.

### Interface Documentation Adaptation

Add the documentation of the new interface in the `codegen/templates/_op_plugin_docs.py` file. Generally, only the forward operator is set. The specific example is as follows:

  ```python
  _add_torch_npu_docstr(
      "npu_transpose",
      """
  torch_npu.npu_transpose(self, perm, require_contiguous=True) -> Tensor

  Function description
  Returns a view of the original tensor with its dimensions permuted, and the result is contiguous. The FakeTensor mode is supported.

  Parameter description
  self (Tensor): Input tensor.
  perm (ListInt): Corresponding dimension permutation.
  require_contiguous(Bool): Indicates whether the user needs to convert the input tensor to contiguous. If it is set to False, the input tensor is not converted to contiguous. You can set it to True only when you confirm that the input tensor is a contiguous tensor or a transposed tensor. The default value is True.

  Example
  >>> x = torch.randn(2, 3, 5).npu()
  >>> x.shape
  torch.Size([2, 3, 5])
  >>> x1 = torch_npu.npu_transpose(x, (2, 0, 1))
  >>> x1.shape
  torch.Size([5, 2, 3])
  """
  )
  ```

### Public Interface Adaptation

To expose a public interface, you need to add the interface configuration in the following files:

  - `test/allowlist_for_publicAPI.json`

    ```json
    {
    "torch_npu": 
      {
        "all_version": ["npu_transpose"]
      }
    }
    ```

  - `test/core_tests/torch_npu_OpApi_schema_all.json`
    > [!NOTE]  
    > Entries prefixed with "op-api:" represent Python interfaces, while entries prefixed with "func:" represent C++ interfaces.
    
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

### Meta Implementation Adaptation

When using features such as fx and compile, you need to register the meta implementation of the operator interface so that it can run normally when using fake tensors. Currently, the meta implementations of operators are uniformly registered in the file `op_plugin/python/meta/_meta_registrations.py`.

  ```python
  @impl(m, "npu_transpose")
  def npu_transpose_meta(self, perm, require_contiguous=True):
      output = self.permute(perm)
      return torch.empty_like(output, dtype=self.dtype)
  ```
