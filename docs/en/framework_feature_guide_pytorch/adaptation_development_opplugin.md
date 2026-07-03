# Adaptation Development

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-06-15T07:48:46.733Z pushedAt=2026-06-15T12:00:44.062Z -->

## Adaptation File Structure

```text
├── op_plugin
│   ├── config                                 # Operator adaptation configuration file directory
│   │   ├── derivatives.yaml                   # Operator forward/backward binding configuration file
│   │   └── op_plugin_functions.yaml           # Operator external interface configuration file
│   ├── ops                                    # Operator adaptation code implementation file directory
│   │   ├── aclops                             # aclop operator adaptation directory
│   │   │   ├── AbsKernelNpu.cpp
│   │   │   └── ...
│   │   └── opapi                              # aclnn operator adaptation directory
│   │       ├── AbsKernelNpuOpApi.cpp
│   │       └── ...
|   ├── python
|   │   └── meta
|   │       └── _meta_registrations.py         # Operator meta implementation registration file
│   ├── OpInterface.h                         # Header file automatically generated at compile time for op_plugin external interfaces, used by the framework side to call operators
│   ├── OpInterface.cpp                        # Auto-generated op_plugin external interface routing implementation, containing branch selection code for different operator types internally
│   ├── AclOpsInterface.h                      # Auto-generated header file for aclop operator plugin adaptation
│   ├── OpApiInterface.h                       # Auto-generated header file for aclnn operator plugin adaptation
│   └── ...
├── codegen
│   └── templates
│       └── _op_plugin_docs.py                 # Operator external interface documentation
├── test
│   ├── allowlist_for_publicAPI.json           # Public API allowlist
│   ├── core_tests
│   |   └── torch_npu_OpApi_schema_all.json    # Operator interface schema configuration file
|   └── test_custom_ops                        # Operator developer test directory
│       └── ...
```

## Operator YAML Adaptation

> [!NOTE]  
> The following YAML configuration and adaptation files for `abs` are existing configurations and files, provided here only as examples. You should modify them based on actual scenarios.

OpPlugin uses logic similar to native PyTorch to declare various operator information in YAML. By configuring operators in YAML, operator declaration and registration code is automatically generated. The Aten IR definitions for operators are located in the `op_plugin/config/op_plugin_functions.yaml` file. All version definitions are in this file, distinguished by configuring different versions.

### YAML Operator Adaptation Rules

```yaml
# op_plugin_functions.yaml
all_version: [v2.1, v2.2, v2.3, v2.4, v2.5, v2.6, v2.7, v2.8, v2.9, v2.10]

# native operator
official:
  - func: abs(Tensor self) -> Tensor
    acl_op: all_version
    op_api: all_version
    gen_opapi:
      structured_inherit: abs.out

# custom operator
custom:
  - func: my_abs(Tensor self) -> Tensor
    acl_op: all_version
    op_api: all_version
    exposed: all_version
    internal_format_opapi: all_version

# operator with symint in parameters
symint:
  - func: zeros(SymInt[] size, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
    acl_op: [v2.1, newest]
```

Parameter description:

- `all_version`: Indicates all versions currently supported by PyTorch. The version list will be adjusted as torch_npu evolves, and the specific code shall prevail. You can use [] to set the version range supported by the operator. For example, [v2.1, newest] means the operator supports versions from v2.1 to the latest.
- `official` and `custom`: Respectively indicate that the operators under this field are PyTorch native and custom operators. The symint field indicates that the operator supports symint-type input parameters. For such operators, refer to [symint Operator Adaptation](#symint-operator-adaptation).
- `func`: Indicates the schema (operator description specification) that defines the operator. Its content fully follows the definition rules of the PyTorch native Aten IR operator schema, and completely describes the operator's calling interface and semantic constraints through the structured form of "operator name + input parameter list + return parameter". For specific rules, refer to [PyTorch Schema Rules](reference.md#pytorch-schema-rules).
- `acl_op`: Indicates that `acl_op` calls are supported in this version. If the supported version is consistent with the version indicated by `all_version`, `all_version` can be used. This is an optional field.
- `op_api`: Indicates that `op_api` calls are supported in this version. If the supported version is consistent with the version indicated by `all_version`, `all_version` can be used. This is an optional field.
- `gen_opapi`: For operators that support `op_api` calls, if the adaptation code is simple and can directly call the underlying operator without additional adaptation, you can consider using structured adaptation to automatically generate the adaptation code. For details, see the [(Optional) Structured Adaptation](#optional-structured-adaptation) section.
- `exposed`: Indicates the versions supported by the commercial operator. Generally, only the forward operator interface needs to be set.
- `internal_format_opapi`: Indicates the allowlist mechanism that supports distributing Ascend affinity format NZ data to `op_api` operator calls. Currently, input parameters in Ascend affinity format are distributed to `acl_op` calls by default. Only when the operator explicitly adds the `internal_format_opapi` field and is added to the allowlist will NZ format data be distributed to the `op_api` call path.

> [!NOTE]  
> If there are two versions of an operator adaptation that are inconsistent, both need to be added. For example, the input parameter names of `std.correction` differ between PyTorch 1.11.0 and PyTorch 2.1.0 and above, so they need to be written separately and distinguished by version.
>
>```yaml
>  - func: std.correction(Tensor self, int[1]? dim, *, int? correction, bool keepdim=False) -> Tensor
>    acl_op: v1.11
>    op_api: v1.11
>  - func: std.correction(Tensor self, int[1]? dim=None, *, Scalar? correction=None, bool keepdim=False) -> Tensor
>    acl_op: [v2.1, newest]
>    op_api: [v2.1, newest]
>```

### Automatic Forward-Backward Binding

> [!NOTE]  
> Only applicable to operators that require forward-backward binding.

In neural networks, the forward function is used to compute outputs and losses, while the backward function is used to compute gradients. These two functions are interrelated. When PyTorch executes an operator, it not only performs the forward computation but also saves the necessary information for the backward function. Therefore, forward-backward binding of the operator, that is, binding the forward function and the backward function, must be performed.
For native operators, the official forward-backward binding logic already exists, and the plugin side only needs to configure the corresponding forward and backward operators. For custom operators, forward-backward automatic binding must be configured on the plugin side. The specific steps include:

1. Implement YAML adaptation for forward and backward operators: Consistent with the [YAML Operator Adaptation Rules](#yaml-operator-adaptation-rules), adapt the forward operator and backward operator separately, and configure the forward and backward operators in `op_plugin_functions.yaml`.
2. Configure forward-backward binding to bind the forward and backward operators: OpPlugin, consistent with native PyTorch, configures the forward-backward binding relationship of operators through `op_plugin/config/derivatives.yaml`. Additionally, compared to the native version, a version field has been added to indicate the supported version.

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

Parameter description:

- `name`: The operator interface that requires forward-backward binding, same as the function declaration in `op_plugin_functions.yaml`.
- `self` and operator interface input parameters: Define the gradient computation method for the input parameters. For simple cases, it can be directly explained using mathematical formulas. For complex cases, it is declared through the backward function implemented at the lower level.
- `output_differentiability`: Defines the differentiability of the outputs, specifying which of the multiple outputs are differentiable through a list.
- `result`: The return result of the operator interface.

> [!NOTE]  
> The forward-backward bindings for all versions of the operator are in the same `derivatives.yaml`, with versions distinguished by the version field.

### symint Operator Adaptation

> [!NOTE]  
> Operators of symint type should be adapted by referring to this section.
> The following YAML configuration and adaptation files are existing configurations and files, provided here only as examples. You need to modify them according to actual scenarios.

symint is a data type newly added in PyTorch v2.0 and above. The corresponding symint type has been added to the `op_plugin/config/op_plugin_functions.yaml` configuration. Functions configured under the symint field indicate that the underlying function implementation supports symint type input parameters. For functions whose underlying implementation does not support symint, there is no need to configure them in the symint field. When configuration in the symint field is required, users should perform the following steps for operator adaptation:

1. In addition to declaring the function under official or custom in the configuration yaml, the operator must also be configured under symint at the same time.
2. Append the `_symint` suffix to the original operator name. For example, to configure the zeros operator that supports symint type input parameters, the yaml configuration is as follows:

    ```yaml
    # Official operator
    official:
     - func: zeros(SymInt[] size, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
       acl_op: all_version
    
    symint:
     - func: zeros(SymInt[] size, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
       acl_op: all_version
    ```

3. In the same operator file, add a new operator named `zeros_symint`, where the first parameter type is the symint-related type `c10::SymIntArrayRef`. Since the symint feature is only supported in PyTorch 2.0 and above, symint-related adaptation code must be controlled by adding the version compilation macro `VERSION_BETWEEN` based on the actual version support.

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

Currently, adaptation is supported for two types of operators: aclnn-based operators and aclop-based operators. aclnn operators are the newer implementation method and are recommended. Their adaptation files are located in the `op_plugin/ops/opapi` directory. aclop operators are an earlier implementation method and are not recommended; their adaptation files are located in the `op_plugin/ops/aclops` directory.
All version adaptation code for a single operator resides in one file, with different versions distinguished by the compilation macro `VERSION_BETWEEN`.
When adding a custom operator, you must synchronously create an operator adaptation file and develop the relevant operator implementation by referring to the following examples.

### (Recommended) aclnn Operator Adaptation

#### General Adaptation

The adaptation file path for aclnn operators is: `op_plugin/ops/opapi/AbsKernelNpuOpApi.cpp`. The file naming convention is `OperatorName + KernelNpuOpApi`, with the first letter of the operator name capitalized. The specific steps are as follows:

1. Create the operator adaptation file and include the required dependency header files, including the aclnn external interface, operator declarations, base functions, etc.
2. Implement the operator interface adaptation. aclnn operators must be defined within the `op_api` namespace, and the function parameters and return values must be consistent with the torch API.

    ```cpp
    //Operator adaptation implementation path: op_plugin/ops/opapi/AbsKernelNpuOpApi.cpp
    // 1. Include dependency header files
    // External interface header file, containing the function prototypes of all ACLNN operators exposed by op_plugin
    #include "op_plugin/OpApiInterface.h" 
    // Include the ACLOP operator declaration header file
    #include "op_plugin/AclOpsInterface.h" 
    // Header file corresponding to the base functions that torch depends on when calling ACLNN operators
    #include "op_plugin/utils/op_api_common.h" 

    // 2. Operator interface adaptation implementation
    // ACLNN operators are defined in the op_api namespace
    namespace op_api { 
    using npu_preparation = at_npu::native::OpPreparation; 

    // abs API implementation function, with a unique name and parameters consistent with the torch API.
    at::Tensor abs(const at::Tensor& self)
    { 
        DO_COMPATIBILITY(aclnnAbs, acl_op::abs(self)); 

        // Construct the NPU output tensor
        at::Tensor result = npu_preparation::apply_tensor_without_format(self); 

        // Compute the NPU output result
        EXEC_NPU_CMD(aclnnAbs, self, result); 
        return result; 
    } 
    //Location for adding abs variant code
    }  // namespace op_api
    ```

    > [!NOTE]  
    > For more common API interfaces for operator adaptation, see the [reference](reference.md#common-api-interfaces-for-operator-adaptation). For more common macro definitions, see the [reference](reference.md#common-macro-definitions-for-operator-adaptation).

3. If the interface contains multiple variants, such as variants with an `out` parameter or in-place operation variants, you need to add the corresponding adaptation code. Refer to the following for adaptation:

    - Variant with `out` parameter:

      ```cpp
      // abs_out API implementation function, with a unique name and parameters consistent with the torch API
      at::Tensor& abs_out(const at::Tensor& self, at::Tensor& result)
      { 
          // Look up the ACLNN operator implementation; if the lookup fails, use the ACLOP operator implementation
          DO_COMPATIBILITY(aclnnAbs, acl_op::abs_out(self, result)); 
          npu_preparation::check_tensor({self}, result, self); 
          // Asynchronously call NPU for execution
          EXEC_NPU_CMD(aclnnAbs, self, result); 
          return result; 
      }
      ```

    - In-place operation variant:

      ```cpp
      // abs_ API implementation function, with a unique name and parameters consistent with the torch API. This interface is an in-place operation, meaning the output result is stored in the input tensor
      at::Tensor& abs_(at::Tensor& self)
      {
          DO_COMPATIBILITY(aclnnAbs, acl_op::abs_(self));
          op_api::abs_out(self, self);
          return self;
      }
      ```

4. If the adaptation code differs between versions, all code is placed in the same file and distinguished using compilation macros.

    ```cpp
    #include "op_plugin/AclOpsInterface.h"
    #include "op_plugin/OpApiInterface.h"
    #include "op_plugin/utils/op_api_common.h"
    namespace op_api {
    using npu_preparation = at_npu::native::OpPreparation;
    // The function parameters for 1.11 differ from those for 2.0 and above, requiring separate implementation, so macros are used to control this
    #if VERSION_BETWEEN(V1R11, V1R11)
    at::Tensor embedding(const at::Tensor& weight, const at::Tensor& indices, int64_t padding_idx, bool scale_grad_by_freq, bool sparse)
    {
        DO_COMPATIBILITY(aclnnEmbedding, acl_op::embedding(weight, indices, padding_idx, scale_grad_by_freq, sparse));
        // Calculate the size of the output tensor
        auto output_size = op_infer::array_to_small_vector(indices.sizes());
        output_size.emplace_back(weight.size(weight.dim() - 1));
        // Construct the NPU output tensor
        at::Tensor result = npu_preparation::apply_tensor_without_format(output_size, weight.options());
        // Calculate the NPU output result
        EXEC_NPU_CMD(aclnnEmbedding, weight, indices, result);
        return result;
    }
    #endif

    #if VERSION_BETWEEN(V2R0, VERSION_NEWEST)
    at::Tensor embedding_symint(const at::Tensor& weight, const at::Tensor& indices, c10::SymInt padding_idx, bool scale_grad_by_freq, bool sparse)
    {
        DO_COMPATIBILITY(aclnnEmbedding, acl_op::embedding_symint(weight, indices, padding_idx, scale_grad_by_freq, sparse));
        // Calculate the size of the output tensor
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

Structured adaptation refers to automatically completing the operator adaptation implementation by configuring in `op_plugin_functions.yaml`. The automatically generated adaptation file is located at `op_plugin/ops/opapi/StructKernelNpuOpApi.cpp`.
The criterion for determining whether structured adaptation is applicable: the aclnn operator corresponding to opapi is semantically aligned with Aten IR, and the adaptation layer has no other adaptation logic except for applying for the output tensor.

There are two ways to configure YAML, which can be selected based on the actual situation. Each structured adaptation function must be configured in `op_plugin_functions.yaml`. The specific implementation is as follows:

- Regular scenario

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

    The field meanings are as follows:

  - gen_opapi: Indicates that the corresponding API can be structured, and other fields need to be configured under this field.
  - out: Indicates the output of the function. This field contains the size and dtype fields. If there are multiple outputs, they can be configured as out0, out1, etc. For out-type interfaces, this field cannot be customized and must match the output parameter name defined in Aten IR. For inplace-type interfaces, this field does not need to be configured.
  - size: Configures the shape of the output tensor. If the size is the same as a parameter in the schema, it can be configured as the name of the input parameter. It can also be configured as a custom infershape function, which must be implemented in KernelNpuOutputSize.h. For out-type interfaces, if the output shape remains unchanged, this field can be omitted. The configuration methods mainly include the following:

      ```yaml
        # Aten IR definition:
        - func: func_name(ArgType arg0, ArgType arg1, ...) -> Return
        # Method 1: same as the input parameter
          size: arg0

        # Method 2: enumerate the value of each dimension
          size: '{4, arg0.size(0), arg0.size(1), arg1.size(0)}'

        # Method 3: conditional expression
          size: 'arg1 == 1? arg0.sizes(): at::ArrayRef<int64_t>()'

        # Method 4: customize the infershape function in KernelNpuOutputSize.h, for example broadcast_ops_npu_output_size
          size: broadcast_ops_npu_output_size(arg0, arg1)
      ```

- `dtype`: Configures the dtype of the output tensor. If the dtype is the same as a parameter in the schema, it can be configured as the input parameter name. It can also be configured as a custom inferdtype function, which must be implemented in `KernelNpuOutputDtype.h`. For out-type interfaces, if the output dtype does not need to be checked, this field can be omitted. The configuration methods mainly include the following:

    ```yaml
        # Aten IR definition:
        - func: func_name(ArgType arg0, ArgType arg1, ...) -> Return
        # Method 1: Same as the input parameter
          dtype: arg0

        # Method 2: Configure as a known dtype type
          dtype: at::kFloat

        # Method 3: Conditional expression
          dtype: 'isIntegralType(arg0.scalar_type(), true) ? at::kFloat : arg0.scalar_type()'

        # Method 4: Customize the inferdtype function in KernelNpuOutputDtype.h.
          dtype: inferdtype(arg0, arg1)
    ```

  - name: Configure this field when the output result involves named tensor logic. Currently, only configurations where the name is the same as the input parameter are supported. If not involved, it can be ignored.
  - exec: Configure the parameters corresponding to EXEC\_NPU\_CMD. If, except for aclnnname, the order of other parameters is the same as that of Aten IR, you can configure only aclnnname, such as _aclnnAbs_. Taking abs as an example, the exec field can be configured in the following two ways.

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

    structured\_inherit: If the field configuration of the original function or in-place class interface is the same as that of the out class interface, you can inherit the corresponding out class interface through this field.

    Taking abs as an example, the out attribute and exec of the original function and the out class function are the same, so they can be inherited through the structured\_inherit field.

    ```yaml
      - func: abs(Tensor self) -> Tensor
        op_api: [v1.11, newest]
        gen_opapi:
          structured_inherit: abs.out
    ```

### aclop Operator Adaptation

The aclop operator is an earlier operator implementation method and is not recommended. The adaptation file path is: `op\_plugin/ops/aclops/AbsKernelNpu.cpp`. The file naming convention is operator name + `KernelNpu`, with the first letter of the operator name capitalized. The specific steps are as follows:

1. Create the operator adaptation file and include the required header files, such as the aclop external interface, operator declarations, and base functions.
2. Implement the operator interface adaptation. aclop operators must be defined in the `acl_op` namespace, and the function parameters and return values must be consistent with the torch API.

    ```cpp
    // Operator adaptation implementation file path: op_plugin/ops/aclops/AbsKernelNpu.cpp
    // 1. Include the required header files
    // External interface header file, containing the function prototypes of all aclop operators in op_plugin
    #include "op_plugin/AclOpsInterface.h" 
    // Header file for the base functions that torch depends on when calling ACLOP operators
    #include "op_plugin/utils/OpAdapter.h" 
    
    // 2. Operator interface adaptation implementation
    // CANN operators are defined in the acl_op namespace
    namespace acl_op { 
    using npu_preparation = at_npu::native::OpPreparation; 
    using npu_utils = at_npu::native::NpuUtils; 
    // Interfaces that are not publicly exposed are all defined in an anonymous namespace. Common examples include xx_nocheck, which directly call ACLOP operators without performing memory or shape checks
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
    
    // abs API implementation function, with a unique name and parameters consistent with the torch API
    at::Tensor abs(const at::Tensor& self)
    { 
        // Construct the output tensor and call the aclop operator
        auto output_size = op_infer::infershape_for_elewise(self); 
        at::Tensor result = npu_preparation::apply_tensor(self, output_size); 
        abs_out_nocheck(result, self); 
        return result; 
    }
    //abs variant code insertion location
    } // namespace acl_op
    ```

3. If the interface contains multiple variants, such as variants with an out parameter or in-place operation variants, you need to add the corresponding adaptation code. Refer to the following for adaptation:

    - Variant with out parameter:

      ```cpp
      // abs_out api implementation function, with a unique name and parameters consistent with the torch api
      at::Tensor& abs_out(const at::Tensor& self, at::Tensor& result)
      { 
          // CheckOut purpose: validates whether the size, dtype, etc. of result meet expectations. If dtype does not meet expectations, an error is thrown. If size does not meet expectations, a resize operation is performed
          npu_preparation::CheckOut({self}, result, self); 
          // check_match purpose: validates whether result is contiguous. Since ACLOP operators do not support non-contiguous output, a separate handling is required when result is non-contiguous.
          if (!npu_utils::check_match(&result)) { 
              // If result is non-contiguous, create a contiguous tensor (contig_tensor) to receive the output of the ACLOP operator (abs). Then copy contig_tensor to the original output result.
              at::Tensor contiguous_result = npu_utils::format_contiguous(result); 
              abs_out_nocheck(contigTensor, self); 
              npu_utils::format_fresh_view(result, contiguous_result); 
          } else { 
              // If result is contiguous, call the ACLOP operator directly.
              abs_out_nocheck(result, self); 
          } 
          return result; 
      }
      ```

    - In-place operation variant:

      ```cpp
      // abs_ API implementation function, with a unique name and parameters consistent with the torch API. This interface is an in-place operation, meaning the output result is stored in the input tensor.
      at::Tensor& abs_(at::Tensor& self)
      {
          // Call the out interface to avoid errors when directly calling the aclop operator in non-contiguous scenarios where self is used as the output.
          return acl_op::abs_out(self, self);
      }
      ```

4. If there are differences in adaptation code between different versions, all code is placed in the same file and distinguished using compilation macros.

    ```cpp
    #include "op_plugin/AclOpsInterface.h"
    #include "op_plugin/utils/custom_functions/aclops/inner_compute.h"
    namespace acl_op {
    // The function parameters for 1.11 differ from those for 2.0 and later versions, so macros are used to control this
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

    // The code for 2.0 and later versions is consistent
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

## Operator Auxiliary Adaptation

> [!NOTE]  
> The following auxiliary adaptation for npu_transpose is only an example. Users need to modify it according to actual scenarios.

### API Documentation Adaptation

Add the documentation for the new interface in the `codegen/templates/_op_plugin_docs.py` file. Generally, only the forward operator is documented. The specific example is as follows:

  ```python
  _add_torch_npu_docstr(
      "npu_transpose",
      """
  torch_npu.npu_transpose(self, perm, require_contiguous=True) -> Tensor

  Function description
  Returns a view of the original tensor with its dimensions permuted, and the result is contiguous. Supports FakeTensor mode.

  Parameter description
  self (Tensor): Input tensor.
  perm (ListInt): The corresponding dimension permutation.
  require_contiguous (Bool): Whether the user needs to make the input tensor contiguous. When set to False, it means the input tensor will not be made contiguous. It can only be set to True when the user explicitly knows that the input tensor is contiguous or a transposed tensor. The default value is True.

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

### Public API Adaptation

To add a public API, you need to add the interface configuration in the following files:

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
    > Entries starting with "op-api:" represent Python interfaces, and entries starting with "func:" represent C++ interfaces.

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

When using features such as `fx` and `compile`, the meta implementation of the operator interface must be registered so that it can execute normally when using faketensor. Currently, the meta implementations of operators are uniformly registered in the file `op_plugin/python/meta/_meta_registrations.py`.

  ```python
  @impl(m, "npu_transpose")
  def npu_transpose_meta(self, perm, require_contiguous=True):
      output = self.permute(perm)
      return torch.empty_like(output, dtype=self.dtype)
  ```
