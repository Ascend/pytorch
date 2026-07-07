# Common Reference

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-06-15T07:52:07.805Z pushedAt=2026-06-15T12:00:44.105Z -->

## PyTorch Schema Rules<a id="section001"></a>

For the official schema (operator description specification) guide, see the [README](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/README.md).

Since PyTorch 2.1 uses the official torchgen for code generation, certain official generation specifications must be followed. Operators that do not meet the schema specification will cause compilation errors. The currently involved cases are:

- Functions whose names start with "new_", end with "_like", or have `tensor_options` in their parameters but no tensor input parameter must have a `CompositeExplicitAutograd` dispatch.

    ```yaml
    - func: empty_with_format(int[] size, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None, int acl_format=2) -> Tensor
      dispatch:
        CompositeExplicitAutograd: empty_with_format
    ```

- Functions with "rand" or "dropout" in their names, or those with a generator parameter, must have the nondeterministic_seeded tag.

    ```yaml
    - func: dropout_with_byte_mask(Tensor self, float p, bool train) -> Tensor
      tags: nondeterministic_seeded
    ```

## Common API Interfaces for Operator Adaptation

For the basic interfaces of torch_npu operator operations, see the [OpPreparation](https://gitcode.com/Ascend/pytorch/blob/v2.7.1-26.0.0/torch_npu/csrc/framework/utils/OpPreparation.h). Common interfaces are as follows:

- **`at_npu::native::OpPreparation::apply_tensor`**  
  - Function: Creates an output tensor with the same attributes (device, data type, format) as the input tensor, suitable for output initialization of most operators.  
  - Example:  

     ```cpp
     at::Tensor result = at_npu::native::OpPreparation::apply_tensor(rois, output_size);
     ```

- **`at_npu::native::OpPreparation::apply_tensor_without_format`**  
  - Function: Creates an output tensor that matches the size and data type of the input tensor, but does not specify a format (such as channel order). Suitable for scenarios where no format constraint is required.  
  - Example:  

     ```cpp
     at::Tensor result = at_npu::native::OpPreparation::apply_tensor_without_format(output_size, input.options());
     ```

- **`at_npu::native::OpPreparation::check_tensor`**  
  - Function: Checks the attribute consistency (size, data type, etc.) between the output tensor and the input tensor, and adjusts them if they do not match, ensuring operator robustness.  
  - Example:  

     ```cpp
     at_npu::native::OpPreparation::check_tensor({self}, out, out, output_size);
     ```

- **`copy_scalar_to_device`**  
  - Function: Copies a scalar value from the host (CPU) to the device (NPU), ensuring that the scalar data can be correctly used in NPU computations and resolving cross-device data access issues.  
  - Example (reference logic):  

     ```cpp
     at::Scalar scalar = 2.0f;
     at::Tensor device_scalar = copy_scalar_to_device(scalar, input.device());
     ```

- **`binary_op_check`**  
  - Function: Checks whether the two input tensors of a binary operator meet the operation conditions (such as device consistency, data type compatibility, etc.), identifies invalid inputs in advance, and throws an exception.  
  - Example (reference logic):  

     ```cpp
     binary_op_check(input1, input2, "add");
     // Ensure that input1 and input2 meet the requirements of the add operator in terms of device, data type, etc.
     ```

## Common Macro Definitions for Operator Adaptation

For operator adaptation macro definition APIs, refer to the [operator APIs](https://gitcode.com/Ascend/op-plugin/blob/26.0.0/op_plugin/utils/op_api_common.h). Common macro definitions are as follows:

- **`DO_COMPATIBILITY`**  
  - Function: Used for operator compatibility handling. When the native NPU operator is unavailable, it automatically degrades to an alternative implementation (such as the CPU version), ensuring functional compatibility across different environments.  
  - Example:  

     ```cpp
     DO_COMPATIBILITY(aclnnForeachTan, at::native::foreach_tensor_tan_slow_(self));
     ```

- **`EXEC_NPU_CMD`**  
  - Function: Encapsulates the underlying NPU operator invocation logic, automatically handles input and output tensor passing, simplifies interaction with the NPU hardware interface, and supports batch splitting execution.  
  - Example:  

     ```cpp
     EXEC_NPU_CMD(aclnnForeachAddScalarV2, temp_tensors1, scalar_, temp_result);
     ```

- **`OPS_ERROR`**  
  - Function: Generates exception information with error codes, used for parameter validation or runtime error prompts, improving error location accuracy.  
  - Example:  

     ```cpp
     TORCH_CHECK(src.is_sparse(), "add(sparse, dense) is not supported. Use add(dense, sparse) instead.", OPS_ERROR(ErrCode::VALUE));
     ```

- **`VERSION_BETWEEN`**  
  - Function: Performs conditional compilation based on the NPU version range, taking effect only within the specified version interval, adapting to feature differences across hardware versions (combined with CMake version judgment logic).  
  - Example (logic reference):  

     ```cpp
     #if VERSION_BETWEEN(V2R1, VERSION_NEWEST)
     // Code that takes effect for a specific version
     #endif
     ```

- **`FLOP_COUNT`**  
  - Function: Counts the floating-point operations (FLOPs) of an operator, used for performance analysis and optimization.  
  - Example (logical reference):  

     ```cpp
     FLOP_COUNT(FlopCounter::mm_flop, input, weight_t);
     ```
