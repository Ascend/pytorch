# Standard Access: EXEC_NPU_CMD_EXT (only adapted for aclnn)

## Usage Guide

`EXEC_NPU_CMD_EXT` is a macro provided by torch_npu. The macro automatically handles TaskQueue adaptation details such as stream acquisition, lambda capture, and parameter conversion. Developers only need to pass the operator name and parameters with the correct signature.

The workflow of `EXEC_NPU_CMD_EXT` is divided into four steps:

1. **Obtain the stream**: The macro internally calls `c10_npu::getCurrentNPUStream().stream(false)` to obtain `aclrtStream` without triggering TaskQueue flushing. This process is managed by the macro, so developers do not need to call it manually.
2. **Encapsulate the lambda**: The macro wraps the operator call and parameters into a `PROC_FUNC` (`std::function<int()>`) and captures the required variables by value.
3. **Enqueue and execute**: Submit to the TaskQueue through `OpCommand::RunOpApiV2`, and the operator dispatch thread calls the lambda to complete the Host API call.
4. **Reclamation management**: The macro is responsible for the lifecycle management of captured variables, so developers do not need to worry about it.

> [!NOTE]
>
> - Do not manually manage `aclrtStream`. The macro has already managed the acquisition, passing, and synchronization of the stream; additionally calling `getCurrentNPUStream().stream(false)` or managing `aclrtStream` yourself may conflict with the internal logic of the macro.
> - If you only need the `NPUStream` object for stream inspection (such as comparing stream ids or querying stream attributes), you can directly call `c10_npu::getCurrentNPUStream()`. This call does not trigger queue flushing.
> - The input `at::Tensor` is captured by value internally by the macro, so developers do not need to manually extend its lifecycle in the operator function.

## Usage Examples

```cpp
#include <ATen/OpCommand.h>

// A simple addition operator that ultimately calls aclnnAdd
at::Tensor custom_add(const at::Tensor &x, const at::Tensor &y)
{
    // 1. Create the output tensor (main thread)
    at::Tensor z = at::empty_like(x);

    // 2. Use EXEC_NPU_CMD_EXT; the macro internally handles stream, lambda, and enqueue automatically
    EXEC_NPU_CMD_EXT(aclnnAdd, x, y, z);
    return z;
}
```

**Example Description**:

- The first parameter `aclnnAdd` is the aclnn operator to be called (you can also use namespace-prefixed operators such as `at::native::CustomAclnnOp`).
- The subsequent parameters `x`, `y`, and `z` are the inputs and outputs of the aclnn operator, which the macro captures by value into the lambda.
- Based on the aclnn operator parameter list, the macro automatically converts `at::Tensor` to `aclTensor*`; developers do not need to perform type conversion manually.

For custom operator adaptation development, see: [Adaptation Development and Invocation (Basic Example)](https://gitcode.com/Ascend/op-plugin/tree/26.2.0/examples/cpp_extension_base/README.md).
