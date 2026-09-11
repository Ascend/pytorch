# 标准接入：EXEC_NPU_CMD_EXT（仅适配aclnn）

## 使用指导

`EXEC_NPU_CMD_EXT` 是torch_npu提供的宏，宏内部已自动处理stream获取、lambda捕获、参数转换等TaskQueue适配细节，开发者仅需按正确签名传入算子名称与参数即可。

`EXEC_NPU_CMD_EXT` 的工作流程分为四步：

1. **获取stream**：宏内部调用 `c10_npu::getCurrentNPUStream().stream(false)` 获取 `aclrtStream`，不触发TaskQueue清空。该过程由宏托管，开发者无需手动调用。
2. **封装lambda**：宏将算子调用与参数包成一个 `PROC_FUNC`（`std::function<int()>`），按值捕获所需变量。
3. **入队执行**：通过 `OpCommand::RunOpApiV2` 提交到TaskQueue队列，由算子下发线程调用lambda完成Host API调用。
4. **回收管理**：宏内部负责捕获变量的生命周期管理，开发者无需关注。

> [!NOTE]
>
> - 禁止手动管理 `aclrtStream`。宏内部已托管stream的获取、传递与同步；若额外调用 `getCurrentNPUStream().stream(false)` 或自行管理 `aclrtStream`，可能与宏内部逻辑冲突。
> - 若仅需 `NPUStream` 对象进行查流（如比较stream id、查询流属性等），可直接调用 `c10_npu::getCurrentNPUStream()`，该调用不会触发队列清空。
> - 入参的 `at::Tensor` 由宏内部按值捕获，开发者无需在算子函数中手动延长其生命周期。

## 使用样例

```cpp
#include <ATen/OpCommand.h>

// 一个简单的加法算子，最终调用aclnnAdd
at::Tensor custom_add(const at::Tensor &x, const at::Tensor &y)
{
    // 1. 创建输出tensor（主线程）
    at::Tensor z = at::empty_like(x);

    // 2. 使用EXEC_NPU_CMD_EXT，宏内部自动处理stream、lambda、入队
    EXEC_NPU_CMD_EXT(aclnnAdd, x, y, z);
    return z;
}
```

**示例说明**：

- 第一个参数 `aclnnAdd` 为待调用的aclnn算子（也可使用带命名空间前缀的 `at::native::CustomAclnnOp` 等）。
- 后续参数 `x`、`y`、`z` 为aclnn算子的输入与输出，宏内部将其按值捕获至lambda中。
- 宏依据aclnn算子参数列表自动将 `at::Tensor` 转换为 `aclTensor*`，开发者无需手动执行类型转换。

自定义算子适配开发请参见：[适配开发及调用（基础样例）](https://gitcode.com/Ascend/op-plugin/tree/master/examples/cpp_extension_base)。
