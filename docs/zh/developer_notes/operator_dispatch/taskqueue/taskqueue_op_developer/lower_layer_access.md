# 底层接入：RunOpApiV2

## 使用指导

该接入方式适用于AscendC kernel launch等非aclnn场景，需手动管理stream及lambda生命周期，以实现更精细的性能控制。完整用法包含三步：

1. 获取流与处理队列：先获取当前NPUStream，应显式调用`.stream(false)`转换为`aclrtStream`以获取CANN层句柄。

    ```cpp
    at::Tensor my_custom_op(const at::Tensor &x, const at::Tensor &y)
    {
        // 1. 获取aclrtStream（不清空队列）
        auto acl_stream = c10_npu::getCurrentNPUStream().stream(false);
        at::Tensor z = at::empty_like(x);
        //...
        //...
    }
    ```

    > [!NOTE]
    >
    > - 如果调用`.stream()`、`.stream(true)`会清空队列，这会破坏TaskQueue的异步下发流水线，降低性能。
    > - 直接将NPUStream传给需要aclrtStream的函数触发隐式类型转换，也会清空队列。
    >
    >   ```cpp
    >   auto npu_stream = c10_npu::getCurrentNPUStream();  // 获取NPUStream，此时不触发隐式类型转换
    >   // 其他逻辑...
    >   add_custom<<<blockDim, nullptr, npu_stream>>>; // npu_stream触发隐式类型转换，清空队列
    >   ```
    > 
    > - 若不使用TaskQueue而要直接在stream上执行算子，务必先清空TaskQueue。
    >
    >   ```cpp
    >   auto acl_stream = c10_npu::getCurrentNPUStream().stream(true);  // 或直接stream()
    >   // 注意：确保在清空队列之后再执行操作，且清空后不应再有入队操作
    >   ```

2. 定义lambda函数捕获，在lambda函数体内调用`my_kernel`并传入`acl_stream`作为执行流，再将定义好的函数提交至TaskQueue实现异步处理。

    ```cpp
    at::Tensor my_custom_op(const at::Tensor &x, const at::Tensor &y)
    {
        // 1. 获取aclrtStream（不清空队列）
        auto acl_stream = c10_npu::getCurrentNPUStream().stream(false);
        at::Tensor z = at::empty_like(x);
        // 2. 定义lambda：封装内核启动逻辑，值捕获所需变量
        auto acl_call = [=]() -> int {
            my_kernel<<<...>>>(..., acl_stream);
            return 0;
        };
        //...
    }
    ```

    **基本原则**：lambda函数进入TaskQueue队列后会在另一个线程执行，其生命周期晚于主线程，所有数据类型应采用值捕获（[=]），避免使用引用捕获（[&]），否则容易导致悬空引用。

    数据类型的具体捕获信息与示例如下：

    | 数据类型 | 推荐捕获方式 | 生命周期说明 |
    |---------|-------------|------------|
    | [基础类型](#样例1基础类型捕获) | 值捕获 `[=]` | 拷贝到lambda闭包，生命周期安全 |
    | [张量类型](#样例2张量类型捕获) | 传递裸指针（`data_ptr`） | 不延长tensor生命周期，内存由NPU内存池保障 |
    | [自定义数据](#样例3自定义类型捕获) | 智能指针（`std::shared_ptr`）值捕获 | 智能指针生命周期延长到lambda结束，自动析构 |
    | [显式管理类型](#样例4显式管理捕获) | 值捕获指针，手动释放| 在lambda内调用 `aclDestroyTensor` 等接口 |

3. 调用 `OpCommand::RunOpApiV2` 入队，将封装好的内核启动逻辑（lambda函数）通过`at_npu::native::OpCommand::RunOpApiV2`提交到NPU的TaskQueue中，实现异步执行。

    ```cpp
    at::Tensor my_custom_op(const at::Tensor &x, const at::Tensor &y)
    {
        // 1. 获取aclrtStream（不清空队列）
        auto acl_stream = c10_npu::getCurrentNPUStream().stream(false);
        at::Tensor z = at::empty_like(x);

        // 2. 定义lambda：封装内核启动逻辑，值捕获所需变量
        auto acl_call = [=]() -> int {
            my_kernel<<<...>>>(..., acl_stream);
            return 0;
        };

        // 3. 调用RunOpApiV2入队（op_name用于trace/profiling）
        at_npu::native::OpCommand::RunOpApiV2("my_custom_op", acl_call);
        return z;
    }
    ```

    **示例说明**

    接口`RunOpApiV2`的函数签名是`RunOpApiV2(const string &op_name, const PROC_FUNC &func, bool sync = false)`，参数信息如下：
    
    - `PROC_FUNC`：类型别名为 `std::function<int()>`，即用户传入的算子执行回调函数，返回 `int` 类型的错误码（0表示成功）。
    - `op_name`：算子名称（如 `Add`），用于`RECORD_FUNCTION`、`profiling`及t`race`标记。若长度超过99字节，将会被截断。
    - `func`：算子执行回调。`RunOpApiV2`以常引用传入（避免拷贝）。
    - `sync`：是否在算子下发后阻塞同步当前stream，默认 `false`。

## 使用样例

### 完整调用示例

使用 `stream(false)` 获取 `aclrtStream` 但不清空TaskQueue，配合 `OpCommand::RunOpApiV2` 将lambda正确入队。内核启动被封装在lambda中，通过正确的入队与出队确保执行顺序，避免与之前的任务乱序执行。

```cpp
at::Tensor ascendc_add_good(const at::Tensor &x, const at::Tensor &y)
{
    //1. 获取aclrtStream（不清空队列）
    auto acl_stream = c10_npu::getCurrentNPUStream().stream(false);
    at::Tensor z = at::empty_like(x);
    uint32_t blockDim = 8;
    uint32_t totalLength = 1;
    for (uint32_t size : x.sizes()) {
        totalLength *= size;
    }
    auto xGm = (uint8_t *)(x.mutable_data_ptr());
    auto yGm = (uint8_t *)(y.mutable_data_ptr());
    auto zGm = (uint8_t *)(z.mutable_data_ptr());

    // 2. 定义lambda：封装内核启动逻辑，值捕获所需变量
    auto acl_call = [=]() -> int {
        add_custom<<<blockDim, nullptr, acl_stream>>>(xGm, yGm, zGm, totalLength);
        return 0;
    };
    // 3. 调用RunOpApiV2入队
    at_npu::native::OpCommand::RunOpApiV2("ascendc_add", acl_call);
    return z;
}
```

> [!NOTE]
>
> 使用 `stream(false)` 但不使用 `OpCommand` 入队列，直接启动内核绕过了队列的串行化，可能导致内核在设备上的执行顺序与Host不一致，造成数据错误等问题。
>
> ```cpp
>
> at::Tensor ascendc_add_good(const at::Tensor &x, const at::Tensor &y)
> {
>     //...
>     //4.获取设备侧内存指针
>     auto xGm = (uint8_t *)(x.mutable_data_ptr());
>     auto yGm = (uint8_t *)(y.mutable_data_ptr());
>     auto zGm = (uint8_t *)(z.mutable_data_ptr());
>
>     //5.错误做法：直接启动内核绕过了队列的串行化，TaskQueue中可能还有未下发的任务，它们依赖的stream状态可能不被新内核感知
>     add_custom<<<blockDim, nullptr, acl_stream>>>(xGm, yGm, zGm, totalLength);
>     
>     return z;
> }
> ```

自定义算子扩展开发方法请参见：[自定义算子C++扩展开发示例](https://gitcode.com/Ascend/op-plugin/blob/master/examples/cpp_extension)。

### 数据类型捕获示例

#### 样例1：基础类型捕获

int、bool、float等基础类型直接值捕获即可。该类型体积小、拷贝成本低，值捕获后在lambda中拥有一份独立的副本。

```cpp
// ...
// 平凡类型blockDim和totalLength，lambda值传递捕获副本，跟随lambda的生命周期
uint32_t blockDim = 8;
uint32_t totalLength = 1;
for (uint32_t size : x.sizes()) {
    totalLength *= size;
}
auto acl_call = [=]() -> int {
    // blockDim 和 totalLength 都是值捕获的副本，可以安全使用
    add_custom<<<blockDim, nullptr, acl_stream>>>(xGm, yGm, zGm, totalLength);
    // ...
};
// ...
```

#### 样例2：张量类型捕获

通常不建议直接值捕获 `at::Tensor` 对象。值捕获会递增Tensor引用计数，导致其生命周期非预期延长，底层NPU内存无法及时释放回NPU内存池，可能造成**内存上涨**。

建议做法：捕获Tensor的 `data_ptr` 裸指针。NPU上的Tensor内存由torch_npu的内存池统一管理，即使Tensor在主线程析构，其底层内存也仅归还至内存池而不会被释放。同一stream上的后续算子可安全访问该内存（不同stream的内存不会复用，且stream内保序执行，不会发生数据冲突）。

```cpp
// 正确：传递 at::Tensor 的 data_ptr 裸指针
auto xGm = (uint8_t *)(x.mutable_data_ptr());
auto yGm = (uint8_t *)(y.mutable_data_ptr());
auto zGm = (uint8_t *)(z.mutable_data_ptr());

auto acl_call = [=]() -> int {
    // 裸指针值捕获，xGm/yGm/zGm指向的NPU内存由内存池保证生命周期
    add_custom<<<blockDim, nullptr, acl_stream>>>(xGm, yGm, zGm, totalLength);
    // ...
};
```

#### 样例3：自定义类型捕获

若入参较多，建议自定义数据结构进行封装，并使用**智能指针**（`std::shared_ptr`）进行值捕获以管理生命周期：

- 智能指针本身被值捕获，其引用计数递增，确保堆上对象的生命周期延续至lambda执行完毕。
- lambda执行结束后，智能指针自动析构并释放堆上对象。

    ```cpp
    struct TensorStruct {
        void *data_ptr = nullptr;       // at_tensor.storage().data()
        aclDataType acl_type;           // aclDataType of at_tensor
        aclFormat acl_format;
        size_t nbytes;                  // at_tensor.storage().nbytes()
        size_t itemsize;                // at_tensor.itemsize()
        int64_t storage_offset;         // at_tensor.storage_offset()
        std::vector<int64_t> sizes;     // at_tensor.sizes()
        std::vector<int64_t> strides;   // at_tensor.strides()
        std::vector<int64_t> storage_sizes;

        TensorStruct(
            void *data_ptr_, aclDataType acl_type_, aclFormat acl_format_,
            size_t nbytes_, size_t itemsize_, int64_t storage_offset_,
            at::IntArrayRef sizes_, at::IntArrayRef strides_, at::IntArrayRef storage_sizes_
        ) : data_ptr(data_ptr_), acl_type(acl_type_), acl_format(acl_format_),
            nbytes(nbytes_), itemsize(itemsize_), storage_offset(storage_offset_),
            sizes(sizes_.vec()), strides(strides_.vec()), storage_sizes(storage_sizes_.vec())
        {
        }
    };

    using TensorStructPtr = std::shared_ptr<TensorStruct>;

    // 构造智能指针，将Tensor的内存地址和元数据封装到智能指针中
    auto x_ptr = std::make_shared<TensorStruct>(...);
    auto y_ptr = std::make_shared<TensorStruct>(...);
    auto z_ptr = std::make_shared<TensorStruct>(...);

    // 值捕获智能指针，引用计数递增，对象生命周期延长到lambda结束
    auto acl_call = [=]() -> int {
        auto x_data = x_ptr->data_ptr;  // 安全：智能指针生命周期被延长
        // Launch kernel ...
    };
    // lambda执行后，智能指针自动析构，TensorStruct被自动释放

    ```

> [!NOTE]
>
> 若智能指针中保存了指向NPU Tensor内存的裸指针（如 `data_ptr`），该内存不受智能指针管理，其生命周期由NPU内存池保障。Tensor在主线程析构后，内存池将其标记为可复用，但在同一stream上，lambda执行完成前不被其他算子复用。

#### 样例4：显式管理捕获

对于需显式创建和销毁的`aclTensor` 类对象，需在lambda内部手动管理生命周期，具体有两种方式：

- 在lambda中创建并手动释放。
- 在主线程创建后值捕获指针，在lambda中手动释放。

    ```cpp
    // 在主线程创建，lambda 内手动释放
    aclTensor *acl_x = ConvertType(x);   // 创建aclTensor（堆上内存）
    aclTensor *acl_y = ConvertType(y);
    aclTensor *acl_z = ConvertType(z);

    auto acl_call = [=]() -> int {
        // 使用aclTensor...
        auto api_ret = aclnnAdd(acl_x, acl_y, acl_z, totalLength);
        // 必须在lambda内手动释放
        aclDestroyTensor(acl_x);
        aclDestroyTensor(acl_y);
        aclDestroyTensor(acl_z);
        //...
    };
    ```

`aclTensor` 内部引用的NPU Tensor内存同样由NPU内存池管理，`aclDestroyTensor` 只销毁aclTensor的包装结构，不会释放NPU设备内存。
