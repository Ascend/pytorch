# 各类数据捕获方式示例

本附录为 [Lambda函数捕获](taskqueue_op_developer.md#lambda函数捕获) 中“捕获原则速查表”小节的详细代码示例，逐一展示四个数据类型的具体捕获方法。建议先阅读速查表建立整体认知，再根据所需查阅本附录中的代码片段。

## 基础数据类型

int、bool、float等基础类型直接值捕获即可。这些类型体积小、拷贝成本低，值捕获后在lambda中拥有一份独立的副本。

```cpp
// 基础类型blockDim和totalLength，lambda值传递捕获副本，跟随lambda的生命周期uint32_t blockDim = 8;
uint32_t totalLength = 1;
for (uint32_t size : x.sizes()) {
    totalLength *= size;
}
auto acl_call = [=]() -> int {
    // blockDim和totalLength都是值捕获的副本，可以安全使用add_custom<<<blockDim, nullptr, acl_stream>>>(xGm, yGm, zGm, totalLength);
    // ...
};
```

## 张量数据类型

通常不建议直接值捕获 `at::Tensor` 对象。值捕获会递增Tensor引用计数，导致其生命周期非预期延长，底层NPU内存无法及时释放回NPU内存池，可能造成**内存上涨**。

建议做法：捕获Tensor的 `data_ptr` 裸指针。NPU上的Tensor内存由torch_npu的内存池统一管理，即使Tensor在主线程析构，其底层内存也仅归还至内存池而不会被释放。同一stream上的后续算子可安全访问该内存（不同stream的内存不会复用，且stream内保序执行，不会发生数据冲突）。

```cpp
// 正确：传递at::Tensor的data_ptr裸指针auto xGm = (uint8_t *)(x.mutable_data_ptr());
auto yGm = (uint8_t *)(y.mutable_data_ptr());
auto zGm = (uint8_t *)(z.mutable_data_ptr());

auto acl_call = [=]() -> int {
    // 裸指针值捕获，xGm/yGm/zGm指向的NPU内存由内存池保证生命周期add_custom<<<blockDim, nullptr, acl_stream>>>(xGm, yGm, zGm, totalLength);
    // ...
};
```

## 自定义数据类型

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

    // 构造智能指针，将Tensor的内存地址和元数据封装到智能指针中auto x_ptr = std::make_shared<TensorStruct>(...);
    auto y_ptr = std::make_shared<TensorStruct>(...);
    auto z_ptr = std::make_shared<TensorStruct>(...);

    // 值捕获智能指针，引用计数递增，对象生命周期延长到lambda结束auto acl_call = [=]() -> int {
        auto x_data = x_ptr->data_ptr;  // 安全：智能指针生命周期被延长
        // Launch kernel ...
    };
    // lambda执行后，智能指针自动析构，TensorStruct被自动释放
    ```

> [!NOTE]
>
> 若智能指针中保存了指向NPU Tensor内存的裸指针（如 `data_ptr`），该内存不受智能指针管理，其生命周期由NPU内存池保障——Tensor在主线程析构后，内存池将其标记为可复用，但在同一stream上，lambda执行完成前不被其他算子复用。

## 显式管理数据类型

对于需显式创建和销毁的`aclTensor` 类对象，需在lambda内部手动管理生命周期，具体有两种方式：

- 在lambda中创建并手动释放。
- 在主线程创建后值捕获指针，在lambda中手动释放。

    ```cpp
    // 在主线程创建，lambda内手动释放aclTensor *acl_x = ConvertType(x);   // 创建aclTensor（堆上内存）
    aclTensor *acl_y = ConvertType(y);
    aclTensor *acl_z = ConvertType(z);

    auto acl_call = [=]() -> int {
        // 使用aclTensor...
        auto api_ret = aclnnAdd(acl_x, acl_y, acl_z, totalLength);
        // 必须在lambda内手动释放aclDestroyTensor(acl_x);
        aclDestroyTensor(acl_y);
        aclDestroyTensor(acl_z);
        // ...
    };
    ```

`aclTensor` 内部引用的NPU Tensor内存同样由NPU内存池管理，`aclDestroyTensor` 只销毁aclTensor的包装结构，不会释放NPU设备内存。
