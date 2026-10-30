# Lower-Layer Access: RunOpApiV2

## Usage Guide

This access method applies to non-aclnn scenarios such as Ascend C kernel launch. You need to manually manage the stream and lambda lifecycle to achieve finer performance control. The complete usage consists of three steps:

1. Obtain the stream and process queue: First obtain the current NPUStream, and explicitly call `.stream(false)` to convert it to `aclrtStream` to obtain the CANN-layer handle.

    ```cpp
    at::Tensor my_custom_op(const at::Tensor &x, const at::Tensor &y)
    {
        // 1. Obtain aclrtStream (without clearing the queue)
        auto acl_stream = c10_npu::getCurrentNPUStream().stream(false);
        at::Tensor z = at::empty_like(x);
        //...
        //...
    }
    ```

    > [!NOTE]
    >
    > - Calling `.stream()` or `.stream(true)` flushes the queue, which destroys the asynchronous dispatch pipeline of TaskQueue and degrades performance.
    > - Directly passing an NPUStream to a function that requires aclrtStream triggers implicit type conversion and also flushes the queue.
    >
    >   ```cpp
    >   auto npu_stream = c10_npu::getCurrentNPUStream();  // Obtain the NPUStream; no implicit type conversion is triggered at this point
    >   // Other logic...
    >   add_custom<<<blockDim, nullptr, npu_stream>>>; // npu_stream triggers implicit type conversion and flushes the queue
    >   ```
    > 
    > - If you do not use TaskQueue but want to execute operators directly on the stream, be sure to flush the TaskQueue first.
    >
    >   ```cpp
    >   auto acl_stream = c10_npu::getCurrentNPUStream().stream(true);  // Or simply stream()
    >   // Note: Ensure that operations are executed after the queue is flushed, and no enqueue operations should occur after flushing
    >   ```

2. Define a lambda function capture: call `my_kernel` in the lambda function body and pass `acl_stream` as the execution stream, then submit the defined function to TaskQueue for asynchronous processing.

    ```cpp
    at::Tensor my_custom_op(const at::Tensor &x, const at::Tensor &y)
    {
        // 1. Obtain aclrtStream (without clearing the queue)
        auto acl_stream = c10_npu::getCurrentNPUStream().stream(false);
        at::Tensor z = at::empty_like(x);
        // 2. Define a lambda: encapsulate the kernel launch logic and capture the required variables by value
        auto acl_call = [=]() -> int {
            my_kernel<<<...>>>(..., acl_stream);
            return 0;
        };
        //...
    }
    ```

    **Basic Principle**: After a lambda function enters the TaskQueue queue, it is executed on another thread whose lifecycle outlives the main thread. All data types should be captured by value ([=]); avoid reference capture ([&]); otherwise, dangling references may occur.

    The specific capture information and examples for each data type are as follows:

    | Data Type | Recommended Capture Method | Lifecycle Description |
    |---------|-------------|------------|
    | [Basic types](#example-1-basic-type-capture) | Value capture `[=]` | Copied to the lambda closure; lifecycle is safe |
    | [Tensor types](#example-2-tensor-type-capture) | Pass a raw pointer (`data_ptr`) | Does not extend the tensor lifecycle; memory is guaranteed by the NPU memory pool |
    | [Custom data](#example-3-custom-type-capture) | Value capture of a smart pointer (`std::shared_ptr`) | The smart pointer lifecycle extends to the end of the lambda and is automatically destroyed |
    | [Explicitly managed types](#example-4-explicit-management-capture) | Value capture of a pointer, manually released | Call interfaces such as `aclDestroyTensor` within the lambda |

3. Call `OpCommand::RunOpApiV2` to enqueue: submit the encapsulated kernel launch logic (lambda function) to the NPU TaskQueue through `at_npu::native::OpCommand::RunOpApiV2` for asynchronous execution.

    ```cpp
    at::Tensor my_custom_op(const at::Tensor &x, const at::Tensor &y)
    {
        // 1. Obtain aclrtStream (without clearing the queue)
        auto acl_stream = c10_npu::getCurrentNPUStream().stream(false);
        at::Tensor z = at::empty_like(x);

        // 2. Define a lambda: encapsulate the kernel launch logic and capture the required variables by value
        auto acl_call = [=]() -> int {
            my_kernel<<<...>>>(..., acl_stream);
            return 0;
        };

        // 3. Call RunOpApiV2 to enqueue (op_name is used for trace/profiling)
        at_npu::native::OpCommand::RunOpApiV2("my_custom_op", acl_call);
        return z;
    }
    ```

    **Example Description**

    The function signature of the `RunOpApiV2` interface is `RunOpApiV2(const string &op_name, const PROC_FUNC &func, bool sync = false)`. The parameter information is as follows:
    
    - `PROC_FUNC`: A type alias of `std::function<int()>`, i.e., the operator execution callback function passed by the user, returning an `int` error code (0 indicates success).
    - `op_name`: The operator name (such as `Add`), used for `RECORD_FUNCTION`, `profiling`, and t`race` marking. If the length exceeds 99 bytes, it will be truncated.
    - `func`: The operator execution callback. `RunOpApiV2` passes it by constant reference (to avoid copying).
    - `sync`: Whether to block and synchronize the current stream after operator dispatch. The default is `false`.

## Usage Examples

### Complete Call Example

Use `stream(false)` to obtain `aclrtStream` without flushing the TaskQueue, and enqueue the lambda correctly with `OpCommand::RunOpApiV2`. The kernel launch is encapsulated in the lambda, and correct enqueue and dequeue ensure the execution order, avoiding out-of-order execution with previous tasks.

```cpp
at::Tensor ascendc_add_good(const at::Tensor &x, const at::Tensor &y)
{
    //1. Obtain aclrtStream (without clearing the queue)
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

    // 2. Define a lambda: encapsulate the kernel launch logic and capture the required variables by value
    auto acl_call = [=]() -> int {
        add_custom<<<blockDim, nullptr, acl_stream>>>(xGm, yGm, zGm, totalLength);
        return 0;
    };
    // 3. Call RunOpApiV2 to enqueue
    at_npu::native::OpCommand::RunOpApiV2("ascendc_add", acl_call);
    return z;
}
```

> [!NOTE]
>
> Using `stream(false)` without enqueuing through `OpCommand`, and directly launching the kernel, bypasses the serialization of the queue, which may cause the kernel execution order on the device to be inconsistent with the Host, leading to data errors and other problems.
>
> ```cpp
>
> at::Tensor ascendc_add_good(const at::Tensor &x, const at::Tensor &y)
> {
>     //...
>     //4. Obtain the device-side memory pointer
>     auto xGm = (uint8_t *)(x.mutable_data_ptr());
>     auto yGm = (uint8_t *)(y.mutable_data_ptr());
>     auto zGm = (uint8_t *)(z.mutable_data_ptr());
>
>     //5. Incorrect practice: directly launching the kernel bypasses the serialization of the queue. There may still be undelivered tasks in the TaskQueue, and the stream state they depend on may not be perceived by the new kernel
>     add_custom<<<blockDim, nullptr, acl_stream>>>(xGm, yGm, zGm, totalLength);
>     
>     return z;
> }
> ```

For custom operator extension development methods, see: [Custom Operator C++ Extension Development Example](https://gitcode.com/Ascend/op-plugin/blob/26.2.0/examples/cpp_extension/README_en.md).

### Data Type Capture Examples

#### Example 1: Basic Type Capture <a id="example-1-basic-type-capture"></a>

Basic types such as int, bool, and float can be directly captured by value. These types are small and have low copy costs. After value capture, the lambda has an independent copy.

```cpp
// ...
// Trivial types blockDim and totalLength are captured by value as copies in the lambda, following the lambda's lifetime
uint32_t blockDim = 8;
uint32_t totalLength = 1;
for (uint32_t size : x.sizes()) {
    totalLength *= size;
}
auto acl_call = [=]() -> int {
    // Both blockDim and totalLength are copies captured by value and can be safely used
    add_custom<<<blockDim, nullptr, acl_stream>>>(xGm, yGm, zGm, totalLength);
    // ...
};
// ...
```

#### Example 2: Tensor Type Capture <a id="example-2-tensor-type-capture"></a>

It is generally not recommended to directly capture `at::Tensor` objects by value. Value capture increments the Tensor reference count, causing its lifecycle to be extended unexpectedly, and the underlying NPU memory cannot be released back to the NPU memory pool in time, which may cause **memory growth**.

Recommended practice: capture the raw `data_ptr` pointer of the Tensor. Tensor memory on the NPU is uniformly managed by the torch_npu memory pool. Even if the Tensor is destroyed on the main thread, its underlying memory is only returned to the memory pool and is not released. Subsequent operators on the same stream can safely access this memory (memory of different streams is not reused, and execution within a stream is ordered, so no data conflicts occur).

```cpp
// Correct: pass the raw data_ptr pointer of at::Tensor
auto xGm = (uint8_t *)(x.mutable_data_ptr());
auto yGm = (uint8_t *)(y.mutable_data_ptr());
auto zGm = (uint8_t *)(z.mutable_data_ptr());

auto acl_call = [=]() -> int {
    // Raw pointers are captured by value; the NPU memory pointed to by xGm/yGm/zGm has its lifetime guaranteed by the memory pool
    add_custom<<<blockDim, nullptr, acl_stream>>>(xGm, yGm, zGm, totalLength);
    // ...
};
```

#### Example 3: Custom Type Capture <a id="example-3-custom-type-capture"></a>

If there are many input parameters, it is recommended to define a custom data structure for encapsulation and use a **smart pointer** (`std::shared_ptr`) for value capture to manage the lifecycle:

- The smart pointer itself is captured by value, and its reference count is incremented, ensuring that the lifecycle of the heap object extends until the lambda finishes execution.
- After the lambda finishes execution, the smart pointer is automatically destroyed and releases the heap object.

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

    // Construct a smart pointer to encapsulate the memory address and metadata of the Tensor into the smart pointer
    auto x_ptr = std::make_shared<TensorStruct>(...);
    auto y_ptr = std::make_shared<TensorStruct>(...);
    auto z_ptr = std::make_shared<TensorStruct>(...);

    // Capture the smart pointer by value, incrementing the reference count and extending the object's lifetime until the lambda ends
    auto acl_call = [=]() -> int {
        auto x_data = x_ptr->data_ptr;  // Safe: the smart pointer's lifetime is extended
        // Launch kernel ...
    };
    // After the lambda executes, the smart pointer is automatically destructed and TensorStruct is automatically released

    ```

> [!NOTE]
>
> If the smart pointer stores a raw pointer to NPU Tensor memory (such as `data_ptr`), that memory is not managed by the smart pointer; its lifecycle is guaranteed by the NPU memory pool. After the Tensor is destroyed on the main thread, the memory pool marks it as reusable, but on the same stream, it is not reused by other operators until the lambda finishes execution.

#### Example 4: Explicit Management Capture <a id="example-4-explicit-management-capture"></a>

For `aclTensor` objects that need to be explicitly created and destroyed, the lifecycle must be manually managed inside the lambda. There are two specific ways:

- Create and manually release within the lambda.
- Create on the main thread, capture the pointer by value, and manually release within the lambda.

    ```cpp
    // Created in the main thread, manually released inside the lambda
    aclTensor *acl_x = ConvertType(x);   // Create an aclTensor (heap memory)
    aclTensor *acl_y = ConvertType(y);
    aclTensor *acl_z = ConvertType(z);

    auto acl_call = [=]() -> int {
        // Use aclTensor...
        auto api_ret = aclnnAdd(acl_x, acl_y, acl_z, totalLength);
        // Must be manually released inside the lambda
        aclDestroyTensor(acl_x);
        aclDestroyTensor(acl_y);
        aclDestroyTensor(acl_z);
        //...
    };
    ```

The NPU Tensor memory referenced internally by `aclTensor` is also managed by the NPU memory pool. `aclDestroyTensor` only destroys the wrapper structure of aclTensor and does not release NPU device memory.
