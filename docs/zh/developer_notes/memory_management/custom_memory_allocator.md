# 自定义内存分配器


## 简介

支持从so文件中加载自定义NPU内存分配器。

## 使用场景

- 具有特殊内存需求的算法模型，可以自定义内存分配。
- 想提高内存的合理利用率，提高训练的性能，可以考虑该特性。

## 使用指导

接口原型：

```python
torch_npu.npu.NPUPluggableAllocator(path_to_so_file, alloc_fn_name, free_fn_name)
```

参数说明：

- path\_to\_so\_file：（str） so文件路径。
- alloc\_fn\_name：（str）内存申请函数名（与C/C++文件中函数名一致）。
- free\_fn\_name：（str）内存释放函数名（与C/C++文件中函数名一致）。

此接口详情请参考《自定义API》中的“[torch_npu.npu.NPUPluggableAllocator](https://gitcode.com/Ascend/op-plugin/blob/master/docs/zh/custom_APIs/torch_npu-npu/torch-npu-npu-NPUPluggableAllocator.md)”章节。

## 使用样例

以下示例使用 CANN 运行时申请和释放设备内存。加载的 C/C++ 共享库须导出两个符合签名的符号。C++ 实现使用 `extern "C"` 保留符号名。

```cpp
#include <sys/types.h>
#include <acl/acl_rt.h>

extern "C" void* my_malloc(ssize_t size, int, aclrtStream) {
    void* ptr = nullptr;
    if (aclrtMalloc(&ptr, size, ACL_MEM_MALLOC_NORMAL_ONLY) != ACL_SUCCESS) {
        return nullptr;
    }
    return ptr;
}

extern "C" void my_free(void* ptr, ssize_t, int, aclrtStream) {
    aclrtFree(ptr);
}
```

保存为 `pluggable_allocator_extensions.cpp`。初始化 CANN 环境后，在源文件所在目录执行以下命令构建共享库。

```bash
c++ -shared -fPIC -I"$ASCEND_HOME_PATH/include" \
    pluggable_allocator_extensions.cpp -L"$ASCEND_HOME_PATH/lib64" \
    -lascendcl -o pluggable_allocator_extensions.so
```

在一个新的 Python 进程中运行以下示例。

```python
import torch
import torch_npu
# Load the allocator
new_alloc = torch_npu.npu.NPUPluggableAllocator('./pluggable_allocator_extensions.so', 'my_malloc', 'my_free')
# Swap the current allocator
torch_npu.npu.memory.change_current_allocator(new_alloc)
# This will allocate memory in the device using the new allocator
npu_tensor = torch.zeros(10, device='npu')
```

## 约束说明

- 在当前进程首次分配 NPU 内存前调用 `change_current_allocator`。默认分配器初始化后，无法通过该接口替换。

- alloc\_fn\_name内存申请函数名必须与C/C++文件中函数名一致。
- free\_fn\_name内存释放函数名必须与C/C++文件中函数名一致。
<!-- npu="950" id1 -->
- <term>Ascend 950DT系列产品</term>暂不支持自定义内存分配器特性。
<!-- end id1 -->
