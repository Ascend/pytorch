# 自定义内存分配器

## 简介

支持从so文件中加载自定义NPU内存分配器。

## 使用场景

-   具有特殊内存需求的算法模型，可以自定义内存分配。
-   想提高内存的合理利用率，提高训练的性能，可以考虑该特性。

## 使用指导

接口原型：

```
torch_npu.npu.NPUPluggableAllocator(path_to_so_file, alloc_fn_name, free_fn_name)
```

参数说明：

-   path\_to\_so\_file：\(str\) so文件路径。
-   alloc\_fn\_name：\(str\)内存申请函数名（与C/C++文件中函数名一致）。
-   free\_fn\_name：\(str\)内存释放函数名（与C/C++文件中函数名一致）。

此接口详情请参考《API参考》中的“[torch.npu.npu.NPUPluggableAllocator](https://gitcode.com/Ascend/op-plugin/blob/7.3.0/docs/context/torch-npu-npu-NPUPluggableAllocator.md)”章节。

## 使用样例

```Python
import torch 
import torch_npu 
# Load the allocator 
new_alloc = torch_npu.npu.NPUPluggableAllocator('pluggable_allocator_extensions.so', 'my_malloc', 'my_free') 
# Swap the current allocator 
torch_npu.npu.memory.change_current_allocator(new_alloc) 
# This will allocate memory in the device using the new allocator 
npu_tensor = torch.zeros(10, device='npu')
```

## 约束说明

alloc\_fn\_name内存申请函数名必须与C/C++文件中函数名一致。

free\_fn\_name内存释放函数名必须与C/C++文件中函数名一致。

