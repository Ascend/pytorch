# Custom Memory Allocator

## Introduction

Supports loading a custom NPU memory allocator from an so file.

## Use Scenario

- Algorithm models with special memory requirements can customize memory allocation.
- Consider this feature if you want to make memory utilization more reasonable and enhance training performance.

## Usage Guide

Prototype:

```python
torch_npu.npu.NPUPluggableAllocator(path_to_so_file, alloc_fn_name, free_fn_name)
```

Parameters:

- `path_to_so_file`: (str) Path to the so file.
- `alloc_fn_name`: (str) Memory allocation function name (must match the function name in the C/C++ file).
- `free_fn_name`: (str) Name of the memory deallocation function (must match the function name in the C/C++ file).

For details about this interface, refer to the "[torch_npu.npu.NPUPluggableAllocator](https://gitcode.com/Ascend/op-plugin/blob/26.1.0/docs/en/custom_APIs/torch_npu-npu/torch-npu-npu-NPUPluggableAllocator.md)" section in *Custom API*.

## Usage Example

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

## Constraints

The `alloc_fn_name` memory allocation function name must match the function name in the C/C++ file.

The `free_fn_name` memory deallocation function name must match the function name in the C/C++ file.

Ascend 950DT does not support the custom memory allocator feature.
