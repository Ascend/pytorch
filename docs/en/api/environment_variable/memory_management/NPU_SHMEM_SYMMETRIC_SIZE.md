# NPU\_SHMEM\_SYMMETRIC\_SIZE

## Function Description

This environment variable configures the heap size of NPU symmetric memory. Symmetric memory is used for direct memory access between NPU devices and is the foundation of the NPUSHMEM feature.

- Default value: 1 GiB, allocating a 1 GiB symmetric memory heap size.
- Configure as a pure number (in bytes): for example, `1073741824`, allocating a 1073741824-byte symmetric memory heap size.
- Configure as a number with a suffix: `k/K` (KB), `m/M` (MB), `g/G` (GB), `t/T` (TB). For example, `2G` allocates a 2 GiB symmetric memory heap size, and `512M` allocates a 512 MB symmetric memory heap size.

A larger heap can hold more symmetric memory data for direct access between devices. An overly small configuration may cause symmetric memory allocation to fail.

> [!NOTE]
>
> - This environment variable is read and cached when `OptionsManager::GetShmemSymmetricSize` is first called. Modifications made during runtime do not take effect.
> - This value is passed to `aclshmemx_init_attr` or `shmem_set_attr` to initialize the symmetric memory region.

This environment variable is provided by TorchNPU. PyTorch does not have a directly corresponding environment variable.

## Configuration Example

```bash
# Configure it to 2 GiB
export NPU_SHMEM_SYMMETRIC_SIZE=2G

# Configure it to 512 MB
export NPU_SHMEM_SYMMETRIC_SIZE=512M
```

## Usage Constraints

The configuration value must be a positive number. An invalid format raises an `NPU_SHMEM_SYMMETRIC_SIZE is invalid` error.

## Supported Products

<!-- npu="910" id1 -->
- <term>Atlas training products</term>
<!-- end id1 -->
<!-- npu="910b" id2 -->
- <term>Atlas A2 training products</term>
<!-- end id2 -->
<!-- npu="A3" id3 -->
- <term>Atlas A3 training products</term>
<!-- end id3 -->
<!-- npu="950" id4 -->
- <term>Ascend 950DT products</term>
<!-- end id4 -->
