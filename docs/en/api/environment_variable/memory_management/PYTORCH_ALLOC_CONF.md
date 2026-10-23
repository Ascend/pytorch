# PYTORCH_ALLOC_CONF

## Function Description

The PyTorch community officially released the `PYTORCH_ALLOC_CONF` environment variable starting from version 2.10.0. This variable is a unified configuration entry point for the caching allocator across all accelerator backends (such as CUDA, ROCm, and XPU) and is not bound to a specific device. Device-specific variable names such as `PYTORCH_CUDA_ALLOC_CONF` are retained only as backward-compatible aliases.

To keep the naming consistent with the PyTorch community, TorchNPU has introduced this same-named environment variable since version 26.2.0 to configure NPU caching allocator behavior. Its function is the same as `PYTORCH_NPU_ALLOC_CONF`, and it is recommended to use this variable first. It is not configured by default. For parameter descriptions, default values, and supported scope, see [PYTORCH_NPU_ALLOC_CONF](PYTORCH_NPU_ALLOC_CONF.md).

> [!NOTE]
>
> Compared with the same-named variable in native PyTorch, torch_npu supports NPU-specific parameters (such as `page_size`, `base_addr_aligned_kb`, `per_process_memory_fraction`, etc.) in addition to the natively supported parameters.

## Configuration Example

```bash
export PYTORCH_ALLOC_CONF=max_split_size_mb:512
```

## Usage Constraints

- This variable applies to TorchNPU 26.2.0 and PyTorch 2.10.0 and later versions. It is recommended to use this variable first.
- This variable is mutually exclusive with `PYTORCH_NPU_ALLOC_CONF`. Configuring both at the same time reports an error and exits the program.

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
<!-- npu="310p" id4 -->
- <term>Atlas inference products</term>
<!-- end id4 -->
<!-- npu="950" id5 -->
- <term>Ascend 950DT products</term>
<!-- end id5 -->