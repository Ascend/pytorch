# TORCH\_HCCL\_STATUS\_SAVE\_PATH

## Function Description

When HCCL is used as the communication backend, this environment variable configures the saving directory of HCCL process group status files. The status files are created in this directory using a fixed naming rule, and the file name format is `torch_hccl_status-<global_rank>_<master_addr>_<deviceId>_<numRanks>_<pid>_<timestamp>.log`.

- Default value: `/tmp`. The status files are saved in the `/tmp` directory.
- When set to a specified path: The status files are saved in the specified directory, and the file name format remains unchanged.

> [!NOTE]
>
> This environment variable is read during the global initialization phase of `ProcessGroupHCCL.cpp` and should be set before loading the `torch_npu` extension.

This environment variable is provided by TorchNPU. PyTorch has no directly corresponding environment variable.

## Configuration Example

```bash
export TORCH_HCCL_STATUS_SAVE_PATH=/data/hccl_status
```

## Usage Constraints

Only when `TORCH_HCCL_STATUS_SAVE_ENABLE=1` are the status files written to the directory.

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
