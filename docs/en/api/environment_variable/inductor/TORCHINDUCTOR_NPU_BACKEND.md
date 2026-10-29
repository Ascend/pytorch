# TORCHINDUCTOR\_NPU\_BACKEND

## Function Description

This environment variable configures the compilation mode in graph mode (Inductor), supporting switching among Triton, DVM, Ascend C, and other modes.

- Default value `default`: uses the default Triton mode.
- Set to `dvm`: uses DVM mode.
- Set to `ascendc`: uses Ascend C mode.

> [!NOTE]
>
> This environment variable can be set before the first call to `torch.compile` (the backend is lazily loaded at the first compilation, so setting it after `import torch`/`import torch_npu` also takes effect).

PyTorch configures backend behavior through [`torch._inductor.config`](https://github.com/pytorch/pytorch/blob/main/torch/_inductor/config.py); TorchNPU provides this environment variable to select the Inductor compilation mode on NPU.

## Configuration Example

Using the default Triton mode:

```bash
export TORCHINDUCTOR_NPU_BACKEND="default"
```

Using DVM mode:

```bash
export TORCHINDUCTOR_NPU_BACKEND="dvm"
```

Using Ascend C mode:

```bash
export TORCHINDUCTOR_NPU_BACKEND="ascendc"
```

## Usage Constraints

- Ascend C mode requires PyTorch 2.9.0 or later.
- Only the Triton, DVM, and Ascend C modes of the Inductor backend compiler are supported.

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
