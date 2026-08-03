# torch.backends

> [!NOTE]
>
> API的**支持情况**中，&#10004;表示API支持在对应硬件环境上运行，&#10007;表示暂不支持。<br>

## 目录

- [base API](#base-api)

## base API

### torch.backends.cpu.get_cpu_capability

<div style="margin-left: 2em">

**原生文档**：[torch.backends.cpu.get_cpu_capability](https://pytorch.org/docs/2.12/backends.html#torch.backends.cpu.get_cpu_capability)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | &#10004; |
| <term>Atlas A3 训练系列产品</term> | &#10004; |
| <term>Ascend 950DT</term> | &#10007; |

</div>

### torch.backends.cuda.is_built

<div style="margin-left: 2em">

**原生文档**：[torch.backends.cuda.is_built](https://pytorch.org/docs/2.12/backends.html#torch.backends.cuda.is_built)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | &#10004; |
| <term>Atlas A3 训练系列产品</term> | &#10004; |
| <term>Ascend 950DT</term> | &#10007; |

</div>

### torch.backends.cuda.matmul.allow_tf32

<div style="margin-left: 2em">

**原生文档**：[torch.backends.cuda.matmul.allow_tf32](https://pytorch.org/docs/2.12/backends.html#torch.backends.cuda.matmul.allow_tf32)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | &#10004; |
| <term>Atlas A3 训练系列产品</term> | &#10004; |
| <term>Ascend 950DT</term> | &#10007; |

**限制与说明**： 支持fp32

</div>

### torch.backends.cuda.cufft_plan_cache

<div style="margin-left: 2em">

**原生文档**：[torch.backends.cuda.cufft_plan_cache](https://pytorch.org/docs/2.12/backends.html#torch.backends.cuda.cufft_plan_cache)

**NPU 形式名称**：torch.npu.backends.fft_plan_cache

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | &#10004; |
| <term>Atlas A3 训练系列产品</term> | &#10004; |
| <term>Ascend 950DT</term> | &#10007; |

</div>

### torch.backends.cuda.cufft_plan_cache.size

<div style="margin-left: 2em">

**原生文档**：[torch.backends.cuda.cufft_plan_cache.size](https://pytorch.org/docs/2.12/backends.html#torch.backends.cuda.cufft_plan_cache.size)

**NPU 形式名称**：torch.npu.backends.fft_plan_cache.size

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | &#10004; |
| <term>Atlas A3 训练系列产品</term> | &#10004; |
| <term>Ascend 950DT</term> | &#10007; |

</div>

### torch.backends.cuda.cufft_plan_cache.max_size

<div style="margin-left: 2em">

**原生文档**：[torch.backends.cuda.cufft_plan_cache.max_size](https://pytorch.org/docs/2.12/backends.html#torch.backends.cuda.cufft_plan_cache.max_size)

**NPU 形式名称**：torch.npu.backends.fft_plan_cache.max_size

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | &#10004; |
| <term>Atlas A3 训练系列产品</term> | &#10004; |
| <term>Ascend 950DT</term> | &#10007; |

**限制与说明**： 输入范围为1-99

</div>

### torch.backends.cuda.cufft_plan_cache.clear

<div style="margin-left: 2em">

**原生文档**：[torch.backends.cuda.cufft_plan_cache.clear](https://pytorch.org/docs/2.12/backends.html#torch.backends.cuda.cufft_plan_cache.clear)

**NPU 形式名称**：torch.npu.backends.fft_plan_cache.clear

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | &#10004; |
| <term>Atlas A3 训练系列产品</term> | &#10004; |
| <term>Ascend 950DT</term> | &#10007; |

</div>

### torch.backends.cuda.preferred_linalg_library

<div style="margin-left: 2em">

**原生文档**：[torch.backends.cuda.preferred_linalg_library](https://pytorch.org/docs/2.12/backends.html#torch.backends.cuda.preferred_linalg_library)

**产品支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | &#10007; |
| <term>Atlas A3 训练系列产品</term> | &#10007; |
| <term>Ascend 950DT</term> | &#10007; |

</div>

### torch.backends.cuda.flash_sdp_enabled

<div style="margin-left: 2em">

**原生文档**：[torch.backends.cuda.flash_sdp_enabled](https://pytorch.org/docs/2.12/backends.html#torch.backends.cuda.flash_sdp_enabled)

**产品支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | &#10007; |
| <term>Atlas A3 训练系列产品</term> | &#10007; |
| <term>Ascend 950DT</term> | &#10007; |

</div>

### torch.backends.cuda.enable_mem_efficient_sdp

<div style="margin-left: 2em">

**原生文档**：[torch.backends.cuda.enable_mem_efficient_sdp](https://pytorch.org/docs/2.12/backends.html#torch.backends.cuda.enable_mem_efficient_sdp)

**产品支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | &#10007; |
| <term>Atlas A3 训练系列产品</term> | &#10007; |
| <term>Ascend 950DT</term> | &#10007; |

</div>

### torch.backends.cuda.mem_efficient_sdp_enabled

<div style="margin-left: 2em">

**原生文档**：[torch.backends.cuda.mem_efficient_sdp_enabled](https://pytorch.org/docs/2.12/backends.html#torch.backends.cuda.mem_efficient_sdp_enabled)

**产品支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | &#10007; |
| <term>Atlas A3 训练系列产品</term> | &#10007; |
| <term>Ascend 950DT</term> | &#10007; |

</div>

### torch.backends.cuda.enable_flash_sdp

<div style="margin-left: 2em">

**原生文档**：[torch.backends.cuda.enable_flash_sdp](https://pytorch.org/docs/2.12/backends.html#torch.backends.cuda.enable_flash_sdp)

**产品支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | &#10007; |
| <term>Atlas A3 训练系列产品</term> | &#10007; |
| <term>Ascend 950DT</term> | &#10007; |

</div>

### torch.backends.cuda.math_sdp_enabled

<div style="margin-left: 2em">

**原生文档**：[torch.backends.cuda.math_sdp_enabled](https://pytorch.org/docs/2.12/backends.html#torch.backends.cuda.math_sdp_enabled)

**产品支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | &#10007; |
| <term>Atlas A3 训练系列产品</term> | &#10007; |
| <term>Ascend 950DT</term> | &#10007; |

</div>

### torch.backends.cuda.enable_math_sdp

<div style="margin-left: 2em">

**原生文档**：[torch.backends.cuda.enable_math_sdp](https://pytorch.org/docs/2.12/backends.html#torch.backends.cuda.enable_math_sdp)

**产品支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | &#10007; |
| <term>Atlas A3 训练系列产品</term> | &#10007; |
| <term>Ascend 950DT</term> | &#10007; |

</div>

### torch.backends.cuda.sdp_kernel

<div style="margin-left: 2em">

**原生文档**：[torch.backends.cuda.sdp_kernel](https://pytorch.org/docs/2.12/backends.html#torch.backends.cuda.sdp_kernel)

**产品支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | &#10007; |
| <term>Atlas A3 训练系列产品</term> | &#10007; |
| <term>Ascend 950DT</term> | &#10007; |

</div>

### torch.backends.cudnn.is_available

<div style="margin-left: 2em">

**原生文档**：[torch.backends.cudnn.is_available](https://pytorch.org/docs/2.12/backends.html#torch.backends.cudnn.is_available)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | &#10004; |
| <term>Atlas A3 训练系列产品</term> | &#10004; |
| <term>Ascend 950DT</term> | &#10007; |

**限制与说明**： 支持fp32

</div>

### torch.backends.cudnn.enabled

<div style="margin-left: 2em">

**原生文档**：[torch.backends.cudnn.enabled](https://pytorch.org/docs/2.12/backends.html#torch.backends.cudnn.enabled)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | &#10004; |
| <term>Atlas A3 训练系列产品</term> | &#10004; |
| <term>Ascend 950DT</term> | &#10007; |

**限制与说明**： 支持fp32

</div>

### torch.backends.cudnn.allow_tf32

<div style="margin-left: 2em">

**原生文档**：[torch.backends.cudnn.allow_tf32](https://pytorch.org/docs/2.12/backends.html#torch.backends.cudnn.allow_tf32)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | &#10004; |
| <term>Atlas A3 训练系列产品</term> | &#10004; |
| <term>Ascend 950DT</term> | &#10007; |

**限制与说明**： 支持fp32

</div>

### torch.backends.cudnn.deterministic

<div style="margin-left: 2em">

**原生文档**：[torch.backends.cudnn.deterministic](https://pytorch.org/docs/2.12/backends.html#torch.backends.cudnn.deterministic)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | &#10004; |
| <term>Atlas A3 训练系列产品</term> | &#10004; |
| <term>Ascend 950DT</term> | &#10007; |

</div>

### torch.backends.cudnn.benchmark

<div style="margin-left: 2em">

**原生文档**：[torch.backends.cudnn.benchmark](https://pytorch.org/docs/2.12/backends.html#torch.backends.cudnn.benchmark)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | &#10004; |
| <term>Atlas A3 训练系列产品</term> | &#10004; |
| <term>Ascend 950DT</term> | &#10007; |

**限制与说明**： 支持fp32

</div>

### torch.backends.cudnn.benchmark_limit

<div style="margin-left: 2em">

**原生文档**：[torch.backends.cudnn.benchmark_limit](https://pytorch.org/docs/2.12/backends.html#torch.backends.cudnn.benchmark_limit)

**产品支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | &#10007; |
| <term>Atlas A3 训练系列产品</term> | &#10007; |
| <term>Ascend 950DT</term> | &#10007; |

</div>

### torch.backends.mps.is_available

<div style="margin-left: 2em">

**原生文档**：[torch.backends.mps.is_available](https://pytorch.org/docs/2.12/backends.html#torch.backends.mps.is_available)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | &#10004; |
| <term>Atlas A3 训练系列产品</term> | &#10004; |
| <term>Ascend 950DT</term> | &#10007; |

</div>

### torch.backends.mps.is_built

<div style="margin-left: 2em">

**原生文档**：[torch.backends.mps.is_built](https://pytorch.org/docs/2.12/backends.html#torch.backends.mps.is_built)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | &#10004; |
| <term>Atlas A3 训练系列产品</term> | &#10004; |
| <term>Ascend 950DT</term> | &#10007; |

</div>

### torch.backends.mkl.is_available

<div style="margin-left: 2em">

**原生文档**：[torch.backends.mkl.is_available](https://pytorch.org/docs/2.12/backends.html#torch.backends.mkl.is_available)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | &#10004; |
| <term>Atlas A3 训练系列产品</term> | &#10004; |
| <term>Ascend 950DT</term> | &#10007; |

</div>

### torch.backends.mkl.verbose

<div style="margin-left: 2em">

**原生文档**：[torch.backends.mkl.verbose](https://pytorch.org/docs/2.12/backends.html#torch.backends.mkl.verbose)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | &#10004; |
| <term>Atlas A3 训练系列产品</term> | &#10004; |
| <term>Ascend 950DT</term> | &#10007; |

</div>

### torch.backends.mkldnn.is_available

<div style="margin-left: 2em">

**原生文档**：[torch.backends.mkldnn.is_available](https://pytorch.org/docs/2.12/backends.html#torch.backends.mkldnn.is_available)

**产品支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas 训练系列产品</term> | &#10004; |
| <term>Atlas A2 训练系列产品</term> | &#10004; |
| <term>Atlas A3 训练系列产品</term> | &#10007; |
| <term>Ascend 950DT</term> | &#10007; |

</div>

### torch.backends.mkldnn.verbose

<div style="margin-left: 2em">

**原生文档**：[torch.backends.mkldnn.verbose](https://pytorch.org/docs/2.12/backends.html#torch.backends.mkldnn.verbose)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | &#10004; |
| <term>Atlas A3 训练系列产品</term> | &#10004; |
| <term>Ascend 950DT</term> | &#10007; |

</div>

### torch.backends.openmp.is_available

<div style="margin-left: 2em">

**原生文档**：[torch.backends.openmp.is_available](https://pytorch.org/docs/2.12/backends.html#torch.backends.openmp.is_available)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | &#10004; |
| <term>Atlas A3 训练系列产品</term> | &#10004; |
| <term>Ascend 950DT</term> | &#10007; |

**限制与说明**： 支持fp32

</div>

### torch.backends.opt_einsum.is_available

<div style="margin-left: 2em">

**原生文档**：[torch.backends.opt_einsum.is_available](https://pytorch.org/docs/2.12/backends.html#torch.backends.opt_einsum.is_available)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | &#10004; |
| <term>Atlas A3 训练系列产品</term> | &#10004; |
| <term>Ascend 950DT</term> | &#10007; |

**限制与说明**： 支持fp32

</div>

### torch.backends.opt_einsum.get_opt_einsum

<div style="margin-left: 2em">

**原生文档**：[torch.backends.opt_einsum.get_opt_einsum](https://pytorch.org/docs/2.12/backends.html#torch.backends.opt_einsum.get_opt_einsum)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | &#10004; |
| <term>Atlas A3 训练系列产品</term> | &#10004; |
| <term>Ascend 950DT</term> | &#10007; |

**限制与说明**： 支持fp32

</div>

### torch.backends.opt_einsum.enabled

<div style="margin-left: 2em">

**原生文档**：[torch.backends.opt_einsum.enabled](https://pytorch.org/docs/2.12/backends.html#torch.backends.opt_einsum.enabled)

**产品支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | &#10007; |
| <term>Atlas A3 训练系列产品</term> | &#10007; |
| <term>Ascend 950DT</term> | &#10007; |

</div>

### torch.backends.opt_einsum.strategy

<div style="margin-left: 2em">

**原生文档**：[torch.backends.opt_einsum.strategy](https://pytorch.org/docs/2.12/backends.html#torch.backends.opt_einsum.strategy)

**产品支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | &#10007; |
| <term>Atlas A3 训练系列产品</term> | &#10007; |
| <term>Ascend 950DT</term> | &#10007; |

</div>
