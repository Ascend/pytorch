# torch.backends

> [!NOTE]
>
> - 若API标注有“限制与说明”，表示该API在昇腾NPU上的支持度和原生版本存在差异，请务必查阅具体说明，以确保适配昇腾NPU平台。
> - 部分API虽在[PyTorch社区文档](https://pytorch.org/docs/2.11/)中存在，但未收录于本支持清单。此类API尚未验证，请谨慎使用。我们将持续进行验证工作，并在验证完成后更新文档。
> - 产品支持范围说明：文档中仅提供已验证的产品信息，未经过验证产品暂不纳入。
> - 目录下罗列的模块和原生文档一致，对于模块的相关说明请查看原生文档[LINK](https://pytorch.org/docs/2.11/backends.html)。

<div style="border:1px solid #d1d5da;margin:10px 0;padding:16px 20px;background-color:#f3f4f5;border-radius:.25rem">
<div style="margin: 8px 0"><font size="5"><b>目录</b></font></div>

- [torch.backends.cpu](#torchbackendscpu)
- [torch.backends.cuda](#torchbackendscuda)
- [torch.backends.cudnn](#torchbackendscudnn)
- [torch.backends.mps](#torchbackendsmps)
- [torch.backends.mkl](#torchbackendsmkl)
- [torch.backends.mkldnn](#torchbackendsmkldnn)
- [torch.backends.openmp](#torchbackendsopenmp)
- [torch.backends.opt_einsum](#torchbackendsopt_einsum)

</div>

<div style="display:none;">

## &#8203;torch.backends

</div>

## torch.backends.cpu

### torch.backends.cpu.get_cpu_capability

<div style="margin-left: 2em">

**原生文档**：[torch.backends.cpu.get_cpu_capability](https://pytorch.org/docs/2.11/backends.html#torch.backends.cpu.get_cpu_capability)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

</div>

## torch.backends.cuda

### torch.backends.cuda.is_built

<div style="margin-left: 2em">

**原生文档**：[torch.backends.cuda.is_built](https://pytorch.org/docs/2.11/backends.html#torch.backends.cuda.is_built)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

</div>

### torch.backends.cuda.matmul.allow_tf32

<div style="margin-left: 2em">

**原生文档**：[torch.backends.cuda.matmul.allow_tf32](https://pytorch.org/docs/2.11/backends.html#torch.backends.cuda.matmul.allow_tf32)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

**限制与说明**：`input`仅支持fp32

</div>

### torch.backends.cuda.cufft_plan_cache

<div style="margin-left: 2em">

**原生文档**：[torch.backends.cuda.cufft_plan_cache](https://pytorch.org/docs/2.11/backends.html#torch.backends.cuda.cufft_plan_cache)

**NPU 形式名称**：torch.npu.backends.fft_plan_cache

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

</div>

### torch.backends.cuda.cufft_plan_cache.size

<div style="margin-left: 2em">

**原生文档**：[torch.backends.cuda.cufft_plan_cache.size](https://pytorch.org/docs/2.11/backends.html#torch.backends.cuda.cufft_plan_cache.size)

**NPU 形式名称**：torch.npu.backends.fft_plan_cache.size

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

</div>

### torch.backends.cuda.cufft_plan_cache.max_size

<div style="margin-left: 2em">

**原生文档**：[torch.backends.cuda.cufft_plan_cache.max_size](https://pytorch.org/docs/2.11/backends.html#torch.backends.cuda.cufft_plan_cache.max_size)

**NPU 形式名称**：torch.npu.backends.fft_plan_cache.max_size

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

**限制与说明**： 输入范围为1-99

</div>

### torch.backends.cuda.cufft_plan_cache.clear

<div style="margin-left: 2em">

**原生文档**：[torch.backends.cuda.cufft_plan_cache.clear](https://pytorch.org/docs/2.11/backends.html#torch.backends.cuda.cufft_plan_cache.clear)

**NPU 形式名称**：torch.npu.backends.fft_plan_cache.clear

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

</div>

### torch.backends.cuda.preferred_linalg_library

<div style="margin-left: 2em">

**原生文档**：[torch.backends.cuda.preferred_linalg_library](https://pytorch.org/docs/2.11/backends.html#torch.backends.cuda.preferred_linalg_library)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：不支持
- <term>Atlas A3 训练系列产品</term>：不支持
- <term>Ascend 950DT</term>：不支持

</div>

### torch.backends.cuda.flash_sdp_enabled

<div style="margin-left: 2em">

**原生文档**：[torch.backends.cuda.flash_sdp_enabled](https://pytorch.org/docs/2.11/backends.html#torch.backends.cuda.flash_sdp_enabled)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：不支持
- <term>Atlas A3 训练系列产品</term>：不支持
- <term>Ascend 950DT</term>：不支持

</div>

### torch.backends.cuda.enable_mem_efficient_sdp

<div style="margin-left: 2em">

**原生文档**：[torch.backends.cuda.enable_mem_efficient_sdp](https://pytorch.org/docs/2.11/backends.html#torch.backends.cuda.enable_mem_efficient_sdp)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：不支持
- <term>Atlas A3 训练系列产品</term>：不支持
- <term>Ascend 950DT</term>：不支持

</div>

### torch.backends.cuda.mem_efficient_sdp_enabled

<div style="margin-left: 2em">

**原生文档**：[torch.backends.cuda.mem_efficient_sdp_enabled](https://pytorch.org/docs/2.11/backends.html#torch.backends.cuda.mem_efficient_sdp_enabled)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：不支持
- <term>Atlas A3 训练系列产品</term>：不支持
- <term>Ascend 950DT</term>：不支持

</div>

### torch.backends.cuda.enable_flash_sdp

<div style="margin-left: 2em">

**原生文档**：[torch.backends.cuda.enable_flash_sdp](https://pytorch.org/docs/2.11/backends.html#torch.backends.cuda.enable_flash_sdp)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：不支持
- <term>Atlas A3 训练系列产品</term>：不支持
- <term>Ascend 950DT</term>：不支持

</div>

### torch.backends.cuda.math_sdp_enabled

<div style="margin-left: 2em">

**原生文档**：[torch.backends.cuda.math_sdp_enabled](https://pytorch.org/docs/2.11/backends.html#torch.backends.cuda.math_sdp_enabled)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：不支持
- <term>Atlas A3 训练系列产品</term>：不支持
- <term>Ascend 950DT</term>：不支持

</div>

### torch.backends.cuda.enable_math_sdp

<div style="margin-left: 2em">

**原生文档**：[torch.backends.cuda.enable_math_sdp](https://pytorch.org/docs/2.11/backends.html#torch.backends.cuda.enable_math_sdp)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：不支持
- <term>Atlas A3 训练系列产品</term>：不支持
- <term>Ascend 950DT</term>：不支持

</div>

### torch.backends.cuda.sdp_kernel

<div style="margin-left: 2em">

**原生文档**：[torch.backends.cuda.sdp_kernel](https://pytorch.org/docs/2.11/backends.html#torch.backends.cuda.sdp_kernel)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：不支持
- <term>Atlas A3 训练系列产品</term>：不支持
- <term>Ascend 950DT</term>：不支持

</div>

## torch.backends.cudnn

### torch.backends.cudnn.is_available

<div style="margin-left: 2em">

**原生文档**：[torch.backends.cudnn.is_available](https://pytorch.org/docs/2.11/backends.html#torch.backends.cudnn.is_available)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

**限制与说明**：`input`仅支持fp32

</div>

### torch.backends.cudnn.enabled

<div style="margin-left: 2em">

**原生文档**：[torch.backends.cudnn.enabled](https://pytorch.org/docs/2.11/backends.html#torch.backends.cudnn.enabled)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

**限制与说明**：`input`仅支持fp32

</div>

### torch.backends.cudnn.allow_tf32

<div style="margin-left: 2em">

**原生文档**：[torch.backends.cudnn.allow_tf32](https://pytorch.org/docs/2.11/backends.html#torch.backends.cudnn.allow_tf32)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

**限制与说明**：`input`仅支持fp32

</div>

### torch.backends.cudnn.deterministic

<div style="margin-left: 2em">

**原生文档**：[torch.backends.cudnn.deterministic](https://pytorch.org/docs/2.11/backends.html#torch.backends.cudnn.deterministic)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

</div>

### torch.backends.cudnn.benchmark

<div style="margin-left: 2em">

**原生文档**：[torch.backends.cudnn.benchmark](https://pytorch.org/docs/2.11/backends.html#torch.backends.cudnn.benchmark)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

**限制与说明**：`input`仅支持fp32

</div>

### torch.backends.cudnn.benchmark_limit

<div style="margin-left: 2em">

**原生文档**：[torch.backends.cudnn.benchmark_limit](https://pytorch.org/docs/2.11/backends.html#torch.backends.cudnn.benchmark_limit)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：不支持
- <term>Atlas A3 训练系列产品</term>：不支持
- <term>Ascend 950DT</term>：不支持

</div>

## torch.backends.mps

### torch.backends.mps.is_available

<div style="margin-left: 2em">

**原生文档**：[torch.backends.mps.is_available](https://pytorch.org/docs/2.11/backends.html#torch.backends.mps.is_available)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

</div>

### torch.backends.mps.is_built

<div style="margin-left: 2em">

**原生文档**：[torch.backends.mps.is_built](https://pytorch.org/docs/2.11/backends.html#torch.backends.mps.is_built)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

</div>

## torch.backends.mkl

### torch.backends.mkl.is_available

<div style="margin-left: 2em">

**原生文档**：[torch.backends.mkl.is_available](https://pytorch.org/docs/2.11/backends.html#torch.backends.mkl.is_available)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

</div>

### torch.backends.mkl.verbose

<div style="margin-left: 2em">

**原生文档**：[torch.backends.mkl.verbose](https://pytorch.org/docs/2.11/backends.html#torch.backends.mkl.verbose)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

</div>

## torch.backends.mkldnn

### torch.backends.mkldnn.is_available

<div style="margin-left: 2em">

**原生文档**：[torch.backends.mkldnn.is_available](https://pytorch.org/docs/2.11/backends.html#torch.backends.mkldnn.is_available)

**产品支持情况**：

- <term>Atlas 训练系列产品</term>：支持
- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：不支持
- <term>Ascend 950DT</term>：不支持

</div>

### torch.backends.mkldnn.verbose

<div style="margin-left: 2em">

**原生文档**：[torch.backends.mkldnn.verbose](https://pytorch.org/docs/2.11/backends.html#torch.backends.mkldnn.verbose)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

</div>

## torch.backends.openmp

### torch.backends.openmp.is_available

<div style="margin-left: 2em">

**原生文档**：[torch.backends.openmp.is_available](https://pytorch.org/docs/2.11/backends.html#torch.backends.openmp.is_available)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

**限制与说明**：`input`仅支持fp32

</div>

## torch.backends.opt_einsum

### torch.backends.opt_einsum.is_available

<div style="margin-left: 2em">

**原生文档**：[torch.backends.opt_einsum.is_available](https://pytorch.org/docs/2.11/backends.html#torch.backends.opt_einsum.is_available)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

**限制与说明**：`input`仅支持fp32

</div>

### torch.backends.opt_einsum.get_opt_einsum

<div style="margin-left: 2em">

**原生文档**：[torch.backends.opt_einsum.get_opt_einsum](https://pytorch.org/docs/2.11/backends.html#torch.backends.opt_einsum.get_opt_einsum)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

**限制与说明**：`input`仅支持fp32

</div>

### torch.backends.opt_einsum.enabled

<div style="margin-left: 2em">

**原生文档**：[torch.backends.opt_einsum.enabled](https://pytorch.org/docs/2.11/backends.html#torch.backends.opt_einsum.enabled)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：不支持
- <term>Atlas A3 训练系列产品</term>：不支持
- <term>Ascend 950DT</term>：不支持

</div>

### torch.backends.opt_einsum.strategy

<div style="margin-left: 2em">

**原生文档**：[torch.backends.opt_einsum.strategy](https://pytorch.org/docs/2.11/backends.html#torch.backends.opt_einsum.strategy)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：不支持
- <term>Atlas A3 训练系列产品</term>：不支持
- <term>Ascend 950DT</term>：不支持

</div>
