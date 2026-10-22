# torch.backends

> [!NOTE]
>
> - If an API has "Constraints & Notes", it means that the support level of the API on the Ascend NPU differs from the native version. Refer to the specific notes to ensure compatibility with the Ascend NPU platform.
> - Some APIs exist in the [PyTorch Community Documentation](https://pytorch.org/docs/2.10/) but are not included in this support list. These APIs have not been verified yet. Use them with caution. We will continue the verification work and update the document upon completion.
> - Product support scope: This document only provides verified product information. Unverified products are temporarily excluded.
> - The modules listed in this directory are consistent with the native documentation. For module-related notes, refer to the native documentation [LINK](https://pytorch.org/docs/2.10/backends.html).

<div style="border:1px solid #d1d5da;margin:10px 0;padding:16px 20px;background-color:#f3f4f5;border-radius:.25rem">
<div style="margin: 8px 0"><font size="5"><b>Contents</b></font></div>

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

**Native Documentation**: [torch.backends.cpu.get_cpu_capability](https://pytorch.org/docs/2.10/backends.html#torch.backends.cpu.get_cpu_capability)

**Support Status**:

<!-- npu="910b" id1 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id2 -->
<!-- npu="950" id3 -->
- <term>Ascend 950DT products</term>: Not Supported
<!-- end id3 -->

</div>

## torch.backends.cuda

### torch.backends.cuda.is_built

<div style="margin-left: 2em">

**Native Documentation**: [torch.backends.cuda.is_built](https://pytorch.org/docs/2.10/backends.html#torch.backends.cuda.is_built)

**Support Status**:

<!-- npu="910b" id4 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id4 -->
<!-- npu="A3" id5 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id5 -->
<!-- npu="950" id6 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id6 -->

</div>

### torch.backends.cuda.matmul.allow_tf32

<div style="margin-left: 2em">

**Native Documentation**: [torch.backends.cuda.matmul.allow_tf32](https://pytorch.org/docs/2.10/backends.html#torch.backends.cuda.matmul.allow_tf32)

**Support Status**:

<!-- npu="910b" id7 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id7 -->
<!-- npu="A3" id8 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id8 -->
<!-- npu="950" id9 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id9 -->

**Constraints & Notes**: `input` only supports fp32. Although this API can be called on the NPU, it has no actual effect. To implement similar functionality, use [torch_npu.npu.matmul.allow_hf32](https://gitcode.com/Ascend/op-plugin/blob/master/docs/en/custom_APIs/torch_npu-npu/torch_npu-npu-matmul-allow_hf32.md).

</div>

### torch.backends.cuda.cufft_plan_cache

<div style="margin-left: 2em">

**Native Documentation**: [torch.backends.cuda.cufft_plan_cache](https://pytorch.org/docs/2.10/backends.html#torch.backends.cuda.cufft_plan_cache)

**NPU-Adapted Name**: torch.npu.backends.fft_plan_cache

**Support Status**:

<!-- npu="910b" id10 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id10 -->
<!-- npu="A3" id11 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id11 -->
<!-- npu="950" id12 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id12 -->

</div>

### torch.backends.cuda.cufft_plan_cache.size

<div style="margin-left: 2em">

**Native Documentation**: [torch.backends.cuda.cufft_plan_cache.size](https://pytorch.org/docs/2.10/backends.html#torch.backends.cuda.cufft_plan_cache.size)

**NPU-Adapted Name**: torch.npu.backends.fft_plan_cache.size

**Support Status**:

<!-- npu="910b" id13 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id13 -->
<!-- npu="A3" id14 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id14 -->
<!-- npu="950" id15 -->
- <term>Ascend 950DT products</term>: Not Supported
<!-- end id15 -->

</div>

### torch.backends.cuda.cufft_plan_cache.max_size

<div style="margin-left: 2em">

**Native Documentation**: [torch.backends.cuda.cufft_plan_cache.max_size](https://pytorch.org/docs/2.10/backends.html#torch.backends.cuda.cufft_plan_cache.max_size)

**NPU-Adapted Name**: torch.npu.backends.fft_plan_cache.max_size

**Support Status**:

<!-- npu="910b" id16 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id16 -->
<!-- npu="A3" id17 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id17 -->
<!-- npu="950" id18 -->
- <term>Ascend 950DT products</term>: Not Supported
<!-- end id18 -->

**Constraints & Notes**: The input range is 1-99.

</div>

### torch.backends.cuda.cufft_plan_cache.clear

<div style="margin-left: 2em">

**Native Documentation**: [torch.backends.cuda.cufft_plan_cache.clear](https://pytorch.org/docs/2.10/backends.html#torch.backends.cuda.cufft_plan_cache.clear)

**NPU-Adapted Name**: torch.npu.backends.fft_plan_cache.clear

**Support Status**:

<!-- npu="910b" id19 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id19 -->
<!-- npu="A3" id20 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id20 -->
<!-- npu="950" id21 -->
- <term>Ascend 950DT products</term>: Not Supported
<!-- end id21 -->

</div>

### torch.backends.cuda.preferred_linalg_library

<div style="margin-left: 2em">

**Native Documentation**: [torch.backends.cuda.preferred_linalg_library](https://pytorch.org/docs/2.10/backends.html#torch.backends.cuda.preferred_linalg_library)

**Support Status**:

<!-- npu="910b" id22 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id22 -->
<!-- npu="A3" id23 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id23 -->
<!-- npu="950" id24 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id24 -->

</div>

### torch.backends.cuda.flash_sdp_enabled

<div style="margin-left: 2em">

**Native Documentation**: [torch.backends.cuda.flash_sdp_enabled](https://pytorch.org/docs/2.10/backends.html#torch.backends.cuda.flash_sdp_enabled)

**Support Status**:

<!-- npu="910b" id25 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id25 -->
<!-- npu="A3" id26 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id26 -->
<!-- npu="950" id27 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id27 -->

</div>

### torch.backends.cuda.enable_mem_efficient_sdp

<div style="margin-left: 2em">

**Native Documentation**: [torch.backends.cuda.enable_mem_efficient_sdp](https://pytorch.org/docs/2.10/backends.html#torch.backends.cuda.enable_mem_efficient_sdp)

**Support Status**:

<!-- npu="910b" id28 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id28 -->
<!-- npu="A3" id29 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id29 -->
<!-- npu="950" id30 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id30 -->

</div>

### torch.backends.cuda.mem_efficient_sdp_enabled

<div style="margin-left: 2em">

**Native Documentation**: [torch.backends.cuda.mem_efficient_sdp_enabled](https://pytorch.org/docs/2.10/backends.html#torch.backends.cuda.mem_efficient_sdp_enabled)

**Support Status**:

<!-- npu="910b" id31 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id31 -->
<!-- npu="A3" id32 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id32 -->
<!-- npu="950" id33 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id33 -->

</div>

### torch.backends.cuda.enable_flash_sdp

<div style="margin-left: 2em">

**Native Documentation**: [torch.backends.cuda.enable_flash_sdp](https://pytorch.org/docs/2.10/backends.html#torch.backends.cuda.enable_flash_sdp)

**Support Status**:

<!-- npu="910b" id34 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id34 -->
<!-- npu="A3" id35 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id35 -->
<!-- npu="950" id36 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id36 -->

</div>

### torch.backends.cuda.math_sdp_enabled

<div style="margin-left: 2em">

**Native Documentation**: [torch.backends.cuda.math_sdp_enabled](https://pytorch.org/docs/2.10/backends.html#torch.backends.cuda.math_sdp_enabled)

**Support Status**:

<!-- npu="910b" id37 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id37 -->
<!-- npu="A3" id38 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id38 -->
<!-- npu="950" id39 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id39 -->

</div>

### torch.backends.cuda.enable_math_sdp

<div style="margin-left: 2em">

**Native Documentation**: [torch.backends.cuda.enable_math_sdp](https://pytorch.org/docs/2.10/backends.html#torch.backends.cuda.enable_math_sdp)

**Support Status**:

<!-- npu="910b" id40 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id40 -->
<!-- npu="A3" id41 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id41 -->
<!-- npu="950" id42 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id42 -->

</div>

### torch.backends.cuda.sdp_kernel

<div style="margin-left: 2em">

**Native Documentation**: [torch.backends.cuda.sdp_kernel](https://pytorch.org/docs/2.10/backends.html#torch.backends.cuda.sdp_kernel)

**Support Status**:

<!-- npu="910b" id43 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id43 -->
<!-- npu="A3" id44 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id44 -->
<!-- npu="950" id45 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id45 -->

</div>

## torch.backends.cudnn

### torch.backends.cudnn.is_available

<div style="margin-left: 2em">

**Native Documentation**: [torch.backends.cudnn.is_available](https://pytorch.org/docs/2.10/backends.html#torch.backends.cudnn.is_available)

**Support Status**:

<!-- npu="910b" id46 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id46 -->
<!-- npu="A3" id47 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id47 -->
<!-- npu="950" id48 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id48 -->

**Constraints & Notes**: `input` only supports fp32

</div>

### torch.backends.cudnn.enabled

<div style="margin-left: 2em">

**Native Documentation**: [torch.backends.cudnn.enabled](https://pytorch.org/docs/2.10/backends.html#torch.backends.cudnn.enabled)

**Support Status**:

<!-- npu="910b" id49 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id49 -->
<!-- npu="A3" id50 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id50 -->
<!-- npu="950" id51 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id51 -->

**Constraints & Notes**: `input` only supports fp32

</div>

### torch.backends.cudnn.allow_tf32

<div style="margin-left: 2em">

**Native Documentation**: [torch.backends.cudnn.allow_tf32](https://pytorch.org/docs/2.10/backends.html#torch.backends.cudnn.allow_tf32)

**Support Status**:

<!-- npu="910b" id52 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id52 -->
<!-- npu="A3" id53 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id53 -->
<!-- npu="950" id54 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id54 -->

**Constraints & Notes**: `input` only supports fp32

</div>

### torch.backends.cudnn.deterministic

<div style="margin-left: 2em">

**Native Documentation**: [torch.backends.cudnn.deterministic](https://pytorch.org/docs/2.10/backends.html#torch.backends.cudnn.deterministic)

**Support Status**:

<!-- npu="910b" id55 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id55 -->
<!-- npu="A3" id56 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id56 -->
<!-- npu="950" id57 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id57 -->

</div>

### torch.backends.cudnn.benchmark

<div style="margin-left: 2em">

**Native Documentation**: [torch.backends.cudnn.benchmark](https://pytorch.org/docs/2.10/backends.html#torch.backends.cudnn.benchmark)

**Support Status**:

<!-- npu="910b" id58 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id58 -->
<!-- npu="A3" id59 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id59 -->
<!-- npu="950" id60 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id60 -->

**Constraints & Notes**: `input` only supports fp32

</div>

### torch.backends.cudnn.benchmark_limit

<div style="margin-left: 2em">

**Native Documentation**: [torch.backends.cudnn.benchmark_limit](https://pytorch.org/docs/2.10/backends.html#torch.backends.cudnn.benchmark_limit)

**Support Status**:

<!-- npu="910b" id61 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id61 -->
<!-- npu="A3" id62 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id62 -->
<!-- npu="950" id63 -->
- <term>Ascend 950DT products</term>: Not Supported
<!-- end id63 -->

</div>

## torch.backends.mps

### torch.backends.mps.is_available

<div style="margin-left: 2em">

**Native Documentation**: [torch.backends.mps.is_available](https://pytorch.org/docs/2.10/backends.html#torch.backends.mps.is_available)

**Support Status**:

<!-- npu="910b" id64 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id64 -->
<!-- npu="A3" id65 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id65 -->
<!-- npu="950" id66 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id66 -->

</div>

### torch.backends.mps.is_built

<div style="margin-left: 2em">

**Native Documentation**: [torch.backends.mps.is_built](https://pytorch.org/docs/2.10/backends.html#torch.backends.mps.is_built)

**Support Status**:

<!-- npu="910b" id67 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id67 -->
<!-- npu="A3" id68 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id68 -->
<!-- npu="950" id69 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id69 -->

</div>

## torch.backends.mkl

### torch.backends.mkl.is_available

<div style="margin-left: 2em">

**Native Documentation**: [torch.backends.mkl.is_available](https://pytorch.org/docs/2.10/backends.html#torch.backends.mkl.is_available)

**Support Status**:

<!-- npu="910b" id70 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id70 -->
<!-- npu="A3" id71 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id71 -->
<!-- npu="950" id72 -->
- <term>Ascend 950DT products</term>: Not Supported
<!-- end id72 -->

</div>

### torch.backends.mkl.verbose

<div style="margin-left: 2em">

**Native Documentation**: [torch.backends.mkl.verbose](https://pytorch.org/docs/2.10/backends.html#torch.backends.mkl.verbose)

**Support Status**:

<!-- npu="910b" id73 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id73 -->
<!-- npu="A3" id74 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id74 -->
<!-- npu="950" id75 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id75 -->

</div>

## torch.backends.mkldnn

### torch.backends.mkldnn.is_available

<div style="margin-left: 2em">

**Native Documentation**: [torch.backends.mkldnn.is_available](https://pytorch.org/docs/2.10/backends.html#torch.backends.mkldnn.is_available)

**Support Status**:

<!-- npu="910" id76 -->
- <term>Atlas training products</term>: Supported
<!-- end id76 -->
<!-- npu="910b" id77 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id77 -->
<!-- npu="A3" id78 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id78 -->
<!-- npu="950" id79 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id79 -->

</div>

### torch.backends.mkldnn.verbose

<div style="margin-left: 2em">

**Native Documentation**: [torch.backends.mkldnn.verbose](https://pytorch.org/docs/2.10/backends.html#torch.backends.mkldnn.verbose)

**Support Status**:

<!-- npu="910b" id80 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id80 -->
<!-- npu="A3" id81 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id81 -->
<!-- npu="950" id82 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id82 -->

</div>

## torch.backends.openmp

### torch.backends.openmp.is_available

<div style="margin-left: 2em">

**Native Documentation**: [torch.backends.openmp.is_available](https://pytorch.org/docs/2.10/backends.html#torch.backends.openmp.is_available)

**Support Status**:

<!-- npu="910b" id83 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id83 -->
<!-- npu="A3" id84 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id84 -->
<!-- npu="950" id85 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id85 -->

**Constraints & Notes**: `input` only supports fp32

</div>

## torch.backends.opt_einsum

### torch.backends.opt_einsum.is_available

<div style="margin-left: 2em">

**Native Documentation**: [torch.backends.opt_einsum.is_available](https://pytorch.org/docs/2.10/backends.html#torch.backends.opt_einsum.is_available)

**Support Status**:

<!-- npu="910b" id86 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id86 -->
<!-- npu="A3" id87 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id87 -->
<!-- npu="950" id88 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id88 -->

**Constraints & Notes**: `input` only supports fp32

</div>

### torch.backends.opt_einsum.get_opt_einsum

<div style="margin-left: 2em">

**Native Documentation**: [torch.backends.opt_einsum.get_opt_einsum](https://pytorch.org/docs/2.10/backends.html#torch.backends.opt_einsum.get_opt_einsum)

**Support Status**:

<!-- npu="910b" id89 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id89 -->
<!-- npu="A3" id90 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id90 -->
<!-- npu="950" id91 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id91 -->

**Constraints & Notes**: `input` only supports fp32

</div>

### torch.backends.opt_einsum.enabled

<div style="margin-left: 2em">

**Native Documentation**: [torch.backends.opt_einsum.enabled](https://pytorch.org/docs/2.10/backends.html#torch.backends.opt_einsum.enabled)

**Support Status**:

<!-- npu="910b" id92 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id92 -->
<!-- npu="A3" id93 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id93 -->
<!-- npu="950" id94 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id94 -->

</div>

### torch.backends.opt_einsum.strategy

<div style="margin-left: 2em">

**Native Documentation**: [torch.backends.opt_einsum.strategy](https://pytorch.org/docs/2.10/backends.html#torch.backends.opt_einsum.strategy)

**Support Status**:

<!-- npu="910b" id95 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id95 -->
<!-- npu="A3" id96 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id96 -->
<!-- npu="950" id97 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id97 -->

</div>
