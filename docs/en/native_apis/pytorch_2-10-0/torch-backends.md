# torch.backends

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-06-14T07:54:12.266Z pushedAt=2026-06-14T09:16:34.717Z -->

> [!NOTE]
> If the "Supported" column for an API is "Yes" and the "Restrictions and Notes" column is "-", it means the API support is consistent with the native API.

|API Name|NPU Form Name|Supported|Restrictions and Notes|
|--|--|--|--|
|[torch.backends.cpu.get_cpu_capability](https://pytorch.org/docs/2.10/backends.html#torch.backends.cpu.get_cpu_capability)|-|Yes|-|
|[torch.backends.cuda.is_built](https://pytorch.org/docs/2.10/backends.html#torch.backends.cuda.is_built)|-|Yes|-|
|[torch.backends.cuda.matmul.allow_tf32](https://pytorch.org/docs/2.10/backends.html#torch.backends.cuda.matmul.allow_tf32)|-|Yes|Supports fp32|
|[torch.backends.cuda.cufft_plan_cache](https://pytorch.org/docs/2.10/backends.html#torch.backends.cuda.cufft_plan_cache)|torch.npu.backends.fft_plan_cache|Yes|-|
|[torch.backends.cuda.cufft_plan_cache.size](https://pytorch.org/docs/2.10/backends.html#torch.backends.cuda.cufft_plan_cache.size)|torch.npu.backends.fft_plan_cache.size|Yes|-|
|[torch.backends.cuda.cufft_plan_cache.max_size](https://pytorch.org/docs/2.10/backends.html#torch.backends.cuda.cufft_plan_cache.max_size)|torch.npu.backends.fft_plan_cache.max_size|Yes|Input range is 1-99|
|[torch.backends.cuda.cufft_plan_cache.clear](https://pytorch.org/docs/2.10/backends.html#torch.backends.cuda.cufft_plan_cache.clear)|torch.npu.backends.fft_plan_cache.clear|Yes|-|
|[torch.backends.cuda.preferred_linalg_library](https://pytorch.org/docs/2.10/backends.html#torch.backends.cuda.preferred_linalg_library)|-|No|-|
|[torch.backends.cuda.flash_sdp_enabled](https://pytorch.org/docs/2.10/backends.html#torch.backends.cuda.flash_sdp_enabled)|-|No|-|
|[torch.backends.cuda.enable_mem_efficient_sdp](https://pytorch.org/docs/2.10/backends.html#torch.backends.cuda.enable_mem_efficient_sdp)|-|No|-|
|[torch.backends.cuda.mem_efficient_sdp_enabled](https://pytorch.org/docs/2.10/backends.html#torch.backends.cuda.mem_efficient_sdp_enabled)|-|No|-|
|[torch.backends.cuda.enable_flash_sdp](https://pytorch.org/docs/2.10/backends.html#torch.backends.cuda.enable_flash_sdp)|-|No|-|
|[torch.backends.cuda.math_sdp_enabled](https://pytorch.org/docs/2.10/backends.html#torch.backends.cuda.math_sdp_enabled)|-|No|-|
|[torch.backends.cuda.enable_math_sdp](https://pytorch.org/docs/2.10/backends.html#torch.backends.cuda.enable_math_sdp)|-|No|-|
|[torch.backends.cuda.sdp_kernel](https://pytorch.org/docs/2.10/backends.html#torch.backends.cuda.sdp_kernel)|-|No|-|
|[torch.backends.cudnn.is_available](https://pytorch.org/docs/2.10/backends.html#torch.backends.cudnn.is_available)|-|Yes|Supports fp32|
|[torch.backends.cudnn.enabled](https://pytorch.org/docs/2.10/backends.html#torch.backends.cudnn.enabled)|-|Yes|Supports fp32|
|[torch.backends.cudnn.allow_tf32](https://pytorch.org/docs/2.10/backends.html#torch.backends.cudnn.allow_tf32)|-|Yes|Supports fp32|
|[torch.backends.cudnn.deterministic](https://pytorch.org/docs/2.10/backends.html#torch.backends.cudnn.deterministic)|-|Yes|-|
|[torch.backends.cudnn.benchmark](https://pytorch.org/docs/2.10/backends.html#torch.backends.cudnn.benchmark)|-|Yes|Supports fp32|
|[torch.backends.cudnn.benchmark_limit](https://pytorch.org/docs/2.10/backends.html#torch.backends.cudnn.benchmark_limit)|-|No|-|
|[torch.backends.mps.is_available](https://pytorch.org/docs/2.10/backends.html#torch.backends.mps.is_available)|-|Yes|-|
|[torch.backends.mps.is_built](https://pytorch.org/docs/2.10/backends.html#torch.backends.mps.is_built)|-|Yes|-|
|[torch.backends.mkl.is_available](https://pytorch.org/docs/2.10/backends.html#torch.backends.mkl.is_available)|-|Yes|-|
|[torch.backends.mkl.verbose](https://pytorch.org/docs/2.10/backends.html#torch.backends.mkl.verbose)|-|Yes|-|
|[torch.backends.mkldnn.is_available](https://pytorch.org/docs/2.10/backends.html#torch.backends.mkldnn.is_available)|-|<term>Atlas training products</term>: Yes<br><term>Atlas A2 training products</term>: Yes<br><term>Atlas A3 training products</term>: No|-|
|[torch.backends.mkldnn.verbose](https://pytorch.org/docs/2.10/backends.html#torch.backends.mkldnn.verbose)|-|Yes|-|
|[torch.backends.openmp.is_available](https://pytorch.org/docs/2.10/backends.html#torch.backends.openmp.is_available)|-|Yes|Supports fp32|
|[torch.backends.opt_einsum.is_available](https://pytorch.org/docs/2.10/backends.html#torch.backends.opt_einsum.is_available)|-|Yes|Supports fp32|
|[torch.backends.opt_einsum.get_opt_einsum](https://pytorch.org/docs/2.10/backends.html#torch.backends.opt_einsum.get_opt_einsum)|-|Yes|Supports fp32|
|[torch.backends.opt_einsum.enabled](https://pytorch.org/docs/2.10/backends.html#torch.backends.opt_einsum.enabled)|-|No|-|
|[torch.backends.opt_einsum.strategy](https://pytorch.org/docs/2.10/backends.html#torch.backends.opt_einsum.strategy)|-|No|-|
