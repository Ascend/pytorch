# torch.backends

> [!NOTE]  
> 若API“是否支持”为“是”，“限制与说明”为“-”，说明此API和原生API支持度保持一致。

|API名称|NPU形式名称|是否支持|限制与说明|
|--|--|--|--|
|[torch.backends.cpu.get_cpu_capability](https://pytorch.org/docs/2.10/backends.html#torch.backends.cpu.get_cpu_capability)|-|是|-|
|[torch.backends.cuda.is_built](https://pytorch.org/docs/2.10/backends.html#torch.backends.cuda.is_built)|-|是|-|
|[torch.backends.cuda.matmul.allow_tf32](https://pytorch.org/docs/2.10/backends.html#torch.backends.cuda.matmul.allow_tf32)|-|是|支持fp32|
|[torch.backends.cuda.cufft_plan_cache](https://pytorch.org/docs/2.10/backends.html#torch.backends.cuda.cufft_plan_cache)|torch.npu.backends.fft_plan_cache|是|-|
|[torch.backends.cuda.cufft_plan_cache.size](https://pytorch.org/docs/2.10/backends.html#torch.backends.cuda.cufft_plan_cache.size)|torch.npu.backends.fft_plan_cache.size|是|-|
|[torch.backends.cuda.cufft_plan_cache.max_size](https://pytorch.org/docs/2.10/backends.html#torch.backends.cuda.cufft_plan_cache.max_size)|torch.npu.backends.fft_plan_cache.max_size|是|输入范围为1-99|
|[torch.backends.cuda.cufft_plan_cache.clear](https://pytorch.org/docs/2.10/backends.html#torch.backends.cuda.cufft_plan_cache.clear)|torch.npu.backends.fft_plan_cache.clear|是|-|
|[torch.backends.cuda.preferred_linalg_library](https://pytorch.org/docs/2.10/backends.html#torch.backends.cuda.preferred_linalg_library)|-|否|-|
|[torch.backends.cuda.flash_sdp_enabled](https://pytorch.org/docs/2.10/backends.html#torch.backends.cuda.flash_sdp_enabled)|-|否|-|
|[torch.backends.cuda.enable_mem_efficient_sdp](https://pytorch.org/docs/2.10/backends.html#torch.backends.cuda.enable_mem_efficient_sdp)|-|否|-|
|[torch.backends.cuda.mem_efficient_sdp_enabled](https://pytorch.org/docs/2.10/backends.html#torch.backends.cuda.mem_efficient_sdp_enabled)|-|否|-|
|[torch.backends.cuda.enable_flash_sdp](https://pytorch.org/docs/2.10/backends.html#torch.backends.cuda.enable_flash_sdp)|-|否|-|
|[torch.backends.cuda.math_sdp_enabled](https://pytorch.org/docs/2.10/backends.html#torch.backends.cuda.math_sdp_enabled)|-|否|-|
|[torch.backends.cuda.enable_math_sdp](https://pytorch.org/docs/2.10/backends.html#torch.backends.cuda.enable_math_sdp)|-|否|-|
|[torch.backends.cuda.sdp_kernel](https://pytorch.org/docs/2.10/backends.html#torch.backends.cuda.sdp_kernel)|-|否|-|
|[torch.backends.cudnn.is_available](https://pytorch.org/docs/2.10/backends.html#torch.backends.cudnn.is_available)|-|是|支持fp32|
|[torch.backends.cudnn.enabled](https://pytorch.org/docs/2.10/backends.html#torch.backends.cudnn.enabled)|-|是|支持fp32|
|[torch.backends.cudnn.allow_tf32](https://pytorch.org/docs/2.10/backends.html#torch.backends.cudnn.allow_tf32)|-|是|支持fp32|
|[torch.backends.cudnn.deterministic](https://pytorch.org/docs/2.10/backends.html#torch.backends.cudnn.deterministic)|-|是|-|
|[torch.backends.cudnn.benchmark](https://pytorch.org/docs/2.10/backends.html#torch.backends.cudnn.benchmark)|-|是|支持fp32|
|[torch.backends.cudnn.benchmark_limit](https://pytorch.org/docs/2.10/backends.html#torch.backends.cudnn.benchmark_limit)|-|否|-|
|[torch.backends.mps.is_available](https://pytorch.org/docs/2.10/backends.html#torch.backends.mps.is_available)|-|是|-|
|[torch.backends.mps.is_built](https://pytorch.org/docs/2.10/backends.html#torch.backends.mps.is_built)|-|是|-|
|[torch.backends.mkl.is_available](https://pytorch.org/docs/2.10/backends.html#torch.backends.mkl.is_available)|-|是|-|
|[torch.backends.mkl.verbose](https://pytorch.org/docs/2.10/backends.html#torch.backends.mkl.verbose)|-|是|-|
|[torch.backends.mkldnn.is_available](https://pytorch.org/docs/2.10/backends.html#torch.backends.mkldnn.is_available)|-|<term>Atlas 训练系列产品</term>：是<br><term>Atlas A2 训练系列产品</term>：是<br><term>Atlas A3 训练系列产品</term>：否|-|
|[torch.backends.mkldnn.verbose](https://pytorch.org/docs/2.10/backends.html#torch.backends.mkldnn.verbose)|-|是|-|
|[torch.backends.openmp.is_available](https://pytorch.org/docs/2.10/backends.html#torch.backends.openmp.is_available)|-|是|支持fp32|
|[torch.backends.opt_einsum.is_available](https://pytorch.org/docs/2.10/backends.html#torch.backends.opt_einsum.is_available)|-|是|支持fp32|
|[torch.backends.opt_einsum.get_opt_einsum](https://pytorch.org/docs/2.10/backends.html#torch.backends.opt_einsum.get_opt_einsum)|-|是|支持fp32|
|[torch.backends.opt_einsum.enabled](https://pytorch.org/docs/2.10/backends.html#torch.backends.opt_einsum.enabled)|-|否|-|
|[torch.backends.opt_einsum.strategy](https://pytorch.org/docs/2.10/backends.html#torch.backends.opt_einsum.strategy)|-|否|-|
