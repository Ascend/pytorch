# Copyright (c) 2026, Huawei Technologies Co., Ltd
#
"""NPU GEMM template heuristics + benchmarker dispatch for Inductor autotune.

Importing this module performs two registrations (invoked once from
``__init__._activate``; re-import is a no-op via sys.modules):

1. ``triton::mm`` / ``triton::bmm`` template heuristics for device_type "npu".
   Without a device-specific registration,
   ``heuristics.registry.get_template_heuristic`` falls back to the empty base
   heuristic (logging "No template heuristic found ... device_type=npu"), so
   ``tuned_mm``/``tuned_bmm`` only ever produce the ATen extern choice.
   ``select_algorithm`` then has a single candidate and never benchmarks,
   which leaves ``counters["inductor"]["select_algorithm_autotune"]`` at 0
   (fails test_max_autotune) and silently skips GEMM autotuning on NPU.

   The upstream ``triton_mm.py.jinja`` / ``triton_bmm.py.jinja`` templates are
   plain ``tl.dot`` kernels with no CUDA-only deps, and triton-ascend compiles
   and runs them correctly, so we register an NPU config heuristic mirroring
   the XPU registration pattern (heuristics/template/triton.py::XPUMM...).

   Gating: ``MMTemplateConfigMixin`` -> ``GemmMaxAutotuneTemplateConfigHeuristics``
   -> ``should_run`` only yields configs when ``max_autotune`` or
   ``max_autotune_gemm`` is enabled, so default-mode compiles keep the
   extern-only behavior; nothing changes unless autotuning was requested.

2. An "npu" entry in the benchmarker dispatch registry
   (``torch._inductor.runtime.benchmarking.register_benchmarker`` -- the
   public out-of-tree hook). Without it, ``Benchmarker.benchmark(fn,
   device="npu")`` (ExternKernelGPUBenchmarkRequest.do_bench) falls through to
   ``benchmark_gpu`` with ``device_type=None``; the None default normalizes to
   "cuda" on NPU-only builds and dies with "Torch not compiled with CUDA
   enabled" while benchmarking the ATen extern choice under max_autotune.
   The shim mirrors the built-in ``_default_cuda_bench`` / ``_default_xpu_bench``.

3. An ``addmm``/``baddbmm`` (op_name-scoped) registration of an
   ``AddMMConfigMixin`` subclass, mirroring the XPU precedent
   (``XPUAddmmTemplateConfigHeuristic``). The addmm lookup passes
   op_name="addmm" to the registry, so the plain mm/bmm registration above
   is invisible to it: the fallback base heuristic's identity input
   adjustment drops the bias epilogue (prefix_args / epilogue_fn) and trips
   def_kernel's assert (2, 3, 0, 3).
"""

try:
    from torch._inductor.heuristics.registry import register_template_heuristic
    from torch._inductor.heuristics.template.triton import (
        AddMMConfigMixin,
        BaseConfigHeuristic,
        GemmConfig,
        MMTemplateConfigMixin,
    )
except ModuleNotFoundError:
    # master maintains torch 2.13+, where AddMMConfigMixin is re-exported by
    # template_heuristics.triton (from triton_addmm) and the registry accepts
    # op_name.
    from torch._inductor.template_heuristics.registry import register_template_heuristic
    from torch._inductor.template_heuristics.triton import (
        AddMMConfigMixin,
        BaseConfigHeuristic,
        GemmConfig,
        MMTemplateConfigMixin,
    )
from torch._inductor.kernel.bmm import bmm_template
from torch._inductor.kernel.mm import mm_template
from torch._inductor.runtime.benchmarking import register_benchmarker


class NPUConfigHeuristic(BaseConfigHeuristic):
    """BaseConfigHeuristic with an Ascend-friendly GEMM config pool.

    The stock (CUDA-oriented) mm_configs go up to 128x256x64 tiles with 4-5
    pipeline stages; on Ascend the A/B stage buffers multiply against the
    ~192KB UB budget, so the larger tiles only waste autotune compile time.
    Keep a small conservative pool: tile area <= 128x128, BLOCK_K <= 64,
    <= 2 stages, 4 warps. ATen (aclnnMatmul on the cube unit) wins virtually
    every NPU GEMM today (~0.3ms vs ~56ms for a 2048^3 fp32 mm), so these
    choices exist to give autotuning real triton candidates to benchmark and
    a place to tune once triton-ascend GEMM codegen improves.
    """

    def __init__(self) -> None:
        super().__init__()
        self.mm_configs = [
            GemmConfig(32, 32, 32, 1, 4),
            GemmConfig(64, 64, 32, 2, 4),
            GemmConfig(64, 64, 64, 2, 4),
            GemmConfig(128, 64, 32, 2, 4),
            GemmConfig(64, 128, 32, 2, 4),
            GemmConfig(128, 128, 32, 2, 4),
        ]
        # TODO(npux): design an NPU-specific exhaustive search space; reuse the
        # default pool until then.
        self.exhaustive_configs = self.mm_configs


@register_template_heuristic(mm_template.uid, "npu")
@register_template_heuristic(bmm_template.uid, "npu")
class NPUMMTemplateConfigHeuristic(MMTemplateConfigMixin, NPUConfigHeuristic):
    """Standard MM/BMM template heuristic for NPU."""


@register_template_heuristic(mm_template.uid, "npu", op_name="addmm")
@register_template_heuristic(bmm_template.uid, "npu", op_name="baddbmm")
class NPUAddMMTemplateConfigHeuristic(AddMMConfigMixin, NPUMMTemplateConfigHeuristic):
    """Addmm/baddbmm specific mixin for NPU (mirrors CUDAAddMMTemplateConfigHeuristic)."""


def _default_npu_bench(self, f, *, warmup, rep, **kw):
    # Mirrors _default_cuda_bench/_default_xpu_bench: route
    # Benchmarker.benchmark(fn, device="npu") to benchmark_gpu with an explicit
    # device_type instead of the None -> "cuda" fallback.
    kw.setdefault("device_type", "npu")
    return self.benchmark_gpu(f, warmup=warmup, rep=rep, **kw)


register_benchmarker("npu", _default_npu_bench, override=True)
