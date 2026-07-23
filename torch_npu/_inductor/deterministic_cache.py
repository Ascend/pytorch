import os


def _is_ascendc_backend() -> bool:
    return os.getenv("TORCHINDUCTOR_NPU_BACKEND") == "ascendc"


def patch_npu_deterministic_level_cache_keys():
    """Add the exact NPU deterministic level to AscendC Inductor cache keys."""
    import torch_npu
    from torch._functorch._aot_autograd.autograd_cache import AOTAutogradCacheDetails
    from torch._inductor.codecache import FxGraphHashDetails

    if getattr(FxGraphHashDetails, "_npu_deterministic_level_patched", False):
        return

    fx_graph_hash_details_init = FxGraphHashDetails.__init__
    aot_autograd_cache_details_init = AOTAutogradCacheDetails.__init__

    def fx_graph_hash_details_init_with_npu_deterministic_level(self, *args, **kwargs):
        fx_graph_hash_details_init(self, *args, **kwargs)
        if _is_ascendc_backend():
            self.npu_deterministic_level = torch_npu.npu._get_deterministic_level()

    def aot_autograd_cache_details_init_with_npu_deterministic_level(self, *args, **kwargs):
        aot_autograd_cache_details_init(self, *args, **kwargs)
        if _is_ascendc_backend():
            self.npu_deterministic_level = torch_npu.npu._get_deterministic_level()

    FxGraphHashDetails.__init__ = fx_graph_hash_details_init_with_npu_deterministic_level
    AOTAutogradCacheDetails.__init__ = aot_autograd_cache_details_init_with_npu_deterministic_level
    FxGraphHashDetails._npu_deterministic_level_patched = True
