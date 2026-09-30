from typing import List, Dict
from typing import Optional
from torch._inductor.remote_cache import JsonDataTy
from torch._inductor.runtime.triton_compat import Config


# overload this to avoid autotune after best_config already generated
def _load_cached_autotuning(
        best_config: Dict[str, JsonDataTy],
        configs_hash: str,
        configs: List[Config],
        inductor_meta: Dict,
) -> Optional[Config]:
    if best_config is None:
        return None
    if best_config.pop("configs_hash", None) != configs_hash:
        return None
    # Consume cache metadata in place, matching the upstream cache reader.
    best_config.pop("time_taken_ms", None)
    best_config.pop("triton_cache_hash", None)
    best_config.pop("found_by_coordesc", None)
    extra_options = best_config.pop("extra_options", None)

    # Keep Config attributes (including warp-specialization fields) aligned
    # with the PyTorch cache writer rather than placing them in kwargs.
    from torch._inductor.runtime.autotune_cache import _reconstruct_triton_config
    triton_config = _reconstruct_triton_config(best_config, extra_options)
    # Preserve PTA's existing policy: a cached winner skips further coordesc.
    triton_config.found_by_coordesc = True
    return triton_config


def patch_load_cached_autotuning():
    from torch._inductor.runtime import autotune_cache
    autotune_cache._load_cached_autotuning = _load_cached_autotuning
