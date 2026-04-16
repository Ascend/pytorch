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
    # Remove time taken for comparison
    best_config.pop("time_taken_ms", None)

    num_warps = best_config.pop("num_warps")
    num_stages = best_config.pop("num_stages")
    triton_config = Config(best_config, num_warps=num_warps, num_stages=num_stages)
    triton_config.found_by_coordesc = True
    return triton_config


def patch_load_cached_autotuning():
    from torch._inductor.runtime import autotune_cache
    autotune_cache._load_cached_autotuning = _load_cached_autotuning