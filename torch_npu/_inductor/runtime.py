import functools
from typing import List, Dict
from typing import Optional
from torch._inductor.remote_cache import JsonDataTy
from torch._inductor.runtime.hints import DeviceProperties
from torch.utils._triton import has_triton, has_triton_package

from .config import num_vector_core

if has_triton_package():
    from triton import Config


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

    # if inductor_meta.get("coordinate_descent_tuning") :
    num_warps = best_config.pop("num_warps")
    num_stages = best_config.pop("num_stages")
    triton_config = Config(best_config, num_warps=num_warps, num_stages=num_stages)
    triton_config.found_by_coordesc = True
    return triton_config


class NPUDeviceProperties(DeviceProperties):

    @classmethod
    @functools.lru_cache(None)
    def create(cls, device) -> DeviceProperties:
        import torch
        from torch._dynamo.device_interface import get_interface_for_device

        device_type = device.type

        if torch.version.hip and device_type == "cuda":
            device_type = "hip"

        device_interface = get_interface_for_device(device)
        props = device_interface.get_device_properties(device)

        try:
            multi_processor_count = num_vector_core
        except AttributeError:
            if device_type == "xpu":
                multi_processor_count = props.gpu_subslice_count
            else:
                raise
        return cls(
            type=device_type,
            index=device.index,
            multi_processor_count=multi_processor_count,
            cc=device_interface.get_compute_capability(device),
            major=getattr(props, "major", None),
            regs_per_multiprocessor=getattr(props, "regs_per_multiprocessor", None),
            max_threads_per_multi_processor=getattr(
                props, "max_threads_per_multi_processor", None
            ),
            warp_size=getattr(props, "warp_size", 32 if device_type != "cpu" else None),
        )
