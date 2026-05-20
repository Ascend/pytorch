import torch._inductor.codegen.common
from torch._inductor.codegen.common import device_op_overrides_dict
from torch._inductor.codecache import CacheBase
import hashlib
import json
import torch_npu


def register_device_op_overrides_npu():
    if not device_op_overrides_dict:
        from torch._inductor.codegen import cpu_device_op_overrides, mps_device_op_overrides # noqa: F401
        from torch_npu._inductor import npu_device # noqa: F401
    elif "npu" not in device_op_overrides_dict:
        from torch_npu._inductor import npu_device

def patch_cache_base_get_system():
    # patch function CacheBase.get_system with get_system_npu, add logic to support CANN
    @staticmethod
    def get_system():
        try:
            from triton.compiler.compiler import triton_key

            # Use triton_key instead of triton.__version__ as the version
            # is not updated with each code change
            triton_version = triton_key()
        except ModuleNotFoundError:
            triton_version = None

        try:
            system: Dict[str, Any] = {
                "device": {"name": None},
                "version": {
                    "triton": triton_version,
                },
            }
            device_properties = torch_npu.npu.get_device_properties(
                torch_npu.npu.current_device()
            )
            if torch.version.cann is not None:
                system["device"]["name"] = device_properties.name
                system["version"]["cann"] = torch.version.cann
            elif torch.version.cuda is not None:
                system["device"]["name"] = device_properties.name
                system["version"]["cuda"] = torch.version.cuda
            else:
                system["device"]["name"] = device_properties.gcnArchName
                system["version"]["hip"] = torch.version.hip
        except (AssertionError, RuntimeError):
            # If deivce is not installed, none of the above config is relevant.
            system = {}

        system["hash"] = hashlib.sha256(
            json.dumps(system, sort_keys=True).encode("utf-8")
        ).hexdigest()

        return system

    CacheBase.get_system = get_system
