import warnings

import torch_npu
import torch_npu._C


class NPUFFTPlanCache:
    def __getattr__(self, name):
        if name == "size":
            return torch_npu._C._npu_get_fft_plan_cache_size()
        if name == "max_size":
            return torch_npu._C._npu_get_fft_plan_cache_max_size()
        raise AttributeError("Unknown attribute " + name)
        
    def __setattr__(self, name, value):
        if name == "size":
            raise RuntimeError(".size is a read-only property showing the number of plans currently in the cache.")
        if name == "max_size":
            return torch_npu._C._npu_set_fft_plan_cache_max_size(value)
        raise AttributeError("Unknown attribute " + name)

    def clear(self):
        torch_npu._C._npu_clear_fft_plan_cache()
