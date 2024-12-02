import warnings
import torch_npu

__all__ = []


class _FlopsCounter:
    def __init__(self, ):
        self.flop_count_instance = torch_npu._C._flops_count._FlopCountContext.GetInstance()
     
    def __enter__(self):
        self.count_enable()
    
    def __exit__(self):
        self.count_disable()

    def start(self):
        self.flop_count_instance.enable()

    def stop(self):
        self.flop_count_instance.disable()
        self.flop_count_instance.reset()
    
    def pause(self):
        self.flop_count_instance.pause()

    def resume(self):
        self.flop_count_instance.resume()

    def get_flops(self):
        recorded_count = self.flop_count_instance.recordedCount
        traversed_count = self.flop_count_instance.traversedCount
        return [recorded_count, traversed_count]


class FlopsCounter(_FlopsCounter):
    def __init__(self):
        super().__init__()
        warnings.warn("torch_npu.utils.flops_count.FlopsCounter() will be deprecated. "
                      "If necessary, please use torch_npu.utils.FlopsCounter().", FutureWarning)

