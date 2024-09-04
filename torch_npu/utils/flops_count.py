from torch_npu._C._flops_count import _FlopCountContext


class FlopsCounter:
    def __init__(self, ):
        self.flop_count_instance = _FlopCountContext.GetInstance()
     
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
