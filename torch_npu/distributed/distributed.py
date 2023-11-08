from torch.nn.parallel import DistributedDataParallel
import torch_npu


class Distributed_DataParallel(DistributedDataParallel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        torch_npu.npu.synchronize()