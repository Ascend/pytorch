import torch
from torch import device
import torch_npu


class Repro(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self):
        iota = torch.ops.prims.iota.default(50, start=0, step=1, dtype=torch.int64, device=device(type='cpu'), requires_grad=False)
        unsqueeze_23 = torch.ops.aten.unsqueeze.default(iota, 0)
        expand = torch.ops.aten.expand.default(unsqueeze_23, [16, -1])
        device_put = torch.ops.prims.device_put.default(expand, device(type='npu', index=0))
        convert_element_type = torch.ops.prims.convert_element_type.default(device_put, torch.int64)
        return convert_element_type

mod = Repro()
mod = torch.compile(mod, backend="inductor", dynamic=False)

if __name__ == '__main__':
    with torch.no_grad():
        mod()