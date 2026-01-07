import os
import torch
from torch import tensor, device
from torch._dynamo.testing import rand_strided


class Repro(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, embedding_list, mm, mm_3, arg54_1, arg59_1):
        cat = torch.ops.aten.cat.default(embedding_list, -1)
        sum_1 = torch.ops.aten.sum.dim_IntList(cat, [-1])
        add_1 = torch.ops.aten.add.Tensor(sum_1, mm)
        add_3 = torch.ops.aten.add.Tensor(add_1, mm_3)
        add_4 = torch.ops.aten.add.Tensor(add_3, arg59_1)
        sigmoid = torch.ops.aten.sigmoid.default(add_4)
        return sigmoid

mod = Repro().npu()
mod = torch.compile(mod, backend="inductor", dynamic=False)

if __name__ == '__main__':
    from torch._dynamo.repro.after_aot import run_repro
    with torch.no_grad():
        arg54_1 = rand_strided((13, 1), (1, 1), device='npu', dtype=torch.float32)
        arg59_1 = rand_strided((1, 1), (1, 1), device='npu', dtype=torch.float32)
        mm = rand_strided((1, 1), (1, 1), device='npu', dtype=torch.float32)
        mm_3 = rand_strided((1, 1), (1, 1), device='npu', dtype=torch.float32)
        embedding_list = []
        for _ in range(26, 28):
            embedding_list.append(rand_strided((1, 1, 1), (1, 1, 1), device='npu', dtype=torch.float32))
        mod(embedding_list, mm, mm_3, arg54_1, arg59_1)
