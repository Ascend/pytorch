import os
from math import inf
import torch
from torch import tensor, device
from torch._dynamo.testing import rand_strided
import torch_npu


def run_case1():
    class Repro(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()

        def forward(self, view_473, amax_20):
            sub_25 = torch.ops.aten.sub.Tensor(view_473, amax_20)
            exp_20 = torch.ops.aten.exp.default(sub_25)
            sum_21 = torch.ops.aten.sum.dim_IntList(exp_20, [1], True)
            log_2 = torch.ops.aten.log.default(sum_21)
            sub_26 = torch.ops.aten.sub.Tensor(sub_25, log_2)
            mul_100 = torch.ops.aten.mul.Tensor(sub_26, view_473)
            sum_22 = torch.ops.aten.sum.default(mul_100)
            return sum_22
            
    
    mod = Repro()
    compile_mod = torch.compile(mod, backend="inductor", dynamic=False)

    with torch.no_grad():
        view_473 = rand_strided((1, 2048, 32128), (65798144, 32128, 1), device='npu', dtype=torch.float32)
        amax_20 = rand_strided((1, 1, 32128), (32128, 32128, 1), device='npu', dtype=torch.float32)
        eager_res = mod(view_473, amax_20)
        compile_res = compile_mod(view_473, amax_20)
        torch.testing.assert_close(compile_res, eager_res)


def run_case2():
    class Repro(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()

        def forward(self, arg4_1, view_3, permute):
            addmm = torch.ops.aten.addmm.default(arg4_1, view_3, permute)
            view_4 = torch.ops.aten.view.default(addmm, [128, 300, 384])
            view_5 = torch.ops.aten.view.default(view_4, [128, 300, 3, 128])
            unsqueeze = torch.ops.aten.unsqueeze.default(view_5, 0)
            permute_1 = torch.ops.aten.permute.default(unsqueeze, [3, 1, 2, 0, 4])
            squeeze = torch.ops.aten.squeeze.dim(permute_1, -2)
            clone = torch.ops.aten.clone.default(squeeze, memory_format=torch.contiguous_format)
            select = torch.ops.aten.select.int(clone, 0, 0)
            view_6 = torch.ops.aten.view.default(select, [128, 2400, 16])
            permute_2 = torch.ops.aten.permute.default(view_6, [1, 0, 2])
            mul_1 = torch.ops.aten.mul.Tensor(permute_2, 0.25)
            unsqueeze_1 = torch.ops.aten.unsqueeze.default(mul_1, 0)
            return (unsqueeze_1,)
            
    mod = Repro()

    with torch.no_grad():
        arg4_1 = rand_strided((384, ), (1, ), device='npu', dtype=torch.float32)
        view_3 = rand_strided((38400, 128), (128, 1), device='npu', dtype=torch.float32)
        permute = rand_strided((128, 384), (1, 128), device='npu', dtype=torch.float32)
        compile_mod = torch.compile(mod, backend="inductor", dynamic=False)
        compile_res = compile_mod(arg4_1, view_3, permute)
        eager_res = mod(arg4_1, view_3, permute)
        torch.testing.assert_close(compile_res, eager_res)


def run_case3():
    
    B, N, S, D = (1, 12, 4096, 8)
    
    def foo_with_permute_reshape(a, b, c):
        y = a + b
        y = y.permute(2, 0, 1, 3)
        y = y.reshape(S, B, N * D)
        y = c + y
        return y

    input1, input2 = [torch.randn(1, 12, 4096, 8, requires_grad=False, dtype=torch.float32, device="npu") for _ in range(2)]
    input3 = torch.randn(4096, 1, 96, requires_grad=False, dtype=torch.float32, device="npu")

    c_foo_with_permute_reshape = torch.compile(foo_with_permute_reshape, backend="inductor", dynamic=False)

    compile_r = c_foo_with_permute_reshape(input1, input2, input3)
    eager_r = foo_with_permute_reshape(input1, input2, input3)
    torch.testing.assert_close(compile_r, eager_r)


def main():
    run_case1()
    run_case2()
    run_case3()

if __name__ == '__main__':
    main()