# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
import torch
import torch.nn.functional as F
from testutils import benchmark_test
import triton
import triton.language as tl
import torch_npu
import torch_npu._inductor


class Test_issue57():
    def op_sum(self, view_12, embedding_1, slice_11):
        # 原网络

        permute_7 = torch.ops.aten.permute.default(embedding_1, [2, 0, 1]);
        embedding_1 = None
        unsqueeze_4 = torch.ops.aten.unsqueeze.default(permute_7, 0);
        permute_7 = None

        add_5 = torch.ops.aten.add.Tensor(unsqueeze_4, slice_11);
        slice_8 = slice_11 = None
        add_6 = torch.ops.aten.add.Tensor(view_12, add_5);
        view_12 = None
        return add_6

    def test_issue57(self):
        device = 'npu'
        test = Test_issue57()
        embedding_1 = torch.randn((512, 512, 64), device=device, dtype=torch.float32)
        primals_221 = torch.randn((1, 1, 1, 512), device=device, dtype=torch.float32)
        view_12 = torch.randn((1, 64, 512, 512), device=device, dtype=torch.float32)
        slice_11 = torch.randn((1, 1, 1, 512), device=device, dtype=torch.float32)

        ref = test.op_sum(view_12, embedding_1, primals_221)
        func = torch.compile(test.op_sum, backend="inductor", dynamic=False)
        calc = func(view_12, embedding_1, primals_221)

        torch.testing.assert_close(ref, calc, rtol=1e-3, atol=1e-3)

        print("valid ok")
        benchmark_test(test.op_sum, func, args=(view_12, embedding_1, primals_221),
                       name="issue57", times=10, repeat=10, profile=False)


@triton.jit
def triton_unk_fused_add_0(in_ptr0, in_ptr1, in_ptr2, out_ptr0, y0_numel, x2_numel, x1_numel, 
                           X2BLOCK: tl.constexpr, Y0BLOCK_SUB: tl.constexpr, X2BLOCK_SUB: tl.constexpr, X1BLOCK_SUB: tl.constexpr):
    base_y0 = tl.arange(0, Y0BLOCK_SUB)
    loops_y0 = (y0_numel + Y0BLOCK_SUB - 1) // Y0BLOCK_SUB
    x2_offset = tl.program_id(0) * X2BLOCK
    base_x2 = tl.arange(0, X2BLOCK_SUB)
    loops_x2 = (X2BLOCK + X2BLOCK_SUB - 1) // X2BLOCK_SUB
    base_x1 = tl.arange(0, X1BLOCK_SUB)
    loops_x1 = (x1_numel + X1BLOCK_SUB - 1) // X1BLOCK_SUB
    for loop_y0 in range(loops_y0):
        y0_2 = (loop_y0 * Y0BLOCK_SUB) + base_y0[None, None, :]
        y0 = (loop_y0 * Y0BLOCK_SUB) + base_y0[:, None, None]
        y0_mask = y0 < y0_numel
        y0_2_mask = y0_2 < y0_numel
        for loop_x2 in range(loops_x2):
            x2_0 = x2_offset + (loop_x2 * X2BLOCK_SUB) + base_x2[:, None, None]
            x2 = x2_offset + (loop_x2 * X2BLOCK_SUB) + base_x2[None, :, None]
            x2_mask = x2 < x2_numel
            x2_0_mask = x2_0 < x2_numel
            for loop_x1 in range(loops_x1):
                x1_1 = (loop_x1 * X1BLOCK_SUB) + base_x1[None, :, None]
                x1 = (loop_x1 * X1BLOCK_SUB) + base_x1[None, None, :]
                x1_mask = x1 < x1_numel
                x1_1_mask = x1_1 < x1_numel
                tmp0 = tl.load(in_ptr0 + (x1 + 512 * x2 + 262144 * y0), x1_mask & x2_mask & y0_mask)
                tmp1 = tl.load(in_ptr1 + (y0_2 + 64 * x1_1 + 32768 * x2_0), x1_1_mask & x2_0_mask & y0_2_mask)
                tmp2 = tmp1.permute([2, 0, 1])
                tmp3 = tl.load(in_ptr2 + (x1), x1_mask)
                tmp4 = tmp3.reshape([1, 1, X1BLOCK_SUB]).broadcast_to([Y0BLOCK_SUB, X2BLOCK_SUB, X1BLOCK_SUB])
                tmp5 = tmp2 + tmp4
                tmp6 = tmp0 + tmp5
                tl.store(out_ptr0 + (x1 + 512 * x2 + 262144 * y0), tmp6, x1_mask & x2_mask & y0_mask)


def triton_var_mean_(view_12, embedding_1, slice_11):
    y0_numel, x2_numel, x1_numel = [64, 512, 512]
 
    # The block size of each loop iteration is the smallest power of two greater than the number of columns in `x`
    num_programs = 16
    X2BLOCK, Y0BLOCK_SUB, X2BLOCK_SUB, X1BLOCK_SUB = 16, 16, 1, 16
    #X2BLOCK': 16, 'Y0BLOCK_SUB': 16, 'X2BLOCK_SUB': 1, 'X1BLOCK_SUB': 16
 
    out = torch.empty((y0_numel, x2_numel, x1_numel), device='npu', dtype=torch.float32)
    
    kernel = triton_unk_fused_add_0.warmup(view_12, embedding_1, slice_11, out,
                                           y0_numel, x2_numel, x1_numel,
                                           X2BLOCK=X2BLOCK, Y0BLOCK_SUB=Y0BLOCK_SUB, 
                                           X2BLOCK_SUB=X2BLOCK_SUB, X1BLOCK_SUB=X1BLOCK_SUB,
                                           grid=(num_programs,))
    kernel._init_handles()
    device = torch.npu.current_device()
    stream = torch.npu.current_stream(device).npu_stream
    # Create a number of persistent programs.
    kernel[(num_programs, 1, 1)](view_12, embedding_1, slice_11, out, y0_numel, x2_numel, x1_numel, stream=stream)
    return out

                
if __name__ == "__main__":
    test = Test_issue57()
    test.test_issue57()