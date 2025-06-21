# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2023-2023. All rights reserved.
import torch.nn.functional as F
import torch
import torch_npu


def op_sum(npu_dropout_backward_9):
    view_337: "f32[32768, 256]" = torch.ops.aten.view.default(npu_dropout_backward_9, [32768, 256]);
    sum_63: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_337, [0], True);
    view_338: "f32[256]" = torch.ops.aten.view.default(sum_63, [256]);
    return view_338

device = 'npu'


def test_high_order_sum():
    npu_dropout_backward_9 = torch.randn((32768, 256), device=device, dtype=torch.float32)
    ref = op_sum(npu_dropout_backward_9)
    func = torch.compile(op_sum, backend="inductor", dynamic=False)
    calc = func(npu_dropout_backward_9)

    torch.testing.assert_close(ref, calc, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(ref, calc, rtol=1e-3, atol=1e-3)

if __name__ == "__main__":
    npu_dropout_backward_9 = torch.randn((32768, 256), device=device, dtype=torch.float32)
    ref = op_sum(npu_dropout_backward_9)
    func = torch.compile(op_sum, backend="inductor", dynamic=False)
    calc = func(npu_dropout_backward_9)

    torch.testing.assert_close(ref, calc, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(ref, calc, rtol=1e-3, atol=1e-3)

