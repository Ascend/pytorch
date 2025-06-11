# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
import torch
import triton
import triton.language as tl
import torch_npu
import torch_npu._inductor


# 实际就是 layernorm的计算过程 ： torch.nn.LayerNorm(convert_element_type_25, elementwise_affine=False, eps=1e-6)
class Test_issue62():
    def op_func(self, addmm_5, add):
        split = torch.ops.aten.split.Tensor(addmm_5, 1536, 1)
        getitem = split[0]
        getitem_1 = split[1]
        getitem_2 = split[2]
        getitem_3 = split[3]
        getitem_4 = split[4]
        getitem_5 = split[5]

        clone_1 = torch.ops.aten.clone.default(add, memory_format=torch.contiguous_format)
        convert_element_type_25 = torch.ops.prims.convert_element_type.default(clone_1, torch.float32)
        var_mean = torch.ops.aten.var_mean.correction(convert_element_type_25, [2], correction=0, keepdim=True)
        getitem_6 = var_mean[0]
        getitem_7 = var_mean[1]
        add_3 = torch.ops.aten.add.Tensor(getitem_6, 1e-06)
        rsqrt = torch.ops.aten.rsqrt.default(add_3)
        sub = torch.ops.aten.sub.Tensor(clone_1, getitem_7)
        mul_7 = torch.ops.aten.mul.Tensor(sub, rsqrt)
        convert_element_type_26 = torch.ops.prims.convert_element_type.default(mul_7, torch.float16)
        slice_11 = torch.ops.aten.slice.Tensor(getitem_1, 0, 0, 9223372036854775807)
        unsqueeze_2 = torch.ops.aten.unsqueeze.default(slice_11, 1)
        add_4 = torch.ops.aten.add.Tensor(unsqueeze_2, 1)
        mul_8 = torch.ops.aten.mul.Tensor(convert_element_type_26, add_4)
        slice_12 = torch.ops.aten.slice.Tensor(getitem, 0, 0, 9223372036854775807)
        unsqueeze_3 = torch.ops.aten.unsqueeze.default(slice_12, 1)
        add_5 = torch.ops.aten.add.Tensor(mul_8, unsqueeze_3)
        return add_5

    def test_issue62(self):
        test = Test_issue62()
        addmm_5 = torch.randn((2, 9216), device='npu:0', dtype=torch.float16)
        add = torch.randn((2, 4096, 1536), device='npu:0', dtype=torch.float16)

        std_ret = test.op_func(addmm_5, add)
        compiled_func = torch.compile(test.op_func, backend="inductor")
        inductor_ret = compiled_func(addmm_5, add)
        torch.testing.assert_close(std_ret, inductor_ret, atol=1e-2, rtol=1e-2)
        print("valid ok")


if __name__ == "__main__":
    test = Test_issue62()
    test.test_issue62()