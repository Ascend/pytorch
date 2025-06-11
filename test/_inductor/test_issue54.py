# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
import torch
from torch import nn
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
from testutils import benchmark_test
import torch_npu
import torch_npu._inductor


class Test_issue54():
    def func_layernorm(self, add_3, primals_6, primals_7, view, primals_9, permute_1, primals_10, primals_11):
        # 原网络
        permute: "f32[256, 256]" = torch.ops.aten.permute.default(primals_6, [1, 0]);
        addmm: "f32[32768, 256]" = torch.ops.aten.addmm.default(primals_7, view, permute);
        view_1: "f32[64, 512, 256]" = torch.ops.aten.view.default(addmm, [64, 512, 256]);
        addmm_1: "f32[32768, 256]" = torch.ops.aten.addmm.default(primals_9, view, permute_1);
        view_3: "f32[64, 512, 256]" = torch.ops.aten.view.default(addmm_1, [64, 512, 256]);
        view_4: "f32[64, 512, 4, 64]" = torch.ops.aten.view.default(view_3, [64, 512, 4, 64]);
        permute_2: "f32[64, 4, 512, 64]" = torch.ops.aten.permute.default(view_4, [0, 2, 1, 3]);
        permute_3: "f32[256, 256]" = torch.ops.aten.permute.default(primals_10, [1, 0]);
        addmm_2: "f32[32768, 256]" = torch.ops.aten.addmm.default(primals_11, view, permute_3);
        view_6: "f32[64, 512, 256]" = torch.ops.aten.view.default(addmm_2, [64, 512, 256]);

        view_8: "f32[64, 512, 4, 64]" = torch.ops.aten.view.default(view_1, [64, 512, 4, 64]);
        permute_5: "f32[64, 4, 512, 64]" = torch.ops.aten.permute.default(view_8, [0, 2, 1, 3]);

        permute_6: "f32[64, 4, 64, 512]" = torch.ops.aten.permute.default(permute_2, [0, 1, 3, 2]);
        expand_1: "f32[64, 4, 512, 64]" = torch.ops.aten.expand.default(permute_5, [64, 4, 512, 64])
        clone: "f32[64, 4, 512, 64]" = torch.ops.aten.clone.default(expand_1, memory_format=torch.contiguous_format);
        view_9: "f32[256, 512, 64]" = torch.ops.aten.view.default(clone, [256, 512, 64]);
        expand_2: "f32[64, 4, 64, 512]" = torch.ops.aten.expand.default(permute_6, [64, 4, 64, 512])
        clone_1: "f32[64, 4, 64, 512]" = torch.ops.aten.clone.default(expand_2, memory_format=torch.contiguous_format);
        view_10: "f32[256, 64, 512]" = torch.ops.aten.view.default(clone_1, [256, 64, 512]);
        bmm: "f32[256, 512, 512]" = torch.ops.aten.bmm.default(view_9, view_10);
        view_7: "f32[64, 512, 4, 64]" = torch.ops.aten.view.default(view_6, [64, 512, 4, 64]);
        permute_4: "f32[64, 4, 512, 64]" = torch.ops.aten.permute.default(view_7, [0, 2, 1, 3]);
        expand_4: "f32[64, 4, 512, 64]" = torch.ops.aten.expand.default(permute_4, [64, 4, 512, 64])
        clone_2: "f32[64, 4, 512, 64]" = torch.ops.aten.clone.default(expand_4, memory_format=torch.contiguous_format);
        view_13: "f32[256, 512, 64]" = torch.ops.aten.view.default(clone_2, [256, 512, 64]);

        return bmm, view_13

    def test_issue54(self):
        device = 'npu'
        test = Test_issue54()
        # add_3, primals_6, primals_7, view, primals_9, permute_1, primals_10, primals_11

        add_3 = torch.randn((64, 512, 256), device=device, dtype=torch.float32)
        primals_6 = torch.randn((256, 256), device=device, dtype=torch.float32)
        primals_7 = torch.randn((256), device=device, dtype=torch.float32)
        view = torch.randn((32768, 256), device=device, dtype=torch.float32)
        primals_9 = torch.randn((256), device=device, dtype=torch.float32)
        permute_1 = torch.randn((256, 256), device=device, dtype=torch.float32)
        primals_10 = torch.randn((256, 256), device=device, dtype=torch.float32)
        primals_11 = torch.randn((256), device=device, dtype=torch.float32)

        ref = test.func_layernorm(add_3, primals_6, primals_7, view, primals_9, permute_1, primals_10, primals_11)
        func = torch.compile(test.func_layernorm, backend="inductor", dynamic=False,
                             options={"unroll_reductions_threshold": 1, "aggressive_fusion": True})
        calc = func(add_3, primals_6, primals_7, view, primals_9, permute_1, primals_10, primals_11)
        torch.testing.assert_close(ref[0], calc[0], rtol=1e-2, atol=1e-2)
        torch.testing.assert_close(ref[1], calc[1], rtol=1e-2, atol=1e-2)
        print("valid ok")

        benchmark_test(test.func_layernorm, func,
                       args=(add_3, primals_6, primals_7, view, primals_9, permute_1, primals_10, primals_11,),
                       name="test_layernorm", times=10, repeat=10, profile=False)


if __name__ == "__main__":
    test = Test_issue54()
    test.test_issue54()