import os
import random
import numpy as np

import torch
from torch import device
from torch.testing._internal.common_utils import run_tests
from testutils import TestUtils
import torch_npu

device_npu = 'npu'


class TestModel(TestUtils):
    def test_opensora_cases_model_9_inference(self):
        def forward(primals_1: "f32[1, 9600, 2304]"):
            permute: "f32[9600, 1, 2304]" = torch.ops.aten.permute.default(primals_1, [1, 0, 2])
            return permute
        primals_2 = torch.randn((1, 9600, 2304), device=device_npu, dtype=torch.float32)
        ref = forward(primals_2)
        forward_calc = torch.compile(forward, backend="inductor", dynamic=False)
        calc = forward_calc(primals_2)
        self.assertEqual(ref, calc, atol=1e-4, rtol=1e-4, equal_nan=True)
        primals_3 = torch.randn((1, 512, 2304), device=device_npu, dtype=torch.float32)
        forward_calc = torch.compile(forward, backend="inductor", dynamic=False)
        calc = forward_calc(primals_3)
        ref = forward(primals_3)
        self.assertEqual(ref, calc, atol=1e-4, rtol=1e-4, equal_nan=True)
        primals_4 = torch.randn((9600, 1, 2304), device=device_npu, dtype=torch.float32)
        forward_calc = torch.compile(forward, backend="inductor", dynamic=False)
        calc = forward_calc(primals_4)
        ref = forward(primals_4)
        self.assertEqual(ref, calc, atol=1e-4, rtol=1e-4, equal_nan=True)

    def test_opensora_cases_model_11_inference(self):
        def forward(arg0_1: "f32[1, 1, 9600]", arg1_1: "f32[1, 1, 512]"):
            unsqueeze: "f32[1, 1, 1, 9600]" = torch.ops.aten.unsqueeze.default(arg0_1, 1)
            arg0_1 = None
            unsqueeze_1: "f32[1, 1, 1, 512]" = torch.ops.aten.unsqueeze.default(arg1_1, 1)
            arg1_1 = None
            constant_pad_nd: "f32[1, 1, 1, 9600]" = torch.ops.aten.constant_pad_nd.default(unsqueeze, [0, 0, 0, 0], -9980.0)
            unsqueeze = None
            view: "f32[1, 9600, 1]" = torch.ops.aten.view.default(constant_pad_nd, [1, 9600, 1])
            permute: "f32[1, 1, 9600]" = torch.ops.aten.permute.default(view, [2, 0, 1])
            view = None
            view_1: "f32[1, 1, 1, 9600]" = torch.ops.aten.view.default(permute, [1, 1, 1, 9600])
            permute = None
            view_2: "f32[1, 9600, 1, 1]" = torch.ops.aten.view.default(constant_pad_nd, [1, 9600, 1, 1])
            constant_pad_nd = None
            permute_1: "f32[1, 1, 9600, 1]" = torch.ops.aten.permute.default(view_2, [2, 0, 1, 3])
            view_2 = None
            view_3: "f32[1, 1, 1, 9600]" = torch.ops.aten.view.default(permute_1, [1, 1, 1, 9600])
            permute_1 = None
            repeat: "f32[1, 1, 1, 512]" = torch.ops.aten.repeat.default(unsqueeze_1, [1, 1, 1, 1])
            unsqueeze_1 = None
            npu_dtype_cast: "b8[1, 1, 1, 9600]" = torch.ops.npu.npu_dtype_cast.default(view_1, torch.bool)
            view_1 = None
            repeat_1: "b8[1, 1, 9600, 9600]" = torch.ops.aten.repeat.default(npu_dtype_cast, [1, 1, 9600, 1])
            npu_dtype_cast = None
            npu_dtype_cast_1: "b8[1, 1, 1, 9600]" = torch.ops.npu.npu_dtype_cast.default(view_3, torch.bool)
            view_3 = None
            repeat_2: "b8[1, 1, 9600, 9600]" = torch.ops.aten.repeat.default(npu_dtype_cast_1, [1, 1, 9600, 1])
            npu_dtype_cast_1 = None
            npu_dtype_cast_2: "b8[1, 1, 1, 512]" = torch.ops.npu.npu_dtype_cast.default(repeat, torch.bool)
            repeat = None
            repeat_3: "b8[1, 1, 9600, 512]" = torch.ops.aten.repeat.default(npu_dtype_cast_2, [1, 1, 9600, 1])
            npu_dtype_cast_2 = None
            return (repeat_1, repeat_3, repeat_2)
        arg0_1 = torch.rand((1, 1, 9600), device=device_npu, dtype=torch.float32)
        arg1_1 = torch.rand((1, 1, 512), device=device_npu, dtype=torch.float32)
        ref = forward(arg0_1, arg1_1)
        forward_calc = torch.compile(forward, backend="inductor", dynamic=False)
        calc = forward_calc(arg0_1, arg1_1)

        for r, c in zip(ref, calc):
            self.assertEqual(r, c, atol=1e-4, rtol=1e-4, equal_nan=True)


    def test_opensora_cases_model_14_backward(self):
        def forward(args):
            primals_5, getitem_3, rsqrt, add_2, view, permute_1, tangents_1 = args
            sub: "f32[1, 9600, 2304]" = torch.ops.aten.sub.Tensor(primals_5, getitem_3)
            mul: "f32[1, 9600, 2304]" = torch.ops.aten.mul.Tensor(sub, rsqrt)
            view_2: "f32[9600, 32]" = torch.ops.aten.view.default(tangents_1, [9600, 32])
            mm: "f32[9600, 2304]" = torch.ops.aten.mm.default(view_2, permute_1)
            permute_2: "f32[32, 9600]" = torch.ops.aten.permute.default(view_2, [1, 0])
            mm_1: "f32[32, 2304]" = torch.ops.aten.mm.default(permute_2, view)
            permute_3: "f32[2304, 32]" = torch.ops.aten.permute.default(mm_1, [1, 0])
            sum_1: "f32[1, 32]" = torch.ops.aten.sum.dim_IntList(view_2, [0], True)
            view_3: "f32[32]" = torch.ops.aten.view.default(sum_1, [32])
            permute_4: "f32[32, 2304]" = torch.ops.aten.permute.default(permute_3, [1, 0])
            view_4: "f32[1, 9600, 2304]" = torch.ops.aten.view.default(mm, [1, 9600, 2304])
            sum_2: "f32[1, 1, 2304]" = torch.ops.aten.sum.dim_IntList(view_4, [1], True)
            mul_2: "f32[1, 9600, 2304]" = torch.ops.aten.mul.Tensor(view_4, mul)
            mul_3: "f32[1, 9600, 2304]" = torch.ops.aten.mul.Tensor(view_4, add_2)
            sum_3: "f32[1, 1, 2304]" = torch.ops.aten.sum.dim_IntList(mul_2, [1], True)
            mul_5: "f32[1, 9600, 2304]" = torch.ops.aten.mul.Tensor(mul_3, 2304)
            sum_4: "f32[1, 9600, 1]" = torch.ops.aten.sum.dim_IntList(mul_3, [2], True)
            mul_6: "f32[1, 9600, 2304]" = torch.ops.aten.mul.Tensor(mul_3, mul)
            sum_5: "f32[1, 9600, 1]" = torch.ops.aten.sum.dim_IntList(mul_6, [2], True)
            mul_7: "f32[1, 9600, 2304]" = torch.ops.aten.mul.Tensor(mul, sum_5)
            sub_2: "f32[1, 9600, 2304]" = torch.ops.aten.sub.Tensor(mul_5, sum_4)
            sub_3: "f32[1, 9600, 2304]" = torch.ops.aten.sub.Tensor(sub_2, mul_7)
            div: "f32[1, 9600, 1]" = torch.ops.aten.div.Tensor(rsqrt, 2304)
            mul_8: "f32[1, 9600, 2304]" = torch.ops.aten.mul.Tensor(div, sub_3)
            cat: "f32[1, 2, 2304]" = torch.ops.aten.cat.default([sum_2, sum_3], 1)
            sum_6: "f32[1, 1, 2304]" = torch.ops.aten.sum.dim_IntList(cat, [1], True)
            squeeze_1: "f32[1, 2304]" = torch.ops.aten.squeeze.dim(sum_6, 1)
            full_default: "f32[1, 2304]" = torch.ops.aten.full.default([1, 2304], 0, dtype=torch.float32, device='npu')
            slice_scatter: "f32[1, 2304]" = torch.ops.aten.slice_scatter.default(full_default, squeeze_1, 0, 0,
                                                                                9223372036854775807)
            squeeze_2: "f32[2, 2304]" = torch.ops.aten.squeeze.dim(cat, 0)
            return [squeeze_2, permute_4, view_3, slice_scatter, mul_8]
        primals_5 = torch.randn((1, 9600, 2304), device=device_npu, dtype=torch.float32)
        getitem_3 = torch.randn((1, 9600, 1), device=device_npu, dtype=torch.float32)
        rsqrt = torch.randn((1, 9600, 1), device=device_npu, dtype=torch.float32)
        add_2 = torch.randn((1, 1, 2304), device=device_npu, dtype=torch.float32)
        view = torch.randn((9600, 2304), device=device_npu, dtype=torch.float32)
        permute_1 = torch.randn((32, 2304), device=device_npu, dtype=torch.float32)
        tangents_1 = torch.randn((1, 9600, 32), device=device_npu, dtype=torch.float32)
        args = (primals_5, getitem_3, rsqrt, add_2, view, permute_1, tangents_1)
        ref = forward(args)
        forward_calc = torch.compile(forward, backend="inductor", dynamic=False)
        calc = forward_calc(args)

        for r, c in zip(ref, calc):
            self.assertEqual(r, c, atol=1e-3, rtol=1e-3, equal_nan=True)


    def test_opensora_cases_model_14_forward(self):
        def forward(primals_1: "f32[2, 2304]", primals_2: "f32[32, 2304]", primals_3: "f32[32]",
                    primals_4: "f32[1, 2304]", primals_5: "f32[1, 9600, 2304]"):
            unsqueeze: "f32[1, 2, 2304]" = torch.ops.aten.unsqueeze.default(primals_1, 0)
            primals_1 = None
            slice_1: "f32[1, 2304]" = torch.ops.aten.slice.Tensor(primals_4, 0, 0, 9223372036854775807)
            primals_4 = None
            unsqueeze_1: "f32[1, 1, 2304]" = torch.ops.aten.unsqueeze.default(slice_1, 1)
            slice_1 = None
            add: "f32[1, 2, 2304]" = torch.ops.aten.add.Tensor(unsqueeze, unsqueeze_1)
            unsqueeze = unsqueeze_1 = None
            split = torch.ops.aten.split.Tensor(add, 1, 1)
            add = None
            getitem: "f32[1, 1, 2304]" = split[0]
            getitem_1: "f32[1, 1, 2304]" = split[1]
            split = None
            var_mean = torch.ops.aten.var_mean.correction(primals_5, [2], correction=0, keepdim=True)
            getitem_2: "f32[1, 9600, 1]" = var_mean[0]
            getitem_3: "f32[1, 9600, 1]" = var_mean[1]
            var_mean = None
            add_1: "f32[1, 9600, 1]" = torch.ops.aten.add.Tensor(getitem_2, 1e-06)
            getitem_2 = None
            rsqrt: "f32[1, 9600, 1]" = torch.ops.aten.rsqrt.default(add_1)
            add_1 = None
            sub: "f32[1, 9600, 2304]" = torch.ops.aten.sub.Tensor(primals_5, getitem_3)
            mul: "f32[1, 9600, 2304]" = torch.ops.aten.mul.Tensor(sub, rsqrt)
            sub = None
            add_2: "f32[1, 1, 2304]" = torch.ops.aten.add.Tensor(getitem_1, 1)
            getitem_1 = None
            mul_1: "f32[1, 9600, 2304]" = torch.ops.aten.mul.Tensor(mul, add_2)
            mul = None
            add_3: "f32[1, 9600, 2304]" = torch.ops.aten.add.Tensor(mul_1, getitem)
            mul_1 = getitem = None
            view: "f32[9600, 2304]" = torch.ops.aten.view.default(add_3, [9600, 2304])
            add_3 = None
            permute: "f32[2304, 32]" = torch.ops.aten.permute.default(primals_2, [1, 0])
            primals_2 = None
            addmm: "f32[9600, 32]" = torch.ops.aten.addmm.default(primals_3, view, permute)
            primals_3 = None
            view_1: "f32[1, 9600, 32]" = torch.ops.aten.view.default(addmm, [1, 9600, 32])
            addmm = None
            # No stacktrace found for following nodes
            squeeze: "f32[1, 9600, 32]" = torch.ops.aten.squeeze.dim(view_1, 1)
            view_1 = None
            permute_1: "f32[32, 2304]" = torch.ops.aten.permute.default(permute, [1, 0])
            permute = None
            return [squeeze, primals_5, getitem_3, rsqrt, add_2, view, permute_1]
        primals_1 = torch.ones((2, 2304), device=device_npu, dtype=torch.float32)
        primals_2 = torch.ones((32, 2304), device=device_npu, dtype=torch.float32)
        primals_3 = torch.ones((32,), device=device_npu, dtype=torch.float32)
        primals_4 = torch.ones((1, 2304), device=device_npu, dtype=torch.float32)
        primals_5 = torch.ones((1, 9600, 2304), device=device_npu, dtype=torch.float32)
        ref = forward(primals_1, primals_2, primals_3, primals_4, primals_5)
        forward_calc = torch.compile(forward, backend="inductor", dynamic=False)
        calc = forward_calc(primals_1, primals_2, primals_3, primals_4, primals_5)
        for r, c in zip(ref, calc):
            self.assertEqual(r, c, atol=1e-4, rtol=1e-4, equal_nan=True)


    def test_opensora_cases_model_15_forward(self):
        def forward(primals_1: "f32[1, 8, 30, 40, 1, 2, 2, 8]", primals_2: "i64[]", primals_3: "i64[]",
                    primals_4: "i64[]"):
            permute: "f32[1, 8, 8, 1, 30, 2, 40, 2]" = torch.ops.aten.permute.default(primals_1, [0, 7, 1, 4, 2, 5, 3, 6])
            mul: "i64[]" = torch.ops.aten.mul.Tensor(primals_2, 1)
            mul_1: "i64[]" = torch.ops.aten.mul.Tensor(primals_3, 2)
            mul_2: "i64[]" = torch.ops.aten.mul.Tensor(primals_4, 2)
            return [permute, mul, mul_1, mul_2]
    
        primals_1 = torch.randn((1, 8, 30, 40, 1, 2, 2, 8), device=device_npu, dtype=torch.float32)
        primals_2 = torch.tensor((1), device=device_npu, dtype=torch.int64)
        primals_3 = torch.tensor((1), device=device_npu, dtype=torch.int64)
        primals_4 = torch.tensor((1), device=device_npu, dtype=torch.int64)
        ref = forward(primals_1, primals_2, primals_3,
                    primals_4)
        forward_calc = torch.compile(forward, backend="inductor", dynamic=False)
        calc = forward_calc(primals_1, primals_2, primals_3,
                    primals_4)
        for r, c in zip(ref, calc):
            self.assertEqual(r, c, atol=1e-4, rtol=1e-4, equal_nan=True)

    def test_opensora_cases_model_16_forward(self):
        def forward(primals_1: "f32[2, 2304]", primals_2: "f32[32, 2304]", primals_3: "f32[32]", primals_4: "f32[1, 2304]", primals_5: "f32[1, 9600, 2304]"):
            unsqueeze: "f32[1, 2, 2304]" = torch.ops.aten.unsqueeze.default(primals_1, 0)
            slice_1: "f32[1, 2304]" = torch.ops.aten.slice.Tensor(primals_4, 0, 0, 9223372036854775807)
            unsqueeze_1: "f32[1, 1, 2304]" = torch.ops.aten.unsqueeze.default(slice_1, 1)
            add: "f32[1, 2, 2304]" = torch.ops.aten.add.Tensor(unsqueeze, unsqueeze_1)
            split = torch.ops.aten.split.Tensor(add, 1, 1)
            getitem: "f32[1, 1, 2304]" = split[0]
            getitem_1: "f32[1, 1, 2304]" = split[1]
            var_mean = torch.ops.aten.var_mean.correction(primals_5, [2], correction=0, keepdim=True)
            getitem_2: "f32[1, 9600, 1]" = var_mean[0]
            getitem_3: "f32[1, 9600, 1]" = var_mean[1]
            add_1: "f32[1, 9600, 1]" = torch.ops.aten.add.Tensor(getitem_2, 1e-06)
            rsqrt: "f32[1, 9600, 1]" = torch.ops.aten.rsqrt.default(add_1)
            sub: "f32[1, 9600, 2304]" = torch.ops.aten.sub.Tensor(primals_5, getitem_3)
            mul: "f32[1, 9600, 2304]" = torch.ops.aten.mul.Tensor(sub, rsqrt)
            add_2: "f32[1, 1, 2304]" = torch.ops.aten.add.Tensor(getitem_1, 1)
            mul_1: "f32[1, 9600, 2304]" = torch.ops.aten.mul.Tensor(mul, add_2)
            add_3: "f32[1, 9600, 2304]" = torch.ops.aten.add.Tensor(mul_1, getitem)
            view: "f32[9600, 2304]" = torch.ops.aten.view.default(add_3, [9600, 2304])
            permute: "f32[2304, 32]" = torch.ops.aten.permute.default(primals_2, [1, 0])
            addmm: "f32[9600, 32]" = torch.ops.aten.addmm.default(primals_3, view, permute)
            view_1: "f32[1, 9600, 32]" = torch.ops.aten.view.default(addmm, [1, 9600, 32])
            squeeze: "f32[1, 9600, 32]" = torch.ops.aten.squeeze.dim(view_1, 1)
            view_2: "f32[1, 8, 30, 40, 1, 2, 2, 8]" = torch.ops.aten.view.default(squeeze, [1, 8, 30, 40, 1, 2, 2, 8])
            permute_1: "f32[1, 8, 8, 1, 30, 2, 40, 2]" = torch.ops.aten.permute.default(view_2, [0, 7, 1, 4, 2, 5, 3, 6])
            clone: "f32[1, 8, 8, 1, 30, 2, 40, 2]" = torch.ops.aten.clone.default(permute_1)
            clone_1: "f32[1, 8, 8, 1, 30, 2, 40, 2]" = torch.ops.aten.clone.default(clone, memory_format=torch.contiguous_format)
            view_3: "f32[1, 8, 8, 60, 80]" = torch.ops.aten.view.default(clone_1, [1, 8, 8, 60, 80])
            permute_3: "f32[32, 2304]" = torch.ops.aten.permute.default(permute, [1, 0])
            return [view_3, primals_5, getitem_3, rsqrt, add_2, view, permute_3]

        def seed_all(seed=1234, mode=False):
            random.seed(seed)
            os.environ['PYTHONHASHSEED'] = str(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.use_deterministic_algorithms(mode)
            torch_npu.npu.manual_seed_all(seed)
            torch_npu.npu.manual_seed(seed)

        seed_all(True)
        primals_1 = torch.randn((2, 2304), device=device_npu, dtype=torch.float32)
        primals_2 = torch.randn((32, 2304), device=device_npu, dtype=torch.float32)
        primals_3 = torch.randn((32,), device=device_npu, dtype=torch.float32)
        primals_4 = torch.randn((1, 2304), device=device_npu, dtype=torch.float32)
        primals_5 = torch.randn((1, 9600, 2304), device=device_npu, dtype=torch.float32)

        ref = forward(primals_1, primals_2, primals_3, primals_4, primals_5)
        forward_calc = torch.compile(forward, backend="inductor", dynamic=False)
        calc = forward_calc(primals_1, primals_2, primals_3, primals_4, primals_5)
        for r, c in zip(ref, calc):
            self.assertEqual(r, c, atol=1e-3, rtol=1e-3, equal_nan=True)


if __name__ == "__main__":
    run_tests()
