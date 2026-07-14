import torch
from torch._dynamo.testing import rand_strided
from torch.testing._internal.common_utils import run_tests
from testutils import TestUtils
import torch_npu


class TestReduction(TestUtils):
    def forward(self, add: "f32[1, 2, 2304]", primals_2: "f32[32, 2304]", primals_5: "f32[1, 9600, 2304]"):
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
        return [None, primals_5, getitem_3, rsqrt, add_2, view, primals_2]

    def test_reduction_cases_shapes(self):
        device = 'npu'
        primals_2: "f32[32, 2304]" = torch.randn((32, 2304), device=device, dtype=torch.float32)
        primals_5: "f32[1, 9600, 2304]" = torch.randn((1, 9600, 2304), device=device, dtype=torch.float32)
        add: "f32[1, 2, 2304]" = torch.randn((1, 2, 2304), device=device, dtype=torch.float32)

        _, primals_5_ref, getitem_3_ref, rsqrt_ref, add_2_ref, view_ref, primals_2_ref = self.forward(add, primals_2, primals_5)

        self.forward = torch.compile(self.forward, backend="inductor", dynamic=False)
        _, primals_5, getitem_3, rsqrt, add_2, view, primals_2 = self.forward(add, primals_2, primals_5)

        self.assertEqual(primals_5_ref, primals_5, atol=1e-3, rtol=1e-3, equal_nan=True)
        self.assertEqual(getitem_3_ref, getitem_3, atol=1e-3, rtol=1e-3, equal_nan=True)
        self.assertEqual(rsqrt_ref, rsqrt, atol=1e-3, rtol=1e-3, equal_nan=True)
        self.assertEqual(add_2_ref, add_2, atol=1e-3, rtol=1e-3, equal_nan=True)
        self.assertEqual(view_ref, view, atol=1e-3, rtol=1e-3, equal_nan=True)
        self.assertEqual(primals_2_ref, primals_2, atol=1e-3, rtol=1e-3, equal_nan=True)

    def test_view_layernorm_bmm(self):
        def view_layernorm_bmm(arg0_1, arg1_1, arg2_1, arg3_1):
            view: "f32[572, 64, 64, 16]" = torch.ops.aten.view.default(arg0_1, [572, 64, 64, 16])
            permute: "f32[572, 64, 64, 16]" = torch.ops.aten.permute.default(view, [0, 2, 1, 3]);  view = None
            clone: "f32[572, 64, 64, 16]" = torch.ops.aten.clone.default(permute, memory_format = torch.contiguous_format);  permute = None
            view_1: "f32[572, 64, 1024]" = torch.ops.aten.view.default(clone, [572, 64, 1024]);  clone = None
            mean: "f32[572, 64, 1]" = torch.ops.aten.mean.dim(view_1, [-1], True)
            var: "f32[572, 64, 1]" = torch.ops.aten.var.correction(view_1, [-1], correction = 0, keepdim = True)
            sub: "f32[572, 64, 1024]" = torch.ops.aten.sub.Tensor(view_1, mean);  view_1 = mean = None
            add: "f32[572, 64, 1]" = torch.ops.aten.add.Tensor(var, 1e-06);  var = None
            sqrt: "f32[572, 64, 1]" = torch.ops.aten.sqrt.default(add);  add = None
            div: "f32[572, 64, 1024]" = torch.ops.aten.div.Tensor(sub, sqrt);  sub = sqrt = None
            mul: "f32[572, 64, 1024]" = torch.ops.aten.mul.Tensor(arg1_1, div);  arg1_1 = div = None
            add_1: "f32[572, 64, 1024]" = torch.ops.aten.add.Tensor(mul, arg2_1);  mul = arg2_1 = None
            permute_1: "f32[64, 572, 1024]" = torch.ops.aten.permute.default(add_1, [1, 0, 2])
            expand: "f32[64, 572, 1024]" = torch.ops.aten.expand.default(permute_1, [64, 572, 1024]);  permute_1 = None
            expand_1: "f32[64, 1024, 3]" = torch.ops.aten.expand.default(arg3_1, [64, 1024, 3]);  arg3_1 = None
            bmm: "f32[64, 572, 3]" = torch.ops.aten.bmm.default(expand, expand_1);  expand = expand_1 = None
            return bmm

        arg0_1 = rand_strided((572, 64, 1024), (65536, 1024, 1), device='npu:0', dtype=torch.float32)
        arg1_1 = rand_strided((1024, ), (1, ), device='npu:0', dtype=torch.float32)
        arg2_1 = rand_strided((1024, ), (1, ), device='npu:0', dtype=torch.float32)
        arg3_1 = rand_strided((64, 1024, 3), (3072, 3, 1), device='npu:0', dtype=torch.float32)
        compile_fn = torch.compile(view_layernorm_bmm, fullgraph=True, backend="inductor", dynamic=False)
        inductor_out = compile_fn(arg0_1, arg1_1, arg2_1, arg3_1)
        eager_out = view_layernorm_bmm(arg0_1, arg1_1, arg2_1, arg3_1)
        torch.testing.assert_close(inductor_out, eager_out, rtol=1e-3, atol=1e-3)

if __name__ == "__main__":
    run_tests()
