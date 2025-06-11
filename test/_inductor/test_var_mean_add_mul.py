import torch
import torch_npu
import torch_npu._inductor


def test_reduction_cases_shapes():
    device = 'npu'

    def forward(add: "f32[1, 2, 2304]", primals_2: "f32[32, 2304]", primals_5: "f32[1, 9600, 2304]"):
        split = torch.ops.aten.split.Tensor(add, 1, 1);
        getitem: "f32[1, 1, 2304]" = split[0]
        getitem_1: "f32[1, 1, 2304]" = split[1];

        var_mean = torch.ops.aten.var_mean.correction(primals_5, [2], correction=0, keepdim=True)
        getitem_2: "f32[1, 9600, 1]" = var_mean[0]
        getitem_3: "f32[1, 9600, 1]" = var_mean[1];
        add_1: "f32[1, 9600, 1]" = torch.ops.aten.add.Tensor(getitem_2, 1e-06);
        rsqrt: "f32[1, 9600, 1]" = torch.ops.aten.rsqrt.default(add_1);
        sub: "f32[1, 9600, 2304]" = torch.ops.aten.sub.Tensor(primals_5, getitem_3)
        mul: "f32[1, 9600, 2304]" = torch.ops.aten.mul.Tensor(sub, rsqrt);

        add_2: "f32[1, 1, 2304]" = torch.ops.aten.add.Tensor(getitem_1, 1);
        mul_1: "f32[1, 9600, 2304]" = torch.ops.aten.mul.Tensor(mul, add_2);
        add_3: "f32[1, 9600, 2304]" = torch.ops.aten.add.Tensor(mul_1, getitem);

        view: "f32[9600, 2304]" = torch.ops.aten.view.default(add_3, [9600, 2304]);
        return [None, primals_5, getitem_3, rsqrt, add_2, view, primals_2]

    torch_npu._inductor.config.enable_npu_indexing = True
    primals_2: "f32[32, 2304]" = torch.randn((32, 2304), device=device, dtype=torch.float32)
    primals_5: "f32[1, 9600, 2304]" = torch.randn((1, 9600, 2304), device=device, dtype=torch.float32)
    add: "f32[1, 2, 2304]" = torch.randn((1, 2, 2304), device=device, dtype=torch.float32)

    _, primals_5_ref, getitem_3_ref, rsqrt_ref, add_2_ref, view_ref, primals_2_ref = forward(add, primals_2, primals_5)

    forward = torch.compile(forward, backend="inductor", dynamic=False)
    _, primals_5, getitem_3, rsqrt, add_2, view, primals_2 = forward(add, primals_2, primals_5)

    torch.testing.assert_close(primals_5_ref, primals_5, equal_nan=True, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(getitem_3_ref, getitem_3, equal_nan=True, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(rsqrt_ref, rsqrt, equal_nan=True, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(add_2_ref, add_2, equal_nan=True, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(primals_2_ref, primals_2, equal_nan=True, rtol=1e-3, atol=1e-3)

test_reduction_cases_shapes()
