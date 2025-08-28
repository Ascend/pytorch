import os
import torch
from torch._inductor.pattern_matcher import register_graph_pattern, CallFunction, Arg, PatternMatcherPass, Match

from torch.library import custom_op, triton_op, wrap_triton
from torch.testing._internal.common_utils import run_tests, parametrize, instantiate_parametrized_tests

import triton
import triton.language as tl

from testutils import TestUtils

import torch_npu
import torch_npu._inductor


@triton.jit
def triton_cross_add(in_ptr0, in_ptr1, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    tmp0 = tl.load(in_ptr0 + offsets, mask)
    tmp1 = tl.load(in_ptr1 + offsets, mask)
    tmp2 = tl.sin(tmp0)
    tmp3 = tl.cos(tmp2)
    tmp4 = tl.cos(tmp1)
    tmp5 = tl.sin(tmp4)
    tmp6 = tl.add(tmp3, tmp5)
    tl.store(out_ptr + offsets, tmp6, mask)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 4}),
        triton.Config({"BLOCK_SIZE": 8}),
        triton.Config({"BLOCK_SIZE": 16}),
    ],
    key=["n_elements"],
)
@triton.jit
def triton_fused_add_one(in_ptr0, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    tmp0 = tl.load(in_ptr0 + offsets, mask)
    tmp1 = tmp0 + 1
    tl.store(out_ptr + offsets, tmp1, mask)


@triton.jit
def triton_fused_add_sin(in_ptr0, in_ptr1, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    tmp0 = tl.load(in_ptr0 + offsets, mask)
    tmp1 = tl.load(in_ptr1 + offsets, mask)
    tmp2 = tl.sin(tmp1)
    tmp3 = tl.add(tmp0, tmp2)
    tl.store(out_ptr + offsets, tmp3, mask)


@register_graph_pattern(
    CallFunction(torch.add, Arg(), CallFunction(torch.sin, Arg())),
    pass_dict=PatternMatcherPass(pass_name="test"),
)
def add_sin_replacement(match: Match, x, y):
    z = torch.zeros_like(x)
    n_element = x.numel()
    BLOCK_SIZE = 16
    grid = (triton.cdiv(n_element, BLOCK_SIZE),)
    triton_fused_add_sin[grid](x, y, z, n_element, BLOCK_SIZE=BLOCK_SIZE)
    return z


@custom_op("my_custom::cpu_add", mutates_args={})
def cpu_add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x_cpu = x.cpu()
    y_cpu = y.cpu()
    z_cpu = x_cpu + y_cpu
    return z_cpu.to("npu")


@cpu_add.register_fake
def _(x, y):
    return torch.zeros_like(x)


@triton.jit
def triton_fused_activation_min_max(in_ptr0, in_ptr1, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    tmp0 = tl.load(in_ptr0 + offsets, mask)
    tmp1 = tl.load(in_ptr1 + offsets, mask)
    tmp2 = tl.sigmoid(tmp0)
    tmp3 = tl.softmax(tmp1)
    tmp4 = tl.min(tmp2, axis=0)
    tmp5 = tl.max(tmp3, axis=0)
    tmp6 = tmp4 + tmp5
    tl.store(out_ptr + offsets, tmp6, mask)


@triton_op("my_triton::activation_min_max", mutates_args={})
def activation_min_max(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    n_element = x.numel()
    BLOCK_SIZE = 16
    grid = (triton.cdiv(n_element, BLOCK_SIZE),)
    z = torch.zeros([1, x.shape[1]], dtype=x.dtype, device=x.device)
    wrap_triton(triton_fused_activation_min_max)[grid](x, y, z, n_element, BLOCK_SIZE=BLOCK_SIZE)
    return z


class Model(torch.nn.Module):
    def __init__(self, dim=32):
        super().__init__()
        self.fc1 = torch.nn.Linear(dim, dim, dtype=torch.float16)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x, y, test_aoti=False):
        # test fused kernel with weights
        x = self.fc1(x + 1)
        x = self.sigmoid(x)
        y = torch.abs(y)

        # test user defined triton kernel
        z1 = torch.zeros_like(x)
        n_element = x.numel()
        BLOCK_SIZE = 8
        grid = (triton.cdiv(n_element, BLOCK_SIZE),)
        triton_cross_add[grid](x, y, z1, n_element, BLOCK_SIZE=BLOCK_SIZE)

        # test user defined triton kernel with autotune
        z2 = torch.zeros_like(x)
        triton_fused_add_one[grid](x, z2, n_element)

        # test register_graph_pattern
        z3 = torch.cos(x) + torch.sin(y)

        # test user defined custom_op, AOTInductor not support custom_op
        if test_aoti:
            z4 = torch.ones_like(x)
        else:
            z4 = torch.ops.my_custom.cpu_add(x, y)

        # test user defined triton_op, output shape is [1, dim]
        z5 = torch.ops.my_triton.activation_min_max(x, y)

        # test op_plugin ascendC kernel
        z6, z7 = torch_npu.npu_rms_norm(x, y)

        # sum of all result, auto broadcast z4
        z8 = z1 + z2 + z3 + z4 + z5 + z6

        return z8


class TestAotiUserDefinedOp(TestUtils):
    def generate_input_tensor(self, batch_size=8, dim=32, device="npu"):
        x_input = torch.arange(0, batch_size * dim, 1, device=device).reshape([batch_size, dim])
        x_input = 1.0 / x_input.to(torch.float16)
        y_input = torch.arange(batch_size * dim, 0, -1, device=device).reshape([batch_size, dim])
        y_input = 1.0 / y_input.to(torch.float16)
        return x_input, y_input


    @parametrize('shape_x', [8])
    @parametrize('shape_y', [32])
    def test_compile(self, shape_x, shape_y):
        with torch.no_grad():
            model = Model().to("npu")
            x_input, y_input = self.generate_input_tensor(shape_x, shape_y)
            eager_res = model.forward(x_input, y_input)

            model_c = torch.compile(model, backend="inductor", dynamic=False)
            compile_res = model_c(x_input, y_input, False)
            self.assertEqual(eager_res, compile_res, atol=1e-3, rtol=1e-3)


    @parametrize('shape_x', [8])
    @parametrize('shape_y', [32])
    @parametrize('autotune_at_compile', [True, False])
    @parametrize('static_mode', [True, False])
    def test_aoti_export(self, shape_x, shape_y, autotune_at_compile, static_mode):
        with torch.no_grad():
            model = Model().to("npu")
            torch._inductor.config.triton.autotune_at_compile_time = autotune_at_compile
            torch_npu._inductor.config.inductor_static_mode = static_mode
            x_input, y_input = self.generate_input_tensor(shape_x, shape_y)

            exported = torch.export.export(model, (x_input, y_input, True))
            model_name = f"model_{os.getpid()}_{shape_x}_{shape_y}_{int(autotune_at_compile)}_{int(static_mode)}.pt2"
            output_path = torch._inductor.aoti_compile_and_package(
                exported,
                package_path=os.path.join(os.getcwd(), model_name),
            )
            self.assertTrue(
                os.path.exists(output_path),
                f"could not find target {output_path} generated by test_aoti_export",
            )

instantiate_parametrized_tests(TestAotiUserDefinedOp)

if __name__ == "__main__":
    run_tests()
