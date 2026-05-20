# Owner(s): ["module: inductor"]
# Copyright 2025 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from unittest import skipUnless

import torch
import torch.nn as nn
from torch.func import functionalize
from torch.fx.experimental.proxy_tensor import make_fx
from torch.testing._internal.common_utils import run_tests, TestCase


try:
    from torch_mlir import ir
    from torch_mlir.dialects import func, torch as torch_d
    from torch_npu._inductor.mfusion.fx_mlir_converter import (
        export_mlir_module_to_fx,
        import_mlir_module_from_fx,
    )

    HAS_TORCH_MLIR = True
except ImportError:
    # Gate / minimal CI omits optional torch_mlir; skip FX↔MLIR tests (not errors).
    HAS_TORCH_MLIR = False


class TestModule(nn.Module):
    def forward(self, x):
        # In-place add
        x.add_(1.0)
        return x


class TestImportSmokeModule(nn.Module):
    def forward(self, x):
        x.add_(1.0)
        return x * 2.0


class TestFusedModule(nn.Module):
    def forward(self, x, y):
        x.add_(1.0)
        a = x * y
        b = a + x
        return b


class WrapperModule(torch.nn.Module):
    def __init__(self, gm):
        super().__init__()
        self.gm = gm

    def forward(self, *args):
        res = self.gm(*args)
        # Match eager nn.Module: single output is a Tensor, not a 1-tuple (for allclose etc.).
        if isinstance(res, tuple) and len(res) == 1:
            return res[0]
        return res


if HAS_TORCH_MLIR:

    def fuse_mul_add_pass(module: ir.Module):
        """
        MLIR pass to fuse mul and add operations into a torch.fused.mul_add op.
        This creates a private function for the fused subgraph and replaces the
        original ops with a torch.operator call.
        """
        with module.context, ir.Location.unknown():
            # Scan for pattern: add(mul(a, b), c)
            main_func = module.body.operations[0]
            block = main_func.regions[0].blocks[0]

            mul_op = None
            add_op = None

            for op in block.operations:
                if op.operation.name == "torch.aten.mul.Tensor":
                    mul_op = op
                elif op.operation.name == "torch.aten.add.Tensor" and mul_op:
                    if op.operands[0].owner == mul_op.operation:
                        add_op = op
                        break

            if not (mul_op and add_op):
                return

            # Build subgraph function
            sub_name = "fused_mul_add_impl"

            input_types = [
                mul_op.operands[0].type,
                mul_op.operands[1].type,
                add_op.operands[1].type,
            ]
            output_types = [add_op.results[0].type]

            with ir.InsertionPoint.at_block_begin(module.body):
                sub_func = func.FuncOp(
                    sub_name, (input_types, output_types), visibility="private"
                )
                entry_block = sub_func.add_entry_block()

                with ir.InsertionPoint(entry_block):
                    arg0, arg1, arg2 = entry_block.arguments

                    new_mul = torch_d.AtenMulTensorOp(arg0.type, arg0, arg1)
                    int_type = ir.IntegerType.get_signless(64)
                    one = ir.IntegerAttr.get(int_type, 1)
                    alpha = torch_d.ConstantIntOp(one)

                    new_add = torch_d.AtenAddTensorOp(
                        output_types[0], new_mul.result, arg2, alpha.result
                    )
                    func.ReturnOp([new_add.result])

            # Replace with torch.operator
            with ir.InsertionPoint(add_op):
                fused_op = torch_d.OperatorOp(
                    [add_op.results[0].type],
                    ir.StringAttr.get("torch.fused.mul_add"),
                    [mul_op.operands[0], mul_op.operands[1], add_op.operands[1]],
                    0,
                )
                fused_op.operation.attributes["subgraph"] = ir.FlatSymbolRefAttr.get(
                    sub_name
                )
                add_op.results[0].replace_all_uses_with(fused_op.result)

                add_op.operation.erase()
                mul_op.operation.erase()

    def _run_roundtrip_pass(g: torch.fx.Graph) -> None:
        """
        A custom compilation pass that round-trips the FX graph through MLIR.
        It exports the graph to MLIR and then imports it back to FX, replacing the original graph.
        It also handles metadata restoration (FakeTensorProp) which is crucial for Inductor.
        """
        gm = g.owning_module
        print("FX Graph Before Import:")
        gm.print_readable()

        # Extract inputs for metadata restoration
        example_inputs = []
        for node in g.nodes:
            if node.op == "placeholder":
                example_inputs.append(node.meta.get("val"))

        mlir_module = import_mlir_module_from_fx(gm)
        print("MLIR Module:")
        print(mlir_module)

        new_gm = export_mlir_module_to_fx(mlir_module)
        print("FX Graph After Export:")
        new_gm.print_readable()

        # Restore metadata using FakeTensorProp
        if example_inputs and all(x is not None for x in example_inputs):
            from torch.fx.passes.fake_tensor_prop import FakeTensorProp

            mode = None
            for x in example_inputs:
                if hasattr(x, "fake_mode"):
                    mode = x.fake_mode
                    break

            if mode:
                with mode:
                    FakeTensorProp(new_gm).propagate(*example_inputs)
            else:
                FakeTensorProp(new_gm).propagate(*example_inputs)

        gm.graph = new_gm.graph


@skipUnless(
    HAS_TORCH_MLIR,
    "torch_mlir is not installed (optional dependency; gate CI omits it)",
)
class TestFxRoundtrip(TestCase):
    def test_import_mlir_module_from_fx_smoke(self):
        model = TestImportSmokeModule().eval()
        example = torch.randn(4, 4)

        # Align with inductor preprocessing: convert in-place ops to functional FX.
        gm = make_fx(functionalize(model))(example.clone())
        mlir_module = import_mlir_module_from_fx(gm)

        self.assertIsNotNone(mlir_module)
        mlir_text = str(mlir_module)
        self.assertIn("func.func @main", mlir_text)
        self.assertIn("torch.aten.mul", mlir_text)

    def test_fused_op_roundtrip(self):
        def fused_backend(gm: torch.fx.GraphModule, example_inputs):
            # Clone inputs to avoid mutating them during tracing
            cloned_inputs = [
                t.clone() if isinstance(t, torch.Tensor) else t for t in example_inputs
            ]
            gm = make_fx(functionalize(gm))(*cloned_inputs)

            mlir_module = import_mlir_module_from_fx(gm)
            print("MLIR Before Fusion:")
            print(mlir_module)

            # Apply fusion pass
            fuse_mul_add_pass(mlir_module)

            print("MLIR After Fusion:")
            print(mlir_module)

            new_gm = export_mlir_module_to_fx(mlir_module)
            print("FX After Roundtrip:")
            new_gm.print_readable()
            return WrapperModule(new_gm)

        model = TestFusedModule()
        compiled_model = torch.compile(model, backend=fused_backend)

        x = torch.randn(3, 3)
        y = torch.randn(3, 3)

        x_input = x.clone()
        res = compiled_model(x_input, y)
        expected_res = model(x, y)

        self.assertTrue(torch.allclose(res, expected_res))
        self.assertTrue(torch.allclose(x_input, x))

    def test_inductor_with_roundtrip_pass(self):
        import torch._inductor.config as inductor_config

        original_custom_pre_pass = inductor_config.post_grad_custom_pre_pass
        inductor_config.post_grad_custom_pre_pass = _run_roundtrip_pass

        try:
            model = TestModule()
            x = torch.randn(4, 4)
            compiled_model = torch.compile(model, backend="inductor")

            x_input = x.clone()
            res = compiled_model(x_input)
            expected_res = model(x)

            self.assertTrue(torch.allclose(res, expected_res))
            self.assertTrue(torch.allclose(x_input, expected_res))
        finally:
            inductor_config.post_grad_custom_pre_pass = original_custom_pre_pass


if __name__ == "__main__":
    run_tests()
