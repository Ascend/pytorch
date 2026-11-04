# Copyright (c) 2026 Huawei Technologies Co., Ltd
# Owner(s): ["module: inductor"]

import sympy
import torch
import torch.nn.functional as F
from torch._inductor.utils import run_and_get_code
from torch.testing._internal.common_utils import (
    TestCase,
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
)

import torch_npu  # noqa: F401
from torch_npu._inductor.triton_experimental.reduction_index_alias import (
    ReductionIndexAliasError,
    ReductionIndexAssignmentState,
    ReductionIndexAssignmentLine,
    attach_reduction_index_aliases,
    build_reduction_index_aliases,
    make_reduction_loop_pass,
    render_reduction_index_aliases,
    superseded_reduction_index_names,
    validate_reduction_index_alias_scope,
)


class TestReductionIndexAliasIR(TestCase):
    def test_aliases_are_attached_to_each_reduction_loop_pass(self):
        r0_1, r0_2, r0_3 = sympy.symbols("r0_1 r0_2 r0_3")
        aliases = build_reduction_index_aliases(
            {"r0_4": r0_3 + 1, "r0_3": r0_1 + 16 * r0_2}
        )
        validate_reduction_index_alias_scope(aliases, {r0_1, r0_2})

        passes = attach_reduction_index_aliases(
            tuple(make_reduction_loop_pass(i, ("r0_",)) for i in range(3)),
            "r0_",
            aliases,
        )

        self.assertEqual([p.scope_id for p in passes], [0, 1, 2])
        self.assertEqual(
            [
                [alias.name for alias in p.reduction_index_aliases]
                for p in passes
            ],
            [["r0_3", "r0_4"]] * 3,
        )
        self.assertEqual(
            superseded_reduction_index_names(passes[0], {"r0_1", "r0_2"}),
            {"r0_1", "r0_2", "r0_3", "r0_4"},
        )
        self.assertEqual(
            render_reduction_index_aliases(passes[0], str, "    "),
            [
                "    r0_3 = r0_1 + 16*r0_2",
                "    r0_4 = r0_3 + 1",
            ],
        )

    def test_alias_scope_rejects_undefined_symbols(self):
        aliases = build_reduction_index_aliases(
            {"r0_3": sympy.Symbol("r0_1") + sympy.Symbol("unknown")}
        )
        with self.assertRaisesRegex(ReductionIndexAliasError, "undefined symbols"):
            validate_reduction_index_alias_scope(aliases, {"r0_1"})

    def test_alias_cycles_are_rejected(self):
        with self.assertRaisesRegex(
            ReductionIndexAliasError, "cyclic reduction index aliases"
        ):
            build_reduction_index_aliases(
                {
                    "r0_3": sympy.Symbol("r0_4"),
                    "r0_4": sympy.Symbol("r0_3"),
                }
            )

    def test_original_index_assignment_is_suppressed_by_identity(self):
        state = ReductionIndexAssignmentState()
        line = ReductionIndexAssignmentLine(
            state, "r0_", "r0_3", "r0_3 = r0_index"
        )
        self.assertEqual(line(), "r0_3 = r0_index")
        state.suppress_assignments("r0_", {"r0_3"})
        self.assertIsNone(line())


@instantiate_parametrized_tests
class TestReductionIndexAliasCodegen(TestCase):
    def setUp(self):
        super().setUp()
        torch._dynamo.reset()

    def tearDown(self):
        torch._dynamo.reset()
        super().tearDown()

    @staticmethod
    def _group_norm_with_affine_epilogue(
        x, bias, weight, shift, scale, offset
    ):
        x = x + bias.view(1, -1, 1, 1)
        x = F.group_norm(x, 8, weight, shift, 1e-5)
        x = x * (1 + scale.view(1, -1, 1, 1))
        return F.silu(x + offset.view(1, -1, 1, 1))

    @parametrize("spatial", [32, 128])
    def test_group_norm_alias_is_visible_in_every_reduction_loop_pass(self, spatial):
        torch.manual_seed(437)
        inputs = (torch.randn(1, 128, spatial, spatial, device="npu"),) + tuple(
            torch.randn(128, device="npu") * 0.25 for _ in range(5)
        )
        compiled = torch.compile(
            self._group_norm_with_affine_epilogue,
            backend="inductor",
            fullgraph=True,
            dynamic=False,
            options={"npu_backend": "triton_experimental"},
        )

        with torch.no_grad():
            expected = self._group_norm_with_affine_epilogue(*inputs)
            actual, codes = run_and_get_code(compiled, *inputs)

        self.assertEqual(actual, expected, rtol=1e-3, atol=1e-4)
        source = "\n".join(codes)
        self.assertIn("@triton.jit", source)
        self.assertNotIn("torch.ops.aten.native_group_norm.default(", source)


if __name__ == "__main__":
    run_tests()
