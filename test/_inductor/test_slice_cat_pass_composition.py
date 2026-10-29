import unittest
from unittest.mock import patch

import torch
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx.experimental.proxy_tensor import make_fx

from torch_npu._inductor.fx_passes.ascend_custom_passes import ascend_graph_pass as passes
from torch_npu._inductor.fx_passes.ascend_custom_passes.register_custom_pass import (
    ASCEND_CUSTOME_PASS_REGISTER,
)
from torch_npu._inductor.fx_passes.utils.fx_pass_level import FxPassLevel, PassType


PIPELINE = (
    passes.remove_cast_after_slice_by_add_cat,
    passes.sink_pointwise_cat_pass,
    passes.multi_slice_concat_pass,
)


def mixed_branches(x, bias):
    parts = [torch.ops.aten.slice.Tensor(x, 1, i, i + 2) for i in (0, 3, 6, 9, 12, 15)]
    return torch.cat(
        [p.float() for p in parts[:3]]
        + [p.float().clamp(-1, 1) for p in parts[3:]] + [bias], 1
    )


class TestSliceCatPassComposition(unittest.TestCase):
    def test_registration_order(self):
        names = [fn.__name__ for fns in ASCEND_CUSTOME_PASS_REGISTER[PassType.POST].values()
                 for fn in fns]
        active = [fn for fn in PIPELINE if fn.__name__ in names]
        self.assertIn(PIPELINE[0], active)
        positions = [names.index(fn.__name__) for fn in active]
        self.assertEqual(positions, sorted(positions))
        self.assertTrue(all(fn.ignore_inference_check for fn in PIPELINE))

    def test_mixed_cast_and_clamp_runs(self):
        from torch._inductor import config

        # Match post-grad's cast decomposition while keeping the eager function differentiable.
        decompositions = {
            torch.ops.aten._to_copy.default:
                lambda x, *, dtype: torch.ops.prims.convert_element_type.default(x, dtype),
        }
        x = torch.randn(4, 40, dtype=torch.bfloat16)[:, 1:39:2]
        x[0, :4] = torch.tensor([float('nan'), float('inf'), -float('inf'), -0.])
        inputs = (x, torch.randn(4, 1))
        for is_ascend950 in (False, True):
            with self.subTest(is_ascend950=is_ascend950), \
                    patch.object(passes, 'is_ascend950', is_ascend950), \
                    config.patch(max_pointwise_cat_inputs=8):
                with FakeTensorMode():
                    gm = make_fx(mixed_branches, decomposition_table=decompositions)(
                        torch.empty(4, 20, device='npu', dtype=torch.bfloat16),
                        torch.empty(4, 1, device='npu'),
                    )
                expected = gm(*inputs)
                for fn in PIPELINE:
                    before = str(gm.graph)
                    fn(gm.graph)
                    gm.graph.lint()
                    expected_change = fn is not passes.multi_slice_concat_pass or is_ascend950
                    self.assertEqual(str(gm.graph) != before, expected_change, fn.__name__)

                def count(target):
                    return sum(n.target is target for n in gm.graph.nodes)

                # Cast packing and sinking each leave a complete three-input Cat.
                # Non-950 keeps these native; 950 uses the indexed route instead.
                self.assertEqual(count(passes.INDEXED_SLICE_CONCAT_TARGET), 2 if is_ascend950 else 0)
                self.assertEqual(count(passes.MULTI_SLICE_CONCAT_TARGET), 0)
                self.assertEqual(count(torch.ops.aten.cat.default), 1 if is_ascend950 else 3)
                self.assertEqual(count(torch.ops.prims.convert_element_type.default), 2)
                self.assertEqual(count(torch.ops.aten.clamp.default), 1)
                gm.recompile()
                actual = gm(*inputs)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0, equal_nan=True)
                zeros = expected == 0
                self.assertTrue(torch.equal(torch.signbit(actual[zeros]), torch.signbit(expected[zeros])))
                before = str(gm.graph)
                for fn in PIPELINE:
                    fn(gm.graph)
                gm.graph.lint()
                self.assertEqual(str(gm.graph), before)

    def test_intermediate_rounding_is_preserved(self):
        def fn(x):
            return torch.cat([x[:, i:i + 2].bfloat16().abs().float()
                              for i in (0, 3, 6)], 1)

        with FakeTensorMode():
            gm = make_fx(fn)(torch.empty(4, 20, device='npu'))
        before = str(gm.graph)
        for rewrite in PIPELINE:
            rewrite(gm.graph)
        gm.graph.lint()
        self.assertEqual(str(gm.graph), before)


@unittest.skipUnless(hasattr(torch, 'npu') and torch.npu.is_available(), 'requires NPU')
class TestSliceCatPassCompositionDevice(unittest.TestCase):
    def test_registered_pipeline_forward_and_backward(self):
        from contextlib import ExitStack
        from functools import wraps
        from torch._inductor import config

        for training in (False, True):
            torch._dynamo.reset()
            called, changed = set(), set()

            def observe(fn):
                @wraps(fn)
                def wrapper(graph):
                    called.add(fn.__name__)
                    before = str(graph)
                    fn(graph)
                    if before != str(graph):
                        changed.add(fn.__name__)
                return wrapper

            with self.subTest(training=training), ExitStack() as stack:
                # The optional passes are absent when the feature flag is off at
                # import time. Exercise the enabled pipeline without depending on
                # the runner's environment, while preserving other registered passes.
                levels = ASCEND_CUSTOME_PASS_REGISTER[PassType.POST]
                enabled = {level: [fn for fn in fns if fn not in PIPELINE]
                           for level, fns in levels.items()}
                original = levels[FxPassLevel.LEVEL1]
                anchor = original.index(PIPELINE[0]) if PIPELINE[0] in original else len(original)
                position = sum(fn not in PIPELINE for fn in original[:anchor])
                enabled[FxPassLevel.LEVEL1][position:position] = [observe(fn) for fn in PIPELINE]
                stack.enter_context(patch.dict(levels, enabled))
                stack.enter_context(config.patch(fx_graph_cache=False, fx_graph_remote_cache=False))
                x = torch.randn(4, 40, dtype=torch.bfloat16, device='npu')[:, 1:39:2]
                x = x.detach().requires_grad_(training)
                bias = torch.randn(4, 1, device='npu', requires_grad=training)
                expected = mixed_branches(x, bias)
                actual = torch.compile(mixed_branches, fullgraph=True)(x, bias)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                self.assertEqual(called, {fn.__name__ for fn in PIPELINE})
                self.assertIn(PIPELINE[0].__name__, changed)
                # Earlier compiler passes can already combine pointwise branches,
                # and small Cats can retain native lowering. The graph test above
                # checks each rewrite; this test checks registration and execution.
                # Backward may also share intermediates that must stay in place.
                if training:
                    grad = torch.ones_like(actual)
                    reference = torch.autograd.grad(expected, (x, bias), grad)
                    result = torch.autograd.grad(actual, (x, bias), grad)
                    torch.testing.assert_close(result, reference, rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
