import unittest
from unittest.mock import patch

import torch
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx.experimental.proxy_tensor import make_fx

from torch_npu._inductor import config as npu_config
from torch_npu._inductor.fx_passes.ascend_custom_passes import ascend_graph_pass as passes
from torch_npu._inductor.fx_passes.ascend_custom_passes.register_custom_pass import (
    ASCEND_CUSTOME_PASS_REGISTER,
)
from torch_npu._inductor.fx_passes.utils.fx_pass_level import PassType


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
        # DEFAULT_SHUT_DOWN_PASSES gates sink and MSC together.
        # Cast packing is independent and remains enabled by default.
        expected = PIPELINE if npu_config.enable_multi_slice_concat else PIPELINE[:1]
        for fn in PIPELINE:
            self.assertEqual(names.count(fn.__name__), int(fn in expected), fn.__name__)
        positions = [names.index(fn.__name__) for fn in expected]
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
        # Fix MSC's native Cat routing policy; sinking does not use this limit.
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
@unittest.skipUnless(npu_config.enable_multi_slice_concat,
                     'requires TORCHINDUCTOR_ENABLE_MULTI_SLICE_CONCAT=1 before import')
class TestSliceCatPassCompositionDevice(unittest.TestCase):
    def test_registered_pipeline_forward_and_backward(self):
        from contextlib import ExitStack
        from functools import wraps
        from torch._inductor import config
        from torch._inductor.lowering import lowerings
        from torch._inductor.utils import run_and_get_code
        from torch_npu.utils._dynamo import _InductorNpuRegistry

        _InductorNpuRegistry.register_inductor_npu()
        for target in (passes.INDEXED_SLICE_CONCAT_TARGET, passes.MULTI_SLICE_CONCAT_TARGET):
            self.assertIn(target, lowerings, f'{target} lowering must be registered')

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
                # Observe the real registration; never add missing passes or reorder them.
                levels = ASCEND_CUSTOME_PASS_REGISTER[PassType.POST]
                registered = [fn for fns in levels.values() for fn in fns]
                for fn in PIPELINE:
                    self.assertEqual(registered.count(fn), 1, f'{fn.__name__} must be registered once')
                positions = [registered.index(fn) for fn in PIPELINE]
                self.assertEqual(positions, sorted(positions))
                observed = {level: [observe(fn) if fn in PIPELINE else fn for fn in fns]
                            for level, fns in levels.items()}
                stack.enter_context(patch.dict(levels, observed))
                stack.enter_context(config.patch(fx_graph_cache=False, fx_graph_remote_cache=False))
                x = torch.randn(4, 40, dtype=torch.bfloat16, device='npu')[:, 1:39:2]
                x = x.detach().requires_grad_(training)
                bias = torch.randn(4, 1, device='npu', requires_grad=training)
                expected = mixed_branches(x, bias)
                actual, codes = run_and_get_code(torch.compile(mixed_branches, fullgraph=True), x, bias)
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
                    result, backward_codes = run_and_get_code(
                        lambda: torch.autograd.grad(actual, (x, bias), grad)
                    )
                    codes.extend(backward_codes)
                    torch.testing.assert_close(result, reference, rtol=0, atol=0)
                self.assertTrue(codes, 'compiled output code must be captured')
                for code in codes:
                    for name in ('indexed_slice_concat', 'multi_slice_concat'):
                        self.assertNotIn(f'= torch.ops.npu_ext.{name}.default(', code,
                                         'external custom-op fallback does not validate fusion')


if __name__ == '__main__':
    unittest.main()
