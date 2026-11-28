#!/usr/bin/env python
# -*- coding: utf-8 -*-
#  Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
import inspect
import os
import unittest
from types import SimpleNamespace
from unittest import mock

# This case targets the Triton (default) sub-backend only: the torch_npu
# make_reduction override replaces the upstream symbol only when that backend
# is loaded.
os.environ.setdefault("TORCHINDUCTOR_NPU_BACKEND", "default")

import torch  # noqa: E402
from torch.testing._internal.common_utils import (  # noqa: E402
    TestCase,
    run_tests,
    parametrize,
    instantiate_parametrized_tests,
)
from testutils import TestUtils  # noqa: E402
import torch_npu  # noqa: E402,F401


def _loaded_backend():
    """Return the sub-backend name loaded by torch_npu._inductor, or None."""
    try:
        from torch_npu.utils._dynamo import _InductorNpuRegistry

        return getattr(_InductorNpuRegistry, "_loaded_backend", None)
    except (ImportError, AttributeError):
        # The registry moved or is unavailable: let the case run, a real
        # backend mismatch still surfaces as a compile failure.
        return None


_BACKEND = _loaded_backend()


@unittest.skipUnless(
    _BACKEND in (None, "default"),
    "make_reduction override only applies to the default (Triton) sub-backend, "
    "but %s is loaded" % _BACKEND,
)
class TestReductionCompile(TestUtils):
    """Regression tests for compiling dim reductions on the default Inductor backend.

    torch_npu replaces torch._inductor.lowering.make_reduction with its own
    implementation.  Upstream sum_ passes strict_sum on Torch 2.14 and
    strict_reduction on newer Torch.  The compat layer selects the matching signature and forwards the flag
    using the installed Reduction.create API.

    These cases cover dim reductions; full reductions need separate
    end-to-end regression coverage.
    """

    def op_calc(self, tensor, dim):
        return torch.sum(tensor, dim)

    @parametrize('shape', TestUtils._pointwise_demo_shapes)
    @parametrize('dim', [0, -1])
    @parametrize('dtype', ['float32'])
    def test_sum_with_dim(self, shape, dim, dtype):
        element = self._generate_tensor(shape, dtype)
        std_sum = self.op_calc(element, dim)
        compiled_op_calc = torch.compile(self.op_calc, backend="inductor")
        inductor_sum = compiled_op_calc(element, dim)
        self.assertEqual(std_sum, inductor_sum, atol=1e-3, rtol=1e-3)

    @parametrize('shape', TestUtils._pointwise_demo_shapes)
    @parametrize('dim', [0, -1])
    @parametrize('dtype', ['float32'])
    @parametrize('dynamic,keepdim', [(True, False), (False, True), (True, True)])
    def test_sum_with_dim_keepdim_dynamic(self, shape, dim, dtype, dynamic, keepdim):
        element = self._generate_tensor(shape, dtype)

        def fn(tensor):
            return torch.sum(tensor, dim, keepdim=keepdim)

        compiled = torch.compile(fn, backend="inductor", fullgraph=True, dynamic=dynamic)
        self.assertEqual(fn(element), compiled(element), atol=1e-3, rtol=1e-3)


@unittest.skipUnless(
    _BACKEND in (None, "default"),
    "make_reduction compatibility tests require the default (Triton) sub-backend",
)
class TestReductionFactoryCompat(TestCase):
    """Check keyword compatibility without requiring a strict NPU kernel."""

    @staticmethod
    def create_api(name):
        def legacy(**kwargs):
            pass

        def strict_sum(*, strict_sum=False, **kwargs):
            pass

        def strict_reduction(*, strict_reduction=False, **kwargs):
            pass

        return locals()[name]

    @parametrize('version', [(2, 13), (2, 14), (2, 15)])
    def test_version_resolution(self, version):
        from torch_npu._compat import _impl, inductor

        with mock.patch.object(_impl, 'CURRENT_VERSION', version):
            adapter = _impl.compat_impl_container['compat_wrap_make_reduction'].resolve()
        expected = (inductor.compat_wrap_make_reduction_upstream if version >= (2, 15)
                    else inductor.compat_wrap_make_reduction_local)
        self.assertIs(adapter, expected)

    @parametrize('keyword', ['strict_sum', 'strict_reduction'])
    def test_entry_signature(self, keyword):
        from torch_npu._compat import inductor

        adapter = (inductor.compat_wrap_make_reduction_local if keyword == 'strict_sum'
                   else inductor.compat_wrap_make_reduction_upstream)
        implementation = mock.Mock(return_value=mock.sentinel.inner)
        factory = adapter(implementation)
        self.assertIs(inspect.signature(factory).parameters[keyword].default, False)
        self.assertIs(factory('sum'), mock.sentinel.inner)
        implementation.assert_called_once_with('sum', None, strict_reduction=False)
        for value in [False, True]:
            implementation.reset_mock()
            self.assertIs(
                factory('sum', mock.sentinel.dtype, **{keyword: value}),
                mock.sentinel.inner,
            )
            implementation.assert_called_once_with(
                'sum', mock.sentinel.dtype, strict_reduction=value
            )
        implementation.reset_mock()
        wrong = 'strict_reduction' if keyword == 'strict_sum' else 'strict_sum'
        with self.assertRaises(TypeError):
            factory('sum', **{wrong: False})
        implementation.assert_not_called()

    @parametrize('create_api', ['legacy', 'strict_sum', 'strict_reduction'])
    @parametrize('value', [False, True])
    @parametrize('reduction_type', ['sum', 'prod'])
    def test_create_kwargs(self, create_api, value, reduction_type):
        from torch_npu._compat.inductor import compat_reduction_create_kwargs

        unsupported = value and (
            create_api == 'legacy' or (create_api == 'strict_sum' and reduction_type == 'prod')
        )
        if unsupported:
            with self.assertRaises(NotImplementedError):
                compat_reduction_create_kwargs(self.create_api(create_api), reduction_type, value)
        else:
            expected = {} if create_api == 'legacy' else {create_api: value}
            self.assertEqual(
                compat_reduction_create_kwargs(self.create_api(create_api), reduction_type, value),
                expected,
            )

    @parametrize('create_api', ['legacy', 'strict_sum', 'strict_reduction'])
    @parametrize('value', [False, True])
    @parametrize('dump_fx_graph', [False, True])
    @parametrize('keyword', ['strict_sum', 'strict_reduction'])
    def test_strict_keyword_forwarding(self, create_api, value, dump_fx_graph, keyword):
        from torch_npu._compat import inductor
        from torch_npu._inductor import lowering as npu_lowering

        adapter = (inductor.compat_wrap_make_reduction_local if keyword == 'strict_sum'
                   else inductor.compat_wrap_make_reduction_upstream)
        make_reduction = adapter(npu_lowering._make_reduction)

        result = SimpleNamespace(data=SimpleNamespace(data=None))
        create = mock.Mock(return_value=result)
        create.__signature__ = inspect.signature(self.create_api(create_api))
        with mock.patch.object(
            npu_lowering.npu_config, 'dump_fx_graph', dump_fx_graph
        ), mock.patch.object(
            npu_lowering, '_make_reduction_inner', return_value={}
        ), mock.patch.object(npu_lowering.Reduction, 'create', create):
            if create_api == 'legacy' and value:
                with self.assertRaises(NotImplementedError):
                    make_reduction('sum', **{keyword: value})
                create.assert_not_called()
                return
            factory = make_reduction('sum', **{keyword: value})
            expected = {'reduction_type': 'sum', 'input_node': mock.sentinel.input}
            if create_api != 'legacy':
                expected[create_api] = value
            if dump_fx_graph:
                with mock.patch.object(npu_lowering, 'fetch_graphs', create=True), mock.patch.object(
                    npu_lowering, 'merge_traced_graphs', create=True,
                    return_value=mock.sentinel.graph,
                ), mock.patch.object(
                    npu_lowering, 'reduction_type_to_aten_fn', {'sum': mock.sentinel.sum}, create=True
                ):
                    self.assertIs(factory(mock.sentinel.input, axis=1), result)
                expected.update(node_name=create.call_args.kwargs['node_name'],
                                traced_graph=mock.sentinel.graph)
            else:
                self.assertIs(factory(mock.sentinel.input, axis=1), result)
            create.assert_called_once_with(**expected)


instantiate_parametrized_tests(TestReductionCompile)
instantiate_parametrized_tests(TestReductionFactoryCompat)

if __name__ == "__main__":
    run_tests()
